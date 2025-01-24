#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import traceback
import numpy as np
import random
import os
import pandas as pd
import torch
import torch.utils.data
from torchvision import transforms
import json

import slowfast.utils.logging as logging
from slowfast.utils.box_ops import box_xyxy_to_cxcywh, zero_empty_boxes
from slowfast.utils import box_ops

from .build import DATASET_REGISTRY
from .ek_MF.epickitchens_record import EpicKitchensVideoRecord, sample_portion_from_data

# from . import autoaugment as autoaugment
from . import transform as transform
from . import utils as utils
from .ek_MF.frame_loader import pack_frames_to_video_clip

from .transform import create_random_augment
from .random_erasing import RandomErasing

import pickle
import os

from transformers import RobertaTokenizer
import spacy

logger = logging.get_logger(__name__)


# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def mask_given_phrase(text, phrase_to_mask):
    doc = nlp(text)
    masked_text = []
    skip_next = 0

    phrase_to_mask = phrase_to_mask.lower().split()

    for i, token in enumerate(doc):
        if skip_next > 0:
            skip_next -= 1
            continue

        # Check if the phrase starts here
        if token.lemma_ == phrase_to_mask[0]:
            match = True
            for j in range(1, len(phrase_to_mask)):
                if i + j >= len(doc) or doc[i + j].lemma_ != phrase_to_mask[j]:
                    match = False
                    break
            if match:
                masked_text.append("<mask>")
                skip_next = len(phrase_to_mask) - 1
                continue

        # Default case: just add the token's text
        masked_text.append(token.text)
    
    return " ".join(masked_text)



@DATASET_REGISTRY.register()
def Epickitchens(cfg, mode):
    if cfg.ORVIT.ENABLE:
        from slowfast.utils.LinkBoxes.epickitchens import get_ek_boxes
        dboxes = get_ek_boxes(cfg.EPICKITCHENS.VISUAL_DATA_DIR, verbose=True, h5=True)
    else:
        dboxes = None
    if cfg.MODEL.MODEL_NAME == "AVION1":
        return Epickitchens_dataset1(cfg, mode, orvit_boxes=dboxes)
    else:
        return Epickitchens_dataset(cfg, mode, orvit_boxes=dboxes)

def read_txt_into_list(file_path):
    """
    Reads a text file and returns a list of entries. Each entry is separated by double newlines.

    Parameters:
        file_path (str): The path to the text file.

    Returns:
        List[str]: A list of entries from the text file.
    """
    with open(file_path, 'r') as file:
        # Read the whole file and split it by double newlines
        data_list = file.read().split('\n\n')
    
    # Clean up any extra whitespace or empty entries
    data_list = [entry.strip() for entry in data_list if entry.strip()]
    
    return data_list

class Epickitchens_dataset(torch.utils.data.Dataset):

    def __init__(self, cfg, mode, orvit_boxes=None):

        assert mode in [
            "train",
            "val",
            "test",
            "train+val"
        ], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        self.cfg = cfg
        self.mode = mode
        self.target_fps = cfg.DATA.TARGET_FPS
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        self.get_orvit_boxes = self.cfg.ORVIT.ENABLE
        self.actions_csv = pd.read_csv(os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.ACTIONS_LIST))
        self.actions_gpt_csv = pd.read_csv(os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, "actions_gpt.csv"))

        if getattr(self, 'get_orvit_boxes', False):
            from .ek_MF.epickitchens_record import EKBoxes
            self.ek_boxes = EKBoxes(cfg, boxes=orvit_boxes)
        logger.info("Constructing EPIC-KITCHENS {}...".format(mode))
        temp_video_record = None
        if cfg.TRAIN.DATASET == "epickitchens" and cfg.EPICKITCHENS.ANTICIPATION:
            mode = self.mode
            _num_clips = self._num_clips
            self.mode = "train"
            self._num_clips = 1
            self._construct_loader()
            self.qns = {}
            for rec in self._video_records:
                _verb = rec.label["verb"]
                _noun = rec.label["noun"]
                _action = self.actions_csv[(self.actions_csv['verb'] == _verb) & (self.actions_csv['noun'] == _noun)]["action"].item()
                if f"{_verb} {_noun}" not in self.qns.keys():
                    self.qns[f"{_action}"] = [{"verb_question": rec.label["verb_question"], "noun_question": rec.label["noun_question"]}]
                else:
                    self.qns[f"{_action}"].append({"verb_question": rec.label["verb_question"], "noun_question": rec.label["noun_question"]})
            self.mode = "test"
            self._num_clips = 1
            self._construct_loader()
            for rec in self._video_records:
                _verb = rec.label["verb"]
                _noun = rec.label["noun"]
                _action = self.actions_csv[(self.actions_csv['verb'] == _verb) & (self.actions_csv['noun'] == _noun)]["action"].item()
                if f"{_verb} {_noun}" not in self.qns.keys():
                    self.qns[f"{_action}"] = [{"verb_question": rec.label["verb_question"], "noun_question": rec.label["noun_question"]}]
                else:
                    self.qns[f"{_action}"].append({"verb_question": rec.label["verb_question"], "noun_question": rec.label["noun_question"]})
                
            self.mode = mode
            self._num_clips = _num_clips
        self._construct_loader()

        self.aug = False
        self.rand_erase = False
        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True
        # add tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
        
        if cfg.TRAIN.DATASET == "epickitchens" and cfg.EPICKITCHENS.ANTICIPATION:
            if self.mode == "train":
                self.actions = json.load(open("/nethome/abeedu3/InAViT/actions/predicted_actions_train.json", 'r'))
            elif (self.mode == "val") or (self.mode == "test"):
                self.actions = json.load(open("/nethome/abeedu3/InAViT/actions/predicted_actions_val.json", 'r'))

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        train_filename = self.cfg.EPICKITCHENS.TRAIN_LIST
        val_filename = self.cfg.EPICKITCHENS.VAL_LIST

        if self.mode == "train":
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, train_filename)]
        elif self.mode == "val":
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, val_filename)]
        elif self.mode == "test": ## Dong: add test file!
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, val_filename)]
                # os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TEST_LIST)]
        else:
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, file)
                    for file in [train_filename, val_filename]]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(
                file
            )

        self._video_records = []
        self._spatial_temporal_idx = []
        self.index_map = {}
        for file in path_annotations_pickle:
            for ind, tup in enumerate(pd.read_pickle(file).iterrows()):
                for idx in range(self._num_clips):
                    self.index_map[(ind*self._num_clips)+idx] = ind
                    rec = EpicKitchensVideoRecord(tup, self.actions_csv)
                    self._video_records.append(rec)
                    self._spatial_temporal_idx.append(idx)
        assert (
                len(self._video_records) > 0
        ), "Failed to load EPIC-KITCHENS split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing epickitchens dataloader (size: {}) from {}".format(
                len(self._video_records), path_annotations_pickle
            )
        )

    def __getitem__(self, index):
        while True:
            try:
                return self.getitem(index)
            except Exception as e:
                logger.warn(f"EK dataloader: {traceback.format_exc()}")
            index = (index+1)%self.__len__()
    def getitem(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        frames, seq = pack_frames_to_video_clip(self.cfg, self._video_records[index], temporal_sample_index, target_fps=self.target_fps, ret_seq = True)
        frames = torch.as_tensor(frames) # [T, H, W, C]
        nid = self._video_records[index].metadata['narration_id']
        if getattr(self, 'get_orvit_boxes', False):
            boxes = self.ek_boxes.get_boxes(self._video_records[index].untrimmed_video_name, seq.tolist(), nid = nid)
        else:
            boxes = None


        # Augmentations
        if self.aug:
            frames = self._aug_frame(
                frames,
                spatial_sample_index,
                min_scale,
                max_scale,
                crop_size,
                boxes = boxes,
            ) # [C, T, H, W] 
            if boxes is not None: frames, boxes = frames
        else:
            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = utils.spatial_sampling(
                frames, boxes=boxes,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )
            if boxes is not None: frames, boxes = frames
        
        label = self._video_records[index].label
        frames = utils.pack_pathway_output(self.cfg, frames)
        metadata = self._video_records[index].metadata
        if boxes is not None:
            # boxes[boxes < 0] = 0
            # boxes[boxes > 1] = 1
            # boxes = box_xyxy_to_cxcywh(boxes.transpose([1,0,2])) # T, O, 4
            # boxes = zero_empty_boxes(boxes, mode='cxcywh', eps = 0.05)
            boxes = torch.from_numpy(boxes)
            boxes = self.ek_boxes.prepare_boxes(boxes, nid = nid)
            # metadata['orvit_bboxes'] = torch.tensor(boxes)
            metadata['orvit_bboxes'] = boxes.clone().detach()

        # save_test_path = f'/srv/essa-lab/flash3/jsheinkopf3/ORViT/slowfast/datasets/get_item_test/save_{index}.pt'

        # if not os.path.exists(save_test_path):
        #     test_dict = {
        #     'frames': frames,
        #     'label': label,
        #     'index': index,
        #     'metadata': metadata
        #     }

        #     torch.save(test_dict, save_test_path)
        #     print(f'Saved {index}')
        # label has keys: 'verb', 'noun', 'verb_text', 'noun_text', 'gpt_desc'
        # sentence = label['gpt_desc']
        noun_question = label['noun_question']
        noun = label['noun_text']

        verb_question = label['verb_question']
        verb = label['verb_text']
        # For evaluation
        # noun_question = noun_question.lower().replace(verb.lower().replace("-", " "), self.tokenizer.mask_token)
        # verb_question = verb_question.lower().replace(noun.lower().replace("-", " "), self.tokenizer.mask_token)
        if self.cfg.TRAIN.DATASET == "epickitchens" and self.cfg.EPICKITCHENS.ANTICIPATION:
            _index = self.index_map[index]
            _actions = list(set(self.actions[f'{_index}']))
            try:
                q_dict = random.choice(self.qns[_actions[0]])
                verb_question, noun_question = q_dict["verb_question"], q_dict["noun_question"]
            except:
                ind = self.actions_gpt_csv.index[self.actions_gpt_csv["Action"]==_actions[0]].item()
                verb_question = random.choice(list(self.actions_gpt_csv.iloc[ind]))
                noun_question = verb_question
            
        noun_question = mask_given_phrase(noun_question, verb.lower().replace("-", " "))
        verb_question = mask_given_phrase(verb_question, noun.lower().replace("-", " "))
        # print(f"noun: {noun}, verb: {verb}, noun_question: {noun_question}, verb_question: {verb_question}")
        

        noun_masked_ids = self.tokenizer(str(noun_question), return_tensors='pt', padding="max_length", truncation=True, max_length=40)['input_ids'].squeeze(0)
        verb_masked_ids = self.tokenizer(str(verb_question), return_tensors='pt', padding="max_length", truncation=True, max_length=40)['input_ids'].squeeze(0)
        # For generic questions
        # noun_masked_ids = self.tokenizer("What is the object in the video?", return_tensors='pt', padding="max_length", truncation=True, max_length=40)['input_ids'].squeeze(0)
        # verb_masked_ids = self.tokenizer("What action is performed using hands in the video?", return_tensors='pt', padding="max_length", truncation=True, max_length=40)['input_ids'].squeeze(0)

        metadata['noun_masked_ids'] = noun_masked_ids
        metadata['verb_masked_ids'] = verb_masked_ids

        return frames, label, index, metadata


    def __len__(self):
        return len(self._video_records)

    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames


    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
        boxes=None,
    ):

        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
            with_boxes = boxes is not None,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2) # [T, C, H, W]
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img, boxes=boxes)
        if boxes is not None:
            list_img, boxes = list_img
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1) # [T, H, W, C]

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ["train"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train"] or len(asp) == 0) else asp
        )

        if boxes is not None:
            orig_shape = boxes.shape
            boxes = boxes.reshape([-1, 4])

        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
            boxes=boxes,
        )
        
        if boxes is not None:
            frames, boxes = frames
            boxes = boxes.reshape(orig_shape)
        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        if boxes is not None: return frames, boxes
        return frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._video_records)
    
    
class Epickitchens_dataset1(torch.utils.data.Dataset):

    def __init__(self, cfg, mode, orvit_boxes=None):

        assert mode in [
            "train",
            "val",
            "test",
            "train+val"
        ], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        self.cfg = cfg
        self.mode = mode
        self.target_fps = cfg.DATA.TARGET_FPS
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        self.get_orvit_boxes = self.cfg.ORVIT.ENABLE


        if getattr(self, 'get_orvit_boxes', False):
            from .ek_MF.epickitchens_record import EKBoxes
            self.ek_boxes = EKBoxes(cfg, boxes=orvit_boxes)
        logger.info("Constructing EPIC-KITCHENS {}...".format(mode))
        self._construct_loader()

        self.aug = False
        self.rand_erase = False
        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True
        # add tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        train_filename = self.cfg.EPICKITCHENS.TRAIN_LIST
        val_filename = self.cfg.EPICKITCHENS.VAL_LIST

        if self.mode == "train":
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, train_filename)]
        elif self.mode == "val":
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, val_filename)]
        elif self.mode == "test": ## Dong: add test file!
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, val_filename)]
                # os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TEST_LIST)]
        else:
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, file)
                    for file in [train_filename, val_filename]]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(
                file
            )

        self._video_records = []
        self._spatial_temporal_idx = []
        for file in path_annotations_pickle:
            for tup in pd.read_pickle(file).iterrows():
                for idx in range(self._num_clips):
                    rec = EpicKitchensVideoRecord(tup)
                    self._video_records.append(rec)
                    self._spatial_temporal_idx.append(idx)
        assert (
                len(self._video_records) > 0
        ), "Failed to load EPIC-KITCHENS split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing epickitchens dataloader (size: {}) from {}".format(
                len(self._video_records), path_annotations_pickle
            )
        )

    def __getitem__(self, index):
        while True:
            try:
                return self.getitem(index)
            except Exception as e:
                logger.warn(f"EK dataloader: {traceback.format_exc()}")
            index = (index+1)%self.__len__()
    def getitem(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        # frames, seq = pack_frames_to_video_clip(self.cfg, self._video_records[index], temporal_sample_index, target_fps=self.target_fps, ret_seq = True)
        # frames = torch.as_tensor(frames) # [T, H, W, C]
        nid = self._video_records[index].metadata['narration_id']
        
        
        label = self._video_records[index].label
        # frames = utils.pack_pathway_output(self.cfg, frames)
        metadata = self._video_records[index].metadata
        path = f"{self.cfg.DATA.PATH_TO_AVION}/{self.mode}/{index}.pkl"
        frames = pickle.load(open(path, "rb"))
        noun_question = label['noun_question']
        noun = label['noun_text']

        verb_question = label['verb_question']
        verb = label['verb_text']
        # For evaluation
        # noun_question = noun_question.lower().replace(verb.lower().replace("-", " "), self.tokenizer.mask_token)
        # verb_question = verb_question.lower().replace(noun.lower().replace("-", " "), self.tokenizer.mask_token)
        noun_question = mask_given_phrase(noun_question, verb.lower().replace("-", " "))
        verb_question = mask_given_phrase(verb_question, noun.lower().replace("-", " "))
        # print(f"noun: {noun}, verb: {verb}, noun_question: {noun_question}, verb_question: {verb_question}")

        noun_masked_ids = self.tokenizer(str(noun_question), return_tensors='pt', padding="max_length", truncation=True, max_length=40)['input_ids'].squeeze(0)
        verb_masked_ids = self.tokenizer(str(verb_question), return_tensors='pt', padding="max_length", truncation=True, max_length=40)['input_ids'].squeeze(0)
        # For generic questions
        # noun_masked_ids = self.tokenizer("What is the object in the video?", return_tensors='pt', padding="max_length", truncation=True, max_length=40)['input_ids'].squeeze(0)
        # verb_masked_ids = self.tokenizer("What action is performed using hands in the video?", return_tensors='pt', padding="max_length", truncation=True, max_length=40)['input_ids'].squeeze(0)

        metadata['noun_masked_ids'] = noun_masked_ids
        metadata['verb_masked_ids'] = verb_masked_ids

        return frames, label, index, metadata


    def __len__(self):
        return len(self._video_records)

    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames


    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
        boxes=None,
    ):

        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
            with_boxes = boxes is not None,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2) # [T, C, H, W]
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img, boxes=boxes)
        if boxes is not None:
            list_img, boxes = list_img
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1) # [T, H, W, C]

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ["train"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train"] or len(asp) == 0) else asp
        )

        if boxes is not None:
            orig_shape = boxes.shape
            boxes = boxes.reshape([-1, 4])

        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
            boxes=boxes,
        )
        
        if boxes is not None:
            frames, boxes = frames
            boxes = boxes.reshape(orig_shape)
        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        if boxes is not None: return frames, boxes
        return frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._video_records)