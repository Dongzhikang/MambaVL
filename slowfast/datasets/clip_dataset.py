import csv
import glob
import os.path as osp
import pickle
import random
import numpy as np
import pandas as pd
import torch
import spacy
import os

import decord
from transformers import RobertaTokenizer

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
from .build import DATASET_REGISTRY

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

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    frame_ids = np.convolve(np.linspace(start_frame, end_frame, num_segments + 1), [0.5, 0.5], mode='valid')
    if jitter:
        seg_size = float(end_frame - start_frame - 1) / num_segments
        shift = (np.random.rand(num_segments) - 0.5) * seg_size
        frame_ids += shift
    return frame_ids.astype(int).tolist()


def get_video_reader(videoname, num_threads, fast_rrc, rrc_params, fast_rcc, rcc_params):
    video_reader = None
    if fast_rrc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rrc_params[0], height=rrc_params[0],
            use_rrc=True, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
        )
    elif fast_rcc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rcc_params[0], height=rcc_params[0],
            use_rcc=True,
        )
    else:
        video_reader = decord.VideoReader(videoname, num_threads=num_threads)
    return video_reader


def video_loader(root, vid, ext, second, end_second,
                 chunk_len=300, fps=30, clip_length=32,
                 threads=1,
                 fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False, rcc_params=(224, ),
                 jitter=False):
    assert fps > 0, 'fps should be greater than 0'

    if chunk_len == -1:
        vr = get_video_reader(
            osp.join(root, '{}.{}'.format(vid, ext)),
            num_threads=threads,
            fast_rrc=fast_rrc, rrc_params=rrc_params,
            fast_rcc=fast_rcc, rcc_params=rcc_params,
        )
        end_second = min(end_second, len(vr) / fps)

        # calculate frame_ids
        frame_offset = int(np.round(second * fps))
        total_duration = max(int((end_second - second) * fps), clip_length)
        frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)

        # load frames
        assert max(frame_ids) < len(vr)
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    
        return torch.from_numpy(frames.astype(np.float32))

    else:
        chunk_start = int(second) // chunk_len * chunk_len
        chunk_end = int(end_second) // chunk_len * chunk_len
        while True:
            video_filename = osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk_end, ext))
            if not osp.exists(video_filename):
                # print("{} does not exists!".format(video_filename))
                chunk_end -= chunk_len
            else:
                vr = decord.VideoReader(video_filename)
                end_second = min(end_second, (len(vr) - 1) / fps + chunk_end)
                assert chunk_start <= chunk_end
                break
        # calculate frame_ids
        frame_ids = get_frame_ids(
            int(np.round(second * fps)),
            int(np.round(end_second * fps)),
            num_segments=clip_length, jitter=jitter
        )
        all_frames = []
        # allocate absolute frame-ids into the relative ones
        for chunk in range(chunk_start, chunk_end + chunk_len, chunk_len):
            rel_frame_ids = list(filter(lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps), frame_ids))
            rel_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_frame_ids]
            vr = get_video_reader(
                osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk, ext)),
                num_threads=threads,
                fast_rrc=fast_rrc, rrc_params=rrc_params,
                fast_rcc=fast_rcc, rcc_params=rcc_params,
            )
            try:
                frames = vr.get_batch(rel_frame_ids).asnumpy()
            except decord.DECORDError as error:
                print(error)
                frames = vr.get_batch([0] * len(rel_frame_ids)).asnumpy()
            except IndexError:
                print(root, vid, ext, second, end_second)
            all_frames.append(frames)
            if sum(map(lambda x: x.shape[0], all_frames)) == clip_length:
                break
        res = torch.from_numpy(np.concatenate(all_frames, axis=0).astype(np.float32))
        assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids)
        return res


class VideoCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata, is_trimmed=True):
        self.dataset = dataset
        self.root = root
        self.metadata = metadata
        self.is_trimmed = is_trimmed
        
        self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

        if self.dataset == 'ego4d':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        elif self.dataset in ['ek100_cls', 'ek100_mir']:
            video_list = glob.glob(osp.join(self.root, '*/*.MP4'))
            fps_dict = {video: decord.VideoReader(video + '/0.MP4').get_avg_fps() for video in video_list}
            self.samples = []
            # with open(metadata) as f:
            #     csv_reader = csv.reader(f)
            #     _ = next(csv_reader)  # skip the header
            #     for ind, row in enumerate(csv_reader):
            #         pid, vid = row[1:3]
            #         start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
            #         narration = row[8]
            #         verb, noun = int(row[10]), int(row[12])
            #         vid_path = '{}/{}'.format(pid, vid)
            #         fps = fps_dict[osp.join(self.root, vid_path + '.MP4')]
            #         # start_frame = int(np.round(fps * start_timestamp))
            #         # end_frame = int(np.ceil(fps * end_timestamp))
            #         self.samples.append((vid_path, start_timestamp, end_timestamp, fps, narration, verb, noun, ind))
            data = pd.read_pickle(metadata)
            for ind, _row in enumerate(data.iterrows()):
                row = _row[1]
                pid, vid = row[0:2]
                start_timestamp, end_timestamp = datetime2sec(row[3]), datetime2sec(row[4])
                narration = row[7]
                verb, noun = int(row[9]), int(row[11])
                verb_text, noun_text = row[8], row[10]
                noun_question, verb_question = row[-2], row[-1]
                vid_path = '{}/{}'.format(pid, vid)
                fps = fps_dict[osp.join(self.root, vid_path + '.MP4')]
                # start_frame = int(np.round(fps * start_timestamp))
                # end_frame = int(np.ceil(fps * end_timestamp))
                self.samples.append((vid_path, start_timestamp, end_timestamp, fps, narration, verb, noun, ind, verb_text, noun_text, verb_question, noun_question))
            if self.dataset == 'ek100_mir':
                self.metadata_sentence = pd.read_csv(metadata[:metadata.index('.csv')] + '_sentence.csv')
                if 'train' in metadata:
                    self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_train.pkl'), 'rb'))
                elif 'test' in metadata:
                    self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_test.pkl'), 'rb'))
                else:
                    raise ValueError('{} should contain either "train" or "test"!'.format(metadata))
                self.relevancy = .1
        else:
            raise NotImplementedError

    def get_raw_item(
        self, i, is_training=True, num_clips=1,
        chunk_len=300, clip_length=32, clip_stride=2,
        sparse_sample=False,
        narration_selection='random',
        threads=1,
        fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False, rcc_params=(224,),
    ):
        if self.dataset == 'ego4d':
            vid, start_second, end_second, narration = self.samples[i][:4]
            frames = video_loader(self.root, vid, 'mp4',
                                  start_second, end_second,
                                  chunk_len=chunk_len,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            if isinstance(narration, list):
                if narration_selection == 'random':
                    narration = random.choice(narration)
                elif narration_selection == 'concat':
                    narration = '. '.join(narration)
                elif narration_selection == 'list':
                    pass
                else:
                    raise ValueError
            return frames, narration
        elif self.dataset == 'ek100_mir':
            vid_path, start_second, end_second, fps, narration, verb, noun = self.samples[i]
            frames = video_loader(self.root, vid_path, 'MP4',
                                  start_second, end_second,
                                  chunk_len=chunk_len, fps=fps,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            if is_training:
                positive_list = np.where(self.relevancy_mat[i] > self.relevancy)[0].tolist()
                if positive_list != []:
                    pos = random.sample(positive_list, min(len(positive_list), 1))[0]
                    if pos < len(self.metadata_sentence) and pos < self.relevancy_mat.shape[1]:
                        return frames, (self.metadata_sentence.iloc[pos][1], self.relevancy_mat[i][pos])
            else:
                return frames, (narration, 1)
        elif self.dataset == 'ek100_cls':
            vid_path, start_second, end_second, fps, narration, verb, noun, ind, verb_text, noun_text, verb_question, noun_question = self.samples[i]
            if self.cfg.TRAIN.DATASET == "epickitchens" and self.cfg.EPICKITCHENS.ANTICIPATION:
                end_second = start_second - fps / self.cfg.DATA.TARGET_FPS
                start_idx = end_idx - clip_length * self.cfg.DATA.SAMPLING_RATE * fps / self.cfg.DATA.TARGET_FPS
            frames = video_loader(self.root, vid_path, 'MP4',
                                  start_second, end_second,
                                  chunk_len=chunk_len, fps=fps,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            noun_question = mask_given_phrase(noun_question, verb_text.lower().replace("-", " "))
            verb_question = mask_given_phrase(verb_question, noun_text.lower().replace("-", " "))
            noun_masked_ids = self.tokenizer(str(noun_question), return_tensors='pt', padding="max_length", truncation=True, max_length=40)['input_ids'].squeeze(0)
            verb_masked_ids = self.tokenizer(str(verb_question), return_tensors='pt', padding="max_length", truncation=True, max_length=40)['input_ids'].squeeze(0)
        
            metadata = {}
            # For generic questions
            # noun_masked_ids = self.tokenizer("What is the object in the video?", return_tensors='pt', padding="max_length", truncation=True, max_length=40)['input_ids'].squeeze(0)
            # verb_masked_ids = self.tokenizer("What action is performed using hands in the video?", return_tensors='pt', padding="max_length", truncation=True, max_length=40)['input_ids'].squeeze(0)
        
            metadata['noun_masked_ids'] = noun_masked_ids
            metadata['verb_masked_ids'] = verb_masked_ids
            
            # label = {'verb': verb, 'noun': noun, 'verb_question':verb_question, 'noun_question': noun_question, 'verb_text': verb_text, 'noun_text': noun_text}
            label = {'verb': verb, 'noun': noun}
        
            return frames, label, ind, metadata
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.samples)
    
    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.samples)


class VideoCaptionDatasetCLIP(VideoCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None,
                 is_training=True, tokenizer=None,
                 chunk_len=300,
                 clip_length=32, clip_stride=2,
                 threads=1,
                 fast_rrc=False,
                 rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False,
                 rcc_params=(224,),
                 subsample_stride=None):
        super().__init__(dataset, root, metadata)

        self.full_samples = self.samples.copy()
        if isinstance(subsample_stride, int):
            self.samples = self.samples[::subsample_stride]
        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params

    def __getitem__(self, i):
        frames, caption = self.get_raw_item(
            i, is_training=self.is_training,
            chunk_len=self.chunk_len,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
        )

        # ek100_mir will also output relevancy value
        if isinstance(caption, tuple):
            caption, relevancy = caption
        else:
            relevancy = 0.

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)[0]

        if isinstance(caption, tuple):
            caption, mask = caption
            return frames, caption, mask, relevancy
        else:
            return frames, caption, relevancy


class VideoClassyDataset(VideoCaptionDatasetBase):
    def __init__(
        self, dataset, root, metadata, transform=None,
        is_training=True, label_mapping=None,
        num_clips=1,
        chunk_len=300,
        clip_length=32, clip_stride=2,
        threads=1,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False,
        rcc_params=(224,),
        sparse_sample=False,
        is_trimmed=True,
        cfg=None):
        super().__init__(dataset, root, metadata, is_trimmed=is_trimmed)

        self.transform = transform
        self.is_training = is_training
        self.label_mapping = label_mapping
        self.num_clips = num_clips
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params
        self.sparse_sample = sparse_sample
        self.cfg = cfg

    def __getitem__(self, i):
        frames, label, ind, metadata = self.get_raw_item(
            i, is_training=self.is_training,
            chunk_len=self.chunk_len,
            num_clips=self.num_clips,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
            sparse_sample=self.sparse_sample,
        )

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        if self.label_mapping is not None:
            if isinstance(label, list):
                # multi-label case
                res_array = np.zeros(len(self.label_mapping))
                for lbl in label:
                    res_array[self.label_mapping[lbl]] = 1.
                label = res_array
            else:
                label = self.label_mapping[label]

        return frames, label, ind, metadata


from slowfast.models.avion.data.transforms import Permute
import torchvision
import torchvision.transforms._transforms_video as transforms_video
@DATASET_REGISTRY.register()
def Epickitchensavion(cfg, mode, label_mapping=None):
    mean, std = [108.3272985, 116.7460125, 104.09373615000001], [68.5005327, 66.6321579, 70.32316305]
    base_train_transform_ls = [
            Permute([3, 0, 1, 2]),
            torchvision.transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms_video.NormalizeVideo(mean=mean, std=std),
        ]
    base_val_transform_ls = [
            Permute([3, 0, 1, 2]),
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            transforms_video.NormalizeVideo(mean=mean, std=std),
        ]
    train_transform = torchvision.transforms.Compose(base_train_transform_ls)
    val_transform = torchvision.transforms.Compose(base_val_transform_ls)
    if mode == 'train':
        return VideoClassyDataset(
            "ek100_cls", os.path.join(cfg.EPICKITCHENS.VISUAL_DATA_DIR,"EK100_320p_15sec_30fps_libx264"), os.path.join(cfg.EPICKITCHENS.ANNOTATIONS_DIR, cfg.EPICKITCHENS.TRAIN_LIST), train_transform,
            is_training=True, label_mapping=label_mapping,
            num_clips=1,
            chunk_len=15,
            clip_length=16, clip_stride=2,
            threads=1,
            fast_rrc=False, 
            rrc_params=(224, (0.5, 1.0)),
            cfg=cfg,
        )
    elif mode == 'val':
        return VideoClassyDataset(
            "ek100_cls", os.path.join(cfg.EPICKITCHENS.VISUAL_DATA_DIR,"EK100_320p_15sec_30fps_libx264"), os.path.join(cfg.EPICKITCHENS.ANNOTATIONS_DIR, cfg.EPICKITCHENS.VAL_LIST), val_transform,
            is_training=False, label_mapping=label_mapping,
            num_clips=1,
            chunk_len=15,
            clip_length=16, clip_stride=2,
            threads=1,
            fast_rrc=False, rcc_params=(224, ),
            is_trimmed=True,
            cfg=cfg
        )
    else:
        assert ValueError("subset should be either 'train' or 'val'")
