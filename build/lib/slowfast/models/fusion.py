import torch.nn as nn
import torch
from functools import partial
from timm.models.layers import trunc_normal_

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # add the mask to enable cross attention
        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class CrossAttention(nn.Module):
    """Cross attention used in transformer decoder"""
    def __init__(self, dim, mem_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        mem_dim = mem_dim or dim
        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(mem_dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(mem_dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mem, attn_mask=None):
        B, N, C = x.shape

        # calculate query, key, value for all heads
        q = self.w_q(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, N, c)
        k = self.w_k(mem).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, N, c)
        v = self.w_v(mem).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, N, c)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # add the mask to enable "causality"
        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None):
        attn_output, attn_weights = self.attn(self.norm1(x), attn_mask)
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_weights


class DecoderBlock(nn.Module):
    """Transformer decoder block with pre-layernorm"""
    def __init__(self, dim, mem_dim=None, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_self = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.cross_attn = CrossAttention(
            dim, mem_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(mem_dim or dim)
        self.norm_mlp = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mem, attn_mask=None):
        attn_output, _ = self.attn(self.norm_self(x), attn_mask)  # attention weights will be ignored
        x = x + self.drop_path(attn_output)  # self attention + short-cut
        x = x + self.drop_path(self.cross_attn(self.norm_q(x), self.norm_kv(mem), attn_mask))  # cross attention + short-cut
        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))  # mlp
        return x
   
def _init_weights(m):
    # Copied from Timm VisionTransformer,
    # removing init for layernorm, since this init is already the default for pytorch
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
 
class Fuser(nn.Module):
    def __init__(self, dim, depth=6, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, embd_drop_rate=0.1, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1, act_layer=nn.GELU, norm_elementwise=True):
        super().__init__()
        # move norm_layer here, so that elementwise affine can be controlled by hydra
        norm_layer = partial(nn.LayerNorm, eps=1e-6, elementwise_affine=norm_elementwise)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer
            ) for i in range(depth)
        ])
        # self.v_norm = norm_layer(dim)
        # self.t_norm = norm_layer(dim)

        self.num_mods = 2 + 1  # add 1 since we add a modal token
        self.embd_drop = nn.Dropout(embd_drop_rate)

        # modality agnostic token
        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_token_text = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed_video = nn.Parameter(torch.zeros(1, 5000, dim))
        self.pos_embed_text = nn.Parameter(torch.zeros(1, 5000, dim))
        self.type_video = nn.Parameter(torch.zeros(1, 1, dim))
        self.type_text = nn.Parameter(torch.zeros(1, 1, dim))

        trunc_normal_(self.cls_token_video, std=.02)
        trunc_normal_(self.cls_token_text, std=.02)
        trunc_normal_(self.pos_embed_video, std=.02)
        trunc_normal_(self.pos_embed_text, std=.02)
        trunc_normal_(self.type_video, std=.02)
        trunc_normal_(self.type_text, std=.02)
        self.apply(_init_weights)

    def forward(self, v_embeds, t_embeds):
        """
        :param modal_feats: {'modality': feature vector}
        :param ordered_feature_list: a function to convert a dict to a list with a specific order
        :return: fused feature
        """
        attn_mask = None
        B = v_embeds.shape[0]

        # n * (B, T, C) -> (B*T, n, C)
        # Prepending with all the tokens
        v_embeds = torch.cat([self.cls_token_video.expand(B, -1, -1), v_embeds], dim=1)
        t_embeds = torch.cat([self.cls_token_text.expand(B, -1, -1), t_embeds], dim=1)
        
        B, N_video, C = v_embeds.shape
        N_text = t_embeds.shape[1]
        v_embeds = v_embeds + self.pos_embed_video[:, :N_video, :] + self.type_video
        t_embeds = t_embeds + self.pos_embed_text[:, :N_text, :] + self.type_text
        
        for_fusion_feats = torch.cat((v_embeds, t_embeds), dim=1)

        # prepare modal token
        # modal_tokens = self.modal_token.expand(B * T, -1, -1)

        # prepend the modality agnostic token
        # for_fusion_feats = torch.cat((modal_tokens, for_fusion_feats), dim=1)

        # fusion part
        x = self.embd_drop(for_fusion_feats)
        attn_weights = []
        for blk in self.blocks:
            x, attn_weight = blk(x, attn_mask)
            attn_weights.append(attn_weight.view(B, *attn_weight.shape[1:]))

        cls_v = x[:, 0, :].view(B, -1, x.shape[-1])
        video = x[:, 1:N_video, :]
        cls_t = x[:, N_video, :].view(B, -1, x.shape[-1])
        text = x[:, N_video+1:, :]
        # video = self.v_norm(video)
        # text = self.t_norm(text)
        
        return cls_v