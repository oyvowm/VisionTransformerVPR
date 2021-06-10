"""
Taken from the official implementation of the pooling transformer: https://github.com/naver-ai/pit/blob/master/pit.py

Modified slightly for improved patch embedding and scaling of the residuals
"""

# PiT
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import math

from utils import to_tuple

from functools import partial
from model import TransformerBlock
from timm.models.layers import trunc_normal_

class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None, residual_scaling=1.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=heads,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                hidden_mult=mlp_ratio,
                residual_scaling=residual_scaling,
                drop_path=drop_path_prob[i],
                #norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)])

    def forward(self, x, cls_tokens):
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')

        token_length = cls_tokens.shape[1] # token length = number of cls tokens
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x, cls_tokens
    




class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size,
                              stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class PatchEmbedExtraConvolution(nn.Module):
    """ 
    Image to Patch Embedding with 4 layer convolution
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        proj_kernel, proj_stride = to_tuple(patch_size // 2) # to maintain the same number of patches

        img_size = to_tuple(img_size)
        patch_size = to_tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(64)
        #self.norm1 = nn.LayerNorm(64, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(64)
        #self.norm2 = nn.LayerNorm(64, eps=1e-6)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(64)
        #self.norm3 = nn.LayerNorm(64, eps=1e-6)
        
        self.proj = nn.Conv2d(64, embed_dim, kernel_size=proj_kernel, stride=(proj_stride//2)) # 27 x 27 as expected by pit
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        #x = self.norm1(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        #x = self.norm1(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x


class PoolingTransformer(nn.Module):
    def __init__(
                    self, 
                    img_size, 
                    patch_size, 
                    stride, 
                    base_dims, 
                    depth, 
                    heads,
                    mlp_ratio, 
                    num_classes=1000, 
                    in_chans=3,
                    attn_drop_rate=.0, 
                    drop_rate=.0, 
                    drop_path_rate=.0, 
                    triplet=False,
                    embed_fn='vit',
                    residual_scaling=1.,
                ):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        height = math.floor((img_size[0] + 2 * padding - patch_size) / stride + 1)
        width = math.floor((img_size[1] + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.triplet = triplet
        self.pos_embed = nn.Parameter(
            torch.randn(1, base_dims[0] * heads[0], height, width),
            requires_grad=True
        )
        self.initial_embed_dim = base_dims[0] * heads[0]
        if embed_fn == 'vit':
            self.patch_embed = conv_embedding(in_chans, self.initial_embed_dim,
                                          patch_size, stride, padding)
        elif embed_fn == 'convolution':
            self.patch_embed = PatchEmbedExtraConvolution(img_size, patch_size, in_chans, self.initial_embed_dim)

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.initial_embed_dim),
            requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], heads[stage],
                            mlp_ratio, drop_rate, attn_drop_rate, drop_path_prob, residual_scaling)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if not self.triplet:
            if num_classes > 0:
                self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
            else:
                self.head = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)

        if not self.triplet:
            cls_tokens = self.norm(cls_tokens)

        return cls_tokens

    def forward(self, x):
        cls_token = self.forward_features(x)
        if self.triplet:
            return cls_token
        cls_token = self.head(cls_token[:, 0])
        return cls_token


class DistilledPoolingTransformer(PoolingTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_token = nn.Parameter(
            torch.randn(1, 2, self.base_dims[0] * self.heads[0]),
            requires_grad=True)
        if not self.triplet:
            if self.num_classes > 0:
                self.head_dist = nn.Linear(self.base_dims[-1] * self.heads[-1],
                                           self.num_classes)
            else:
                self.head_dist = nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        if not self.triplet:
            self.head_dist.apply(self._init_weights)

    def forward(self, x):
        cls_token = self.forward_features(x)
        
        # have to make copies when normalizing to avoid runime error for autograd
        token1 = cls_token[:,0] 
        token2 = cls_token[:,1]
        if self.triplet:
            if self.training:
                token1 = F.normalize(token1, p=2, dim=-1)
                token2 = F.normalize(token2, p=2, dim=-1)
                return token1, token2
            else:
                return (token1 + token2) / 2
        
        x_cls = self.head(token1)
        x_dist = self.head_dist(token2)
        
        return (x_cls + x_dist) / 2



def pit_s(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_s_809.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model


def pit_s_distilled(pretrained, **kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_s_distill_819.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model

