'''
vision transformer with distillation based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''

import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_
from loss_functions import ArcFace
from utils import to_tuple

##### BUILDING BLOCKS FOR THE FULL TRANSFORMER #####

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_mult, dropout = 0.):
        super().__init__()
        self.hidden_mult = hidden_mult
        self.fc1 = nn.Linear(input_dim, hidden_mult * input_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_mult * input_dim, input_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, input_dim, num_heads = 8, dropout_attention = 0., dropout_unify = 0., qkv_bias = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(input_dim, input_dim * 3, bias = qkv_bias)
        self.dropout_attention = nn.Dropout(dropout_attention)
        self.proj = nn.Linear(input_dim, input_dim)
        self.dropout_proj = nn.Dropout(dropout_unify)
        
    def forward(self, x):
        batch_samples, num_tokens, embed_dim = x.shape
        # pass the input tokens through the linear layer and reshape the embedding dimension
        # into 3 * number of heads * dimension of each head
        qkv = self.qkv(x).reshape(batch_samples, num_tokens, 3, self.num_heads, embed_dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)
        
        attn = (q @ k_t) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.dropout_attention(attn)
        # transpose to switch the num_heads and num_tokens dimensions, so that num_heads and head_dim can be unified 
        # back into embed_dim
        x = (attn @ v).transpose(1, 2).reshape(batch_samples, num_tokens, embed_dim)
        x = self.proj(x)
        x = self.dropout_proj(x)
        return x
        
        
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias = False, drop = 0., attn_drop = 0., hidden_mult = 4, residual_scaling = 1., drop_path=0.):
        super().__init__()
        self.residual_scaling = residual_scaling
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = Attention(embed_dim, num_heads = num_heads, dropout_attention = attn_drop, dropout_unify = drop,
                                  qkv_bias = qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = FeedForward(embed_dim, hidden_mult = hidden_mult, dropout = drop)
        
    def forward(self, x):
        # potentially downscaling of the residual connections
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.residual_scaling
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.residual_scaling
        return x

        
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x) # gives a batch_size x embed_dim x H // patch size x W // patch size tensor
        x = x.flatten(2) # unifies the two last dimension into num_patches
        x = x.transpose(1,2) # transpose the tensor, giving num_batches, num_patches, embed_dim 
        return x

class PatchEmbedExtraConvolution(nn.Module):
    """ 
    Atttempted implementation of convolutional backbone 
    slightly modified version of: https://github.com/zihangJiang/TokenLabeling/blob/main/models/layers.py
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

        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        self.bn1 = nn.BatchNorm2d(64)
        #self.norm1 = nn.LayerNorm(64, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn2 = nn.BatchNorm2d(64)
        #self.norm2 = nn.LayerNorm(64, eps=1e-6)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(64)
        #self.norm3 = nn.LayerNorm(64, eps=1e-6)
        
        self.proj = nn.Conv2d(64, embed_dim, kernel_size=proj_kernel, stride=proj_stride)
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

        x = x.flatten(2) # unifies the two last dimension into num_patches
        x = x.transpose(1,2) # transpose the tensor, giving num_batches, num_patches, embed_dim

        return x

    
    
class VisionTransformer(nn.Module):
    """
        A slightly modified version of Ross Wightman's implementation:
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(
                self, 
                img_size=(224, 224), 
                patch_size=16, 
                in_chans=3, 
                num_classes=1000, 
                embed_dim=768, 
                triplet=False,
                depth=12,     
                num_heads=12, 
                hidden_mult=4., 
                qkv_bias=True, 
                drop=0., 
                attn_drop=0.,
                drop_path=0.,
                embed_fn='vit'
                ):

        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.triplet = triplet
        if embed_fn == 'vit':
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_chans, embed_dim=embed_dim)
        elif embed_fn == 'convolution':
            self.patch_embed = PatchEmbedExtraConvolution(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(TransformerBlock(embed_dim = embed_dim, num_heads = num_heads, qkv_bias = qkv_bias, drop = drop, 
                                                       attn_drop = attn_drop, hidden_mult = hidden_mult, drop_path=dpr[i]
                                          ))
        self.blocks = nn.Sequential(*blocks)
        

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Classifier head
        if not self.triplet:
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        

    
    def forward(self, x):
        '''
        input: 
            tensor of shape (batch_size, in_chans, img_height, img_width)
        output:
            logits of shape (batch_size, num_classes)
        '''
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1) # gives x: (batch_size, 1 + num_patches, embed_dim)
        x = x + self.pos_embed # pos_embed gets added to each batch
        x = self.pos_drop(x)
        
        x = self.blocks(x)
        x = self.norm(x)
        
        x = x[:, 0]
        if self.triplet:
            return x
        return self.head(x)
    
    # method that defines the keys for which no weight decay will occur
    def no_weight_decay(self):
        return {'cls_token' ,'pos_embed'}
    

    
    
    
    
class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        if not self.triplet:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
       
    
    def forward(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        dist_token = self.dist_token.expand(batch_size, -1, -1) 
        x = torch.cat((cls_tokens, dist_token, x), dim=1) # gives x: (batch_size, 2 + num_patches, embed_dim)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)

        if not self.triplet:
            x = self.norm(x)
        x_cls, x_dist = x[:, 0], x[:, 1]
        
        if self.triplet:
            if self.training:
                return x_cls, x_dist
            else:
                return (x_cls + x_dist) / 2 

        x_cls = self.head(x_cls)
        x_dist = self.head_dist(x_dist)
        
        return (x_cls + x_dist) / 2
    
    # method that defines the keys for which no weight decay will occur
    def no_weight_decay(self):
        return {'cls_token' ,'pos_embed', 'dist_token'}

class ArcFaceDeit(DistilledVisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = ArcFace(self.embed_dim, self.num_classes)
        self.head_dist = ArcFace(self.embed_dim, self.num_classes)
    
    def forward(self, x, labels):
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        dist_token = self.dist_token.expand(batch_size, -1, -1) 
        x = torch.cat((cls_tokens, dist_token, x), dim=1) # gives x: (batch_size, 2 + num_patches, embed_dim)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)


        x = self.norm(x)
        x_cls, x_dist = x[:, 0], x[:, 1]
        
        loss_cls, x_cls = self.head(x_cls, labels)
        loss_dist, x_dist = self.head_dist(x_dist, labels)
        
        # When training for place recognition a teacher model is not used, thus using the mode usually for inference by default
        #if self.training:
        #    return x_cls, x_dist
        
        return ((loss_cls + loss_dist) / 2), ((x_cls + x_dist) / 2)

    
class ArcFaceDeitEval(DistilledVisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Identity()
        self.head_dist = nn.Identity()
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        dist_token = self.dist_token.expand(batch_size, -1, -1) 
        x = torch.cat((cls_tokens, dist_token, x), dim=1) # gives x: (batch_size, 2 + num_patches, embed_dim)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)


        x = self.norm(x)
        x_cls, x_dist = x[:, 0], x[:, 1]
        
        #x_cls = self.head(x_cls)
        #x_dist = self.head_dist(x_dist)
        
        # When training for place recognition a teacher model is not used, thus using the mode usually for inference by default
        #if self.training:
        #    return x_cls, x_dist
        
        return (x_cls + x_dist) / 2
        

