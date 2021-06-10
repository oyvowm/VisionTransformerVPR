# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

"""
Implementation from https://github.com/facebookresearch/deit/blob/main/cait_models.py
Modified for triplet learning, and to allow the addition of extra tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import DropPath, trunc_normal_


class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    
    def forward(self, x ):
        
        B, N, C = x.shape
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        return x_cls    
    

class Class_Attention_twoQ(nn.Module):
    """
    Modified ClassAttention where a separate linear layer is used for each of the output tokens when projecting into queries. 
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    
    def forward(self, x ):
        
        B, N, C = x.shape
        q = self.q(x[:,0]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q2 = self.q2(x[:,1]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        qs = torch.cat((q, q2), dim=2)
        qs = qs * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (qs @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 2, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        return x_cls
    
        
class LayerScale_Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,
                 Mlp_block=Mlp,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    
    def forward(self, x, x_cls):
        
        u = torch.cat((x_cls,x),dim=1)
        
        
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        
        return x_cls 
        
        
class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.num_heads = num_heads
        
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)
        
        self.proj_drop = nn.Dropout(proj_drop)


    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale , qkv[1], qkv[2] 
    
        attn = (q @ k.transpose(-2, -1)) 
        
        attn = self.proj_l(attn.permute(0,2,3,1)).permute(0,3,1,2)
                
        attn = attn.softmax(dim=-1)
  
        attn = self.proj_w(attn.permute(0,2,3,1)).permute(0,3,1,2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LayerScale_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add layerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention_talking_head,
                 Mlp_block=Mlp,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):        
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x 
    
    
    
    
class cait_models(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to adapt to our cait models
    """
    Implementation from https://github.com/facebookresearch/deit/blob/main/cait_models.py, modified to allow for triplet learning.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = LayerScale_Block,
                 block_layers_token = LayerScale_Block_CA,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention_talking_head,Mlp_block=Mlp,
                init_scale=1e-4,
                Attention_block_token_only=Class_Attention,
                Mlp_block_token_only= Mlp, 
                depth_token_only=2,
                mlp_ratio_clstk = 4.0,
                triplet = False):
        super().__init__()
        

        self.triplet = triplet
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)] 
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        

        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only,init_values=init_scale)
            for i in range(depth_token_only)])
            
        self.norm = norm_layer(embed_dim)


        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        if not self.triplet:
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x,cls_tokens)
        
        if self.triplet:
            if self.training:
                cls_tokens = F.normalize(cls_tokens, p=2, dim=-1)
            return cls_tokens
        else:
            x = torch.cat((cls_tokens, x), dim=1)

            x = self.norm(x)

            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        if self.triplet:
            return x.squeeze()
        x = self.head(x)

        return x 
                

class cait_models_twoQ(nn.Module):
    """
    Modified cait_models class to allow the addition of an extra token
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = LayerScale_Block,
                 block_layers_token = LayerScale_Block_CA,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention_talking_head,Mlp_block=Mlp,
                init_scale=1e-4,
                Attention_block_token_only=Class_Attention_twoQ,
                Mlp_block_token_only= Mlp, 
                depth_token_only=2,
                mlp_ratio_clstk = 4.0,
                triplet = False,):
        super().__init__()
        

        self.triplet = triplet
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # extra cls token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)] 
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        

        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only,init_values=init_scale)
            for i in range(depth_token_only)])
            
        self.norm = norm_layer(embed_dim)


        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        if not self.triplet:
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.extra_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'extra_token'}


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        extra_tokens = self.extra_token.expand(B, -1, -1)
        cls = torch.cat((cls_tokens, extra_tokens), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        for i , blk in enumerate(self.blocks_token_only):
            cls = blk(x,cls)
        if self.training:
            cls = F.normalize(cls, p=2, dim=-1)
        return cls

    def forward(self, x):
        x = self.forward_features(x)

        if self.training:
            if self.triplet:
                return x[:,0].squeeze(), x[:, 1].squeeze()
            else:
                x = self.head(x)
                return x 
        else:
            return (x[:,0].squeeze() + x[:, 1].squeeze()) / 2
