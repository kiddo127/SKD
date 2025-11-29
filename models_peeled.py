# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
from itertools import repeat
import collections.abc
from collections import OrderedDict
import math
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
import logging
import json

import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import PatchEmbed, DropPath, trunc_normal_, lecun_normal_

_logger = logging.getLogger(__name__)

__all__ = [
    'deit_tiny_patch16_224_peeled', 'deit_small_patch16_224_peeled', 'deit_base_patch16_224_peeled', 'vit_base_peeled', 'vit_small_peeled'
]


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb




# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)






class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.sensitivity = torch.tensor(0).cuda()
        self.mask_table = {0:self.hidden_features, -1:self.hidden_features} # dict [num:mlp_mask_point]
        if in_features == 192:
            self.group_size = 24
            self.MACs = 58.44
        elif in_features == 384:
            self.group_size = 48
            self.MACs = 233.07
        elif in_features == 768:
            self.group_size = 96
            self.MACs = 930.92
        self.group_MACs = self.MACs * self.group_size / self.hidden_features
        self.dim_MACs = self.MACs / self.hidden_features

    @torch.no_grad()
    def init_statistics(self):
        self.batch_count = torch.tensor(0).cuda()
        self.imp_accum = torch.zeros(self.mask_table[-1]).cuda()
        self.gradients = None
        self.features = None

    @torch.no_grad()
    def update_importance(self):
        if self.gradients is not None and self.features is not None:
            grad_matrix = self.gradients # B,n,d
            features = self.features # B,n,d
            batch_taylor_imp = (grad_matrix * features).abs() # B,n,d
            batch_taylor_imp = batch_taylor_imp.mean(dim=1) # B,d
            batch_taylor_imp = batch_taylor_imp.mean(dim=0)
            self.imp_accum += batch_taylor_imp

    @torch.no_grad()
    def normalize_contributions(self):
        if self.gradients is not None and self.features is not None:
            imp_accum = self.imp_accum / self.imp_accum.sum()
            self.imp_accum = imp_accum * self.sensitivity

    @torch.no_grad()
    def build_compression_operations(self):
        self.normalize_contributions()
        end_point = self.mask_table[-1]
        self.operations = []
        pre_contribution = self.imp_accum.sum() / self.dim_MACs
        for mask_point in range(0, end_point, self.group_size):
            MACs = (end_point - mask_point) * self.dim_MACs
            contribution = self.imp_accum[mask_point:].sum() / MACs
            # rank_contribution = self.imp_accum[mask_point:mask_point+self.group_size].sum() / self.group_MACs
            if contribution <= pre_contribution:
                self.operations.append({
                    'mask_point': mask_point,
                    'MACs': MACs,
                    'contribution': contribution
                    # 'rank_contribution': rank_contribution
                })
                pre_contribution = contribution
        for op_idx in range(len(self.operations)-1):
            self.operations[op_idx]['MACs'] -= self.operations[op_idx+1]['MACs']

    @torch.no_grad()
    def fix_mask_table(self, add_ops):
        start_code = len(self.mask_table) - 2
        new_start = len(self.mask_table) - 2
        mask_point = self.mask_table[-1]
        for op in reversed(self.operations):
            if op['rank'] > add_ops:
                break
            for op_code in range(new_start,start_code+min(add_ops,op['rank'])):
                self.mask_table[op_code] = mask_point
            mask_point = op['mask_point']
            new_start = start_code+min(add_ops,op['rank'])
        for op_code in range(new_start,start_code+add_ops+1):
            self.mask_table[op_code] = mask_point
        self.mask_table[-1] = mask_point
        print('mlp', self.mask_table)
        assert len(self.mask_table) - 2 - start_code == add_ops

    @torch.no_grad()
    def gather_reductions(self):
        reduced_MACs = (self.hidden_features - self.mask_table[-1]) * self.dim_MACs
        return reduced_MACs

    def drop_param(self, op_code):
        self.hidden_features = self.mask_table[op_code]
        # fc1
        fc1_weight = self.fc1.weight.data[:self.hidden_features, :]
        fc1_bias = self.fc1.bias.data[:self.hidden_features]
        new_fc1 = nn.Linear(in_features=self.fc1.in_features, out_features=self.hidden_features, bias=True)
        new_fc1.weight.data = fc1_weight
        new_fc1.bias.data = fc1_bias
        self.fc1 = new_fc1
        # fc2
        fc2_weight = self.fc2.weight.data[:, :self.hidden_features]
        fc2_bias = self.fc2.bias.data
        new_fc2 = nn.Linear(in_features=self.hidden_features, out_features=self.fc2.out_features, bias=True)
        new_fc2.weight.data = fc2_weight
        new_fc2.bias.data = fc2_bias
        self.fc2 = new_fc2



    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sensitivity = torch.tensor(0).cuda()
        self.mask_table = {0:self.head_dim, -1:self.head_dim} # dict [num:mlp_mask_point]
        self.group_size = 8
        if dim == 192:
            self.MACs = 29.2
        elif dim == 384:
            self.MACs = 116.5
        elif dim == 768:
            self.MACs = 465.39
        self.group_MACs = self.MACs * self.group_size / self.head_dim
        self.dim_MACs = self.MACs / self.head_dim

    @torch.no_grad()
    def init_statistics(self):
        self.batch_count = torch.tensor(0).cuda()
        self.imp_accum = torch.zeros(self.mask_table[-1]).cuda()
        self.gradients = None
        self.features = None

    @torch.no_grad()
    def update_importance(self):
        if self.gradients is not None and self.features is not None:
            grad_matrix = self.gradients # B,n,3d
            features = self.features # B,n,3d
            batch_taylor_imp = (grad_matrix * features).abs() # B,n,3d

            batch_taylor_imp = batch_taylor_imp.mean(dim=1) # B,3d
            batch_taylor_imp = batch_taylor_imp.sum(dim=0) # 3d
            batch_taylor_imp = batch_taylor_imp.reshape(3, self.num_heads, self.mask_table[-1]) # 3, num_head, head_dim
            q, k, v = batch_taylor_imp.unbind(0)
            imp_accum = q + k + v

            self.imp_accum += imp_accum.mean(dim=0)

    @torch.no_grad()
    def normalize_contributions(self):
        if self.gradients is not None and self.features is not None:
            imp_accum = self.imp_accum / self.imp_accum.sum()
            self.imp_accum = imp_accum * self.sensitivity

    @torch.no_grad()
    def build_compression_operations(self):
        self.normalize_contributions()
        end_point = self.mask_table[-1]
        self.operations = []
        pre_contribution = self.imp_accum.sum() / self.dim_MACs
        for mask_point in range(0, end_point, self.group_size):
            MACs = (end_point - mask_point) * self.dim_MACs
            contribution = self.imp_accum[mask_point:].sum() / MACs
            if contribution <= pre_contribution:
                self.operations.append({
                    'mask_point': mask_point,
                    'MACs': MACs,
                    'contribution': contribution
                })
                pre_contribution = contribution
        for op_idx in range(len(self.operations)-1):
            self.operations[op_idx]['MACs'] -= self.operations[op_idx+1]['MACs']

    @torch.no_grad()
    def fix_mask_table(self, add_ops):
        start_code = len(self.mask_table) - 2
        new_start = len(self.mask_table) - 2
        mask_point = self.mask_table[-1]
        for op in reversed(self.operations):
            if op['rank'] > add_ops:
                break
            for op_code in range(new_start,start_code+min(add_ops,op['rank'])):
                self.mask_table[op_code] = mask_point
            mask_point = op['mask_point']
            new_start = start_code+min(add_ops,op['rank'])
        for op_code in range(new_start,start_code+add_ops+1):
            self.mask_table[op_code] = mask_point
        self.mask_table[-1] = mask_point
        print('attn', self.mask_table)
        assert len(self.mask_table) - 2 - start_code == add_ops

    @torch.no_grad()
    def gather_reductions(self):
        reduced_MACs = (self.head_dim - self.mask_table[-1]) * self.dim_MACs
        return reduced_MACs


    def drop_param(self, op_code):
        mask = torch.zeros(self.num_heads, self.head_dim, dtype=torch.bool)
        self.head_dim = self.mask_table[op_code]
        mask[:, :self.head_dim] = True
        mask = mask.view(-1)
        qkv_mask = torch.cat([mask]*3, dim=0)
        # qkv
        qkv_weight = self.qkv.weight.data[qkv_mask]
        qkv_bias = self.qkv.bias.data[qkv_mask]
        new_qkv = nn.Linear(in_features=self.qkv.in_features, out_features=self.head_dim*self.num_heads, bias=True)
        new_qkv.weight.data = qkv_weight
        new_qkv.bias.data = qkv_bias
        self.qkv = new_qkv
        # proj
        proj_weight = self.proj.weight.data[:, mask]
        proj_bias = self.proj.bias.data
        new_proj = nn.Linear(in_features=self.head_dim*self.num_heads, out_features=self.proj.out_features, bias=True)
        new_proj.weight.data = proj_weight
        new_proj.bias.data = proj_bias
        self.proj = new_proj



    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def drop_param(self, op_code):
        if self.attn.mask_table[op_code] == 0:
            self.attn = None
            self.norm1 = None
        else:
            self.attn.drop_param(op_code)
        if self.mlp.mask_table[op_code] == 0:
            self.mlp = None
            self.norm2 = None
        else:
            self.mlp.drop_param(op_code)


    def forward(self, x):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.mlp is not None:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class VisionTransformer_peeled(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 peel_ratio=1.0):
        super().__init__()
        self.op_codes = []
        self.peel_ratio = peel_ratio
        self.num_heads = num_heads


        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()



    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))


    def load_mask_table(self, mask_table_path):
        with open(mask_table_path, 'r') as mask_table_file:
            mask_tables = json.load(mask_table_file)
        for blk_idx, blk in enumerate(self.blocks):
            attn_mask_table = mask_tables['{}.attn'.format(blk_idx)]
            blk.attn.mask_table = {int(k): v for k, v in attn_mask_table.items()}
            mlp_mask_table = mask_tables['{}.mlp'.format(blk_idx)]
            blk.mlp.mask_table = {int(k): v for k, v in mlp_mask_table.items()}


    def drop_param(self, op_code=-1):
        for blk in self.blocks:
            blk.drop_param(op_code)



    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x



@register_model
def vit_tiny_peeled(pretrained=True, **kwargs):
    model = VisionTransformer_peeled(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth", # 72.1
    #         map_location="cpu",
    #         check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vit_small_peeled(pretrained=True, **kwargs):
    model = VisionTransformer_peeled(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    # if pretrained:
    #     url = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth" # 79.8
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url=url,
    #         map_location="cpu",
    #         check_hash=True,
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vit_base_peeled(pretrained=True, **kwargs):
    model = VisionTransformer_peeled(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth", # 81.8
    #         map_location="cpu",
    #         check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model




def create_model(model, pretrained, num_classes, peel_ratio=1.0, drop_path_rate=0.1):
    if 'vit_base' in model:
        model = vit_base_peeled(pretrained=pretrained, num_classes=num_classes, peel_ratio=peel_ratio, drop_path_rate=drop_path_rate)
    elif 'vit_small' in model:
        model = vit_small_peeled(pretrained=pretrained, num_classes=num_classes, peel_ratio=peel_ratio, drop_path_rate=drop_path_rate)
    elif 'vit_tiny' in model:
        model = vit_tiny_peeled(pretrained=pretrained, num_classes=num_classes, peel_ratio=peel_ratio, drop_path_rate=drop_path_rate)

    return model