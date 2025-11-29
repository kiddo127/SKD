import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
import math
import logging
import collections.abc
from itertools import repeat
from collections import OrderedDict

import timm
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import PatchEmbed, DropPath, trunc_normal_, lecun_normal_
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv

_logger = logging.getLogger(__name__)

__all__ = [
    'vit_base', 'vit_small', 'vit_tiny',
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
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

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    @torch.no_grad()
    def init_statistics(self):
        self.sample_count = torch.tensor(0).cuda()
        self.imp_accum = torch.zeros(self.hidden_features).cuda()
        
        self.gradients = None
        self.features = None
        def forward_hook(module, input, output):
            self.features = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        # self.hook_forward = self.fc1.register_forward_hook(forward_hook)
        # self.hook_backward = self.fc1.register_backward_hook(backward_hook)
        self.hook_forward = self.act.register_forward_hook(forward_hook)
        self.hook_backward = self.act.register_backward_hook(backward_hook)

        self.reparamed = False

    @torch.no_grad()
    def update_importance(self):
        grad_matrix = self.gradients     # [B, N, D]
        features = self.features         # [B, N, D]

        first_order = torch.tensor(0)
        hessian_diag_approx = grad_matrix ** 2
        second_order = 0.5 * hessian_diag_approx * (features ** 2)  # [B, N, D]
        total_imp = (first_order - second_order).abs()  # [B, N, D]

        total_imp = total_imp.mean(dim=1)  # [B, D]
        total_imp = total_imp.mean(dim=0)  # [D]

        self.imp_accum += total_imp

    @torch.no_grad()
    def rank_importance(self):
        self.hook_forward.remove()
        self.hook_backward.remove()
        with torch.no_grad():
            importance = self.imp_accum
            sorted_idx = torch.argsort(importance, descending=True)
            hidden_features = sorted_idx.size(0)
            W = torch.zeros((hidden_features, hidden_features), device=sorted_idx.device)
            W.scatter_(1, sorted_idx.unsqueeze(1), 1.0)
            return W.T

    @torch.no_grad()
    def apply_transform(self):
        W = self.rank_importance()

        fc1_weight = self.fc1.weight.data
        fc1_bias = self.fc1.bias.data
        self.fc1.weight.data = (fc1_weight.T @ W).T
        self.fc1.bias.data = fc1_bias @ W

        fc2_weight = self.fc2.weight.data
        self.fc2.weight.data = fc2_weight @ W

        self.reparamed = True

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)

        if not self.reparamed:
            with torch.no_grad():
                self.sample_count += x.size(0)

        # if self.reparamed:
        #     reserved_dims = int(self.hidden_features * 0.5)
        #     x[:, :, reserved_dims:] = 0.0

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

    @torch.no_grad()
    def init_statistics(self):
        self.sample_count = torch.tensor(0).cuda()
        self.v_cov_accum = [torch.zeros((self.head_dim, self.head_dim)).cuda() for _ in range(self.num_heads)]
        self.qk_cov_accum = [torch.zeros((self.head_dim, self.head_dim)).cuda() for _ in range(self.num_heads)]
        self.centered_v = [torch.zeros((256*197, self.head_dim)).cuda() for _ in range(self.num_heads)]
        self.centered_qk = [torch.zeros((2*256*197, self.head_dim)).cuda() for _ in range(self.num_heads)]
        self.centered_q = [torch.zeros((256*197, self.head_dim)).cuda() for _ in range(self.num_heads)]
        self.centered_k = [torch.zeros((256*197, self.head_dim)).cuda() for _ in range(self.num_heads)]
        self.gradients = None
        self.features = None
        def forward_hook(module, input, output):
            self.features = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.hook_forward = self.qkv.register_forward_hook(forward_hook)
        self.hook_backward = self.qkv.register_backward_hook(backward_hook)

        self.reparamed = False

    @torch.no_grad()
    def update_importance(self):
        B, N, C = self.gradients.shape
        C = C // 3
        grad_matrix = self.gradients # B,n,3d
        features = self.features # B,n,3d
        batch_taylor_imp = (grad_matrix * features).abs() # B,n,3d
        qkv_imp = batch_taylor_imp.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_imp, k_imp, v_imp = qkv_imp.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        for h in range(self.num_heads):
            # ======V======
            v_imp_h = v_imp[:, h, :, :].reshape(-1, self.head_dim)  # (B*N, head_dim)
            v_imp_h = v_imp_h.mean(dim=1).unsqueeze(1)
            centered_v = torch.pow(v_imp_h, 1/2) * self.centered_v[h]
            cov = centered_v.T @ centered_v
            self.v_cov_accum[h] += cov
            # ======QK======
            q_imp_h = q_imp[:, h, :, :].reshape(-1, self.head_dim)  # (B*N, head_dim)
            k_imp_h = k_imp[:, h, :, :].reshape(-1, self.head_dim)  # (B*N, head_dim)
            q_imp_h = q_imp_h.mean(dim=1).unsqueeze(1)
            k_imp_h = k_imp_h.mean(dim=1).unsqueeze(1)
            centered_q = torch.pow(q_imp_h, 1/2) * self.centered_q[h]
            centered_k = torch.pow(k_imp_h, 1/2) * self.centered_k[h]
            cov = centered_q.T @ centered_q + centered_k.T @ centered_k
            self.qk_cov_accum[h] += cov

    @torch.no_grad()
    def compute_pca_matrices(self):
        self.pca_vproj_matrices = []
        self.pca_qk_matrices = []
        for h in range(self.num_heads):
            # ======V======
            cov = self.v_cov_accum[h] / (self.sample_count - 1)
            eigvals, eigvecs = torch.linalg.eigh(cov.cpu())  # eigvecs: (head_dim, head_dim)
            idx = torch.argsort(eigvals, descending=True)
            W = eigvecs[:, idx]  # shape (head_dim, head_dim)
            self.pca_vproj_matrices.append(W.cuda())
            # ======QK======
            cov = self.qk_cov_accum[h] / (self.sample_count - 1)
            eigvals, eigvecs = torch.linalg.eigh(cov.cpu())  # eigvecs: (head_dim, head_dim)
            idx = torch.argsort(eigvals, descending=True)
            W = eigvecs[:, idx]  # shape (head_dim, head_dim)
            self.pca_qk_matrices.append(W.cuda())

    @torch.no_grad()
    def build_block_diagonal_W(self):
        self.compute_pca_matrices()
        vproj_blocks = [self.pca_vproj_matrices[h] for h in range(self.num_heads)]  # each (d_h, d_h)
        vproj_W = torch.block_diag(*vproj_blocks)  # (d, d)

        qk_blocks = [self.pca_qk_matrices[h] for h in range(self.num_heads)]  # each (d_h, d_h)
        qk_W = torch.block_diag(*qk_blocks)  # (d, d)
        return vproj_W, qk_W

    @torch.no_grad()
    def apply_transform(self):
        vproj_W, qk_W = self.build_block_diagonal_W()  # (d, d)

        # qkv: (dim, dim*3)
        dim = self.proj.in_features
        qkv_weight = self.qkv.weight.data  # (d, 3d)
        qkv_bias = self.qkv.bias.data  # (d, 3d)
        v_start = 2 * dim

        self.qkv.weight.data[v_start:, :] = (qkv_weight[v_start:, :].T @ vproj_W).T  # (d, d)
        self.qkv.bias.data[v_start:] = qkv_bias[v_start:] @ vproj_W  # (d)

        # proj: (d, d)
        proj_weight = self.proj.weight.data
        self.proj.weight.data = (vproj_W.T @ proj_weight.T).T

        self.qkv.weight.data[:dim, :] = (qkv_weight[:dim, :].T @ qk_W).T  # (d, d)
        self.qkv.bias.data[:dim] = qkv_bias[:dim] @ qk_W  # (d)

        self.qkv.weight.data[dim:v_start, :] = (qkv_weight[dim:v_start, :].T @ qk_W).T  # (d, d)
        self.qkv.bias.data[dim:v_start] = qkv_bias[dim:v_start] @ qk_W  # (d)

        self.reparamed = True

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if self.reparamed:
            used_head_dim = int(self.head_dim * 0.5)
            v[:,:,:,used_head_dim:] = 0.0
            q[:,:,:,used_head_dim:] = 0.0
            k[:,:,:,used_head_dim:] = 0.0

        if not self.reparamed:
            with torch.no_grad():
                self.sample_count += B
                for h in range(self.num_heads):
                    # ======V======
                    v_h_flat = v[:, h, :, :].reshape(-1, self.head_dim)  # (B*N, head_dim)
                    self.centered_v[h] = v_h_flat - v_h_flat.mean(dim=0)
                    # ======QK======
                    q_h_flat = q[:, h, :, :].reshape(-1, self.head_dim)  # (B*N, head_dim)
                    k_h_flat = k[:, h, :, :].reshape(-1, self.head_dim)  # (B*N, head_dim)
                    qk_h_flat = torch.cat([q_h_flat,k_h_flat], dim=0)  # (2*B*N, head_dim)
                    self.centered_qk[h] = qk_h_flat - qk_h_flat.mean(dim=0)
                    self.centered_q[h] = q_h_flat - q_h_flat.mean(dim=0)
                    self.centered_k[h] = k_h_flat - k_h_flat.mean(dim=0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__()
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

    def init_statistics(self):
        for blk in self.blocks:
            blk.attn.init_statistics()
            blk.mlp.init_statistics()

    def update_importance(self):
        for blk in self.blocks:
            blk.attn.update_importance()
            blk.mlp.update_importance()

    def apply_transform(self):
        for blk in self.blocks:
            blk.attn.apply_transform()
            blk.mlp.apply_transform()


    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        # x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x=None):
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



class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2



@register_model
def vit_tiny(pretrained=True, **kwargs):
    model = VisionTransformer(
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
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth", # 72.1
            map_location="cpu",
            check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vit_small(pretrained=True, **kwargs):
    model = VisionTransformer(
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
    if pretrained:
        url = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth" # 79.8
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url,
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vit_base(pretrained=True, **kwargs):
    model = VisionTransformer(
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
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth", # 81.8
            map_location="cpu",
            check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model




def create_model(model, pretrained, num_classes):
    if 'vit_base' in model:
        model = vit_base(pretrained=pretrained, num_classes=num_classes)
    elif 'vit_small' in model:
        model = vit_small(pretrained=pretrained, num_classes=num_classes)
    elif 'vit_tiny' in model:
        model = vit_tiny(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(model)


    return model