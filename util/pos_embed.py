# Copyright (c) SenseTime.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch
from torch.nn import functional as F

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/legacy/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# MAE:  https://github.com/facebookresearch/mae
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, cossin=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, cossin=cossin)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, cossin=False):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], cossin=cossin)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], cossin=cossin)  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, cossin=False):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    if cossin:
        emb = np.concatenate([emb_cos, emb_sin], axis=1)  # (M, D)
    else:
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model, remove_index=True):
    if 'absolute_pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['absolute_pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = 0

        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size), flush=True)
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['absolute_pos_embed'] = new_pos_embed
    
    if 'decoder_pos_embed' in checkpoint_model and hasattr(model, 'decoder_pos_embed'):
        pos_embed_checkpoint = checkpoint_model['decoder_pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.decoder_pos_embed.shape[-2]
        num_extra_tokens = 0

        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size), flush=True)
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['decoder_pos_embed'] = new_pos_embed

    # interpolate position bias table if needed
    relative_position_bias_table_keys = [k for k in checkpoint_model.keys() if "relative_position_bias_table" in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = checkpoint_model[table_key]
        if not table_key in model.state_dict():
            print(f"Key {table_key} not in model")
            continue
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {table_key}, pass")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                print(f"relative bias from {S1} to {S2}")
                table_pretrained_resized = F.interpolate(
                     table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                     size=(S2, S2), mode='bicubic')
                checkpoint_model[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)
    
    bias_index_keys = []
    for k in checkpoint_model.keys():
        if 'relative_position_index' in k and remove_index:
            print(f'del {k} from pretrain ckpt')
            # do not load relative position index for workaround of changing input size
            bias_index_keys.append(k)
    for k in bias_index_keys:
        del checkpoint_model[k]

    
    bias_index_keys = []
    for k in checkpoint_model.keys():
        if 'attn_mask' in k:
            print(f'del {k} from pretrain ckpt')
            bias_index_keys.append(k)
    for k in bias_index_keys:
        del checkpoint_model[k]