# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import pdb
from this import d
import torch
import torch.nn as nn

from vision_transformer import PatchEmbed, Block, CBlock

from util.pos_embed import get_2d_sincos_pos_embed

from torch.nn import functional as F

class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        decoder_depth=8
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed1 = PatchEmbed(
                img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
                img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
                img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])

        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.stage1_output_decode = nn.Conv2d(embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv2d(embed_dim[1], embed_dim[2], 2, stride=2)

        num_patches = self.patch_embed3.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[2]), requires_grad=False)
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0],  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1],  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2],  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[2])])
        self.norm = norm_layer(embed_dim[-1])

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio[0], qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size[0] * patch_size[1] * patch_size[2])**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.mse_loss = nn.MSELoss(reduction='sum')

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed3.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed3.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def masks_splitting(self, ids_restore, ids_shuffle, start_idx, end_idx):
        """
        Perform complementary masks splitting with no overlap keep ids
        ids_restore: (N, L)
        ids_shuffle: (N, L)
        """
        N, L = ids_restore.shape
        ids_keep = ids_shuffle[:, start_idx:end_idx]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=ids_restore.device)
        mask[:, start_idx:end_idx] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return ids_keep, mask

    def embeddings_gatherings(self, x, ids_keep, stage1_embed, stage2_embed):
        """
        x: [N, L, D], sequence
        ids_keep: [N, 4, L_keep]
        """
        def quadruple_batch(t):
            # [N, L, D] -> [N, 4, L, D]
            return t.unsqueeze(1).expand(t.shape[0], 4, t.shape[1], t.shape[2])

        def merge_quadrupled_batch(t):
            return t.reshape([-1, t.shape[2], t.shape[3]])

        x = quadruple_batch(x)
        stage1_embed = quadruple_batch(stage1_embed)
        stage2_embed = quadruple_batch(stage2_embed)

        x = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).expand([-1, -1, -1, x.shape[-1]]))
        stage1_embed = torch.gather(stage1_embed, dim=2, index=ids_keep.unsqueeze(-1).expand([-1, -1, -1, stage1_embed.shape[-1]]))
        stage2_embed = torch.gather(stage2_embed, dim=2, index=ids_keep.unsqueeze(-1).expand([-1, -1, -1, stage2_embed.shape[-1]]))

        assert x.is_contiguous()
        assert stage1_embed.is_contiguous()
        assert stage2_embed.is_contiguous()

        # [N, 4, L, D] -> [N * 4, L, D]
        x = merge_quadrupled_batch(x)
        stage1_embed = merge_quadrupled_batch(stage1_embed)
        stage2_embed = merge_quadrupled_batch(stage2_embed)

        return x, stage1_embed, stage2_embed

    def embeddings_gathering(self, x, ids_keep, stage1_embed, stage2_embed):
        """
        Perform complementary masked embeddings gathering from stage 1 & 2
        x: [N, L, D], sequence
        ids_keep: [N, L_keep]
        """
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        stage1_embed = torch.gather(stage1_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
        stage2_embed = torch.gather(stage2_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))
        return x, stage1_embed, stage2_embed

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        Per-sample masking is done by four complementary masks.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = self.patch_embed3.num_patches
#        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # complementary masks splitting
        assert L % 4 == 0
        assert mask_ratio == 0.75
        ids_keep = ids_shuffle.reshape(N, 4, L // 4)

        mask = torch.ones([N, 4, L], device=ids_restore.device)
        mask[:, 0, 0:len_keep] = 0
        mask[:, 1, len_keep:2*len_keep] = 0
        mask[:, 2, 2*len_keep:3*len_keep] = 0
        mask[:, 3, 3*len_keep:L] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore.unsqueeze(1).expand(-1, 4, -1))

        return ids_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        ids_keep, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask_for_patch1 = [1 - mask[:, i].reshape(-1, 14, 14).unsqueeze(-1).expand(-1, -1, -1, 16).reshape(-1, 14, 14, 4, 4).permute(0, 1, 3, 2, 4).reshape(x.shape[0], 56, 56).unsqueeze(1) for i in range(0, 4)]
        mask_for_patch2 = [1 - mask[:, i].reshape(-1, 14, 14).unsqueeze(-1).expand(-1, -1, -1, 4).reshape(-1, 14, 14, 2, 2).permute(0, 1, 3, 2, 4).reshape(x.shape[0], 28, 28).unsqueeze(1) for i in range(0, 4)]
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x, mask_for_patch1)
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, mask_for_patch2)
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)
        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed
        # gather embeddigns for each mask
        x, stage1_embed, stage2_embed = self.embeddings_gatherings(x, ids_keep, stage1_embed, stage2_embed)
        # apply Transformer blocks
        for blk in self.blocks3:
            x = blk(x)
        x = x + stage1_embed + stage2_embed
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1]  - x.shape[1], self.mask_token.shape[2])
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token

        # split decoder embeddings for each complementary mask
        B = x_.shape[0]
        x_ = x_.reshape([B // 4, 4, x_.shape[1], x.shape[2]])
        x_[:, 1] = x_[:, 1].roll(49 * 1, 1)
        x_[:, 2] = x_[:, 2].roll(49 * 2, 1)
        x_[:, 3] = x_[:, 3].roll(49 * 3, 1)

        ids_restore = ids_restore.unsqueeze(1).unsqueeze(-1).expand([-1, 4, -1, x.shape[2]])

        x = torch.gather(x_, dim=2, index=ids_restore)  # unshuffle

        x = x.reshape([B, x.shape[2], x.shape[3]])

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        target = target.unsqueeze(1).expand([-1, 4, -1, -1])
        #mask = mask.reshape([mask.shape[0] // 4, 4, mask.shape[1]])
        pred = pred.reshape([pred.shape[0] // 4, 4, pred.shape[1], pred.shape[2]])

        mask_bool = (mask == 0)
        pred = torch.where(mask_bool.unsqueeze(-1).expand(-1, -1, -1, target.size(3)), target, pred.float())

        loss = self.mse_loss(pred, target)
        loss /= target.size(3)
        loss /= mask.sum()

        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def fastconvmae_convvit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def fastconvmae_convvit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[128, 256, 384], depth=[2, 2, 11], num_heads=6,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=[4, 4, 4],
        norm_layer=partial(LayerNormFp32, eps=1e-6),
        **kwargs)
    return model

# set recommended archs
fastconvmae_convvit_base_patch16 = fastconvmae_convvit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
fastconvmae_convvit_small_patch16 = fastconvmae_convvit_small_patch16_dec512d8b