"""VRWKV-based image encoder.

Input:  (B, 3, H, W)  images in [0, 1]
Output: (B, N, C)     patch tokens, N = (H // patch_size) * (W // patch_size)

Default arguments match the segmentation checkpoint:
    embed_dims=768, depth=12, patch_size=16, post_norm=True, init_values=1e-5
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath
from torch import Tensor
from torch.utils.cpp_extension import load

from huggingface_hub import hf_hub_download

from src.models.encoders.base import BaseEncoder

wkv_cuda = load(
    name="bi_wkv",
    sources=[
        "cuda/bi_wkv.cpp",
        "cuda/bi_wkv_kernel.cu",
    ],
    verbose=True,
    extra_cuda_cflags=[
        '-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3',
    ]
)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(
            w.float().contiguous(), u.float().contiguous(),
            k.float().contiguous(), v.float().contiguous(),
            gy.float().contiguous()
        )
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())


# ---------------------------------------------------------------------------
# Positional embedding interpolation
# ---------------------------------------------------------------------------

def resize_pos_embed(
    pos_embed: Tensor,
    src_resolution: Tuple[int, int],
    dst_resolution: Tuple[int, int],
    mode: str = "bicubic",
) -> Tensor:
    if src_resolution == dst_resolution:
        return pos_embed
    H_src, W_src = src_resolution
    H_dst, W_dst = dst_resolution
    C = pos_embed.shape[-1]
    pe = pos_embed.reshape(1, H_src, W_src, C).permute(0, 3, 1, 2).float()
    pe = F.interpolate(pe, size=(H_dst, W_dst), mode=mode, align_corners=False)
    return pe.permute(0, 2, 3, 1).reshape(1, H_dst * W_dst, C).to(pos_embed.dtype)


# ---------------------------------------------------------------------------
# q_shift
# ---------------------------------------------------------------------------

def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# SpatialMix, ChannelMix, Block
# ---------------------------------------------------------------------------

class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.key_norm = nn.LayerNorm(attn_sz) if key_norm else None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1)
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)
                zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution):
        B, T, C = x.size()
        self.device = x.device
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = xv = xr = x
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        rwkv = RUN_CUDA(self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        return self.output(rwkv)


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4,
                 init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None
        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.key_norm = nn.LayerNorm(hidden_sz) if key_norm else None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution):
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = xr = x
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False,
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)
        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, init_mode, key_norm=key_norm)
        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, hidden_rate, init_mode, key_norm=key_norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones(n_embd), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones(n_embd), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, patch_resolution):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dims: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dims, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)


# ---------------------------------------------------------------------------
# VRWKVEncoder
# ---------------------------------------------------------------------------

class VRWKVEncoder(BaseEncoder):
    """Frozen Vision-RWKV image encoder.

    Loads the pretrained VRWKV-Base segmentation checkpoint and extracts
    patch token features. Always kept in eval mode. Drop-in replacement
    inside ImageEnhancementPipeline.

    Args:
        checkpoint:       Local path or HF URL to .pth file.
                          Defaults to the ADE20K segmentation checkpoint.
        frozen:           Whether to freeze all parameters (default True).
        img_size:         Training resolution used to initialise pos_embed.
        patch_size:       Patch size (16 for VRWKV-Base).
        embed_dims:       Token feature dimension (768 for VRWKV-Base).
        depth:            Number of transformer blocks (12 for VRWKV-Base).
        drop_path_rate:   Stochastic depth rate (0.3 in seg training, 0 at inference).
        post_norm:        Apply LayerNorm after residual (True for VRWKV-Base).
        init_values:      Layer-scale initial value (1e-5 for VRWKV-Base).
        channel_gamma:    Fraction of channels shifted per direction (0.25).
        shift_pixel:      Pixels to shift in q_shift (1).
        shift_mode:       Shift function name (default 'q_shift').
        hidden_rate:      FFN expansion ratio (4).
        final_norm:       Apply final LayerNorm after last block (True).
        interpolate_mode: Interpolation mode for pos_embed resize ('bicubic').
        with_cp:          Use gradient checkpointing (default False).
    """

    # Keys present in VRWKV_Adapter but not in plain VRWKV — skip these
    _ADAPTER_KEY_FRAGMENTS = [
        "interactions", "spm", "up", "norm1", "norm2", "norm3", "norm4",
        "level_embed",
    ]

    def __init__(
        self,
        checkpoint: Optional[str] = "upernet_vrwkv_adapter_base_512_160k_ade20k.pth",
        frozen: bool = True,
        img_size: int = 512,
        patch_size: int = 16,
        embed_dims: int = 768,
        depth: int = 12,
        drop_path_rate: float = 0.0,     # keep 0 — encoder is frozen
        post_norm: bool = True,
        init_values: Optional[float] = 1e-5,
        channel_gamma: float = 0.25,
        shift_pixel: int = 1,
        shift_mode: str = 'q_shift',
        hidden_rate: int = 4,
        final_norm: bool = True,
        interpolate_mode: str = "bicubic",
        with_cp: bool = False,
    ) -> None:
        super().__init__()

        self._embed_dim            = embed_dims
        self._patch_size           = patch_size
        self._interpolate_mode     = interpolate_mode
        self._final_norm           = final_norm
        self._train_patch_resolution = (img_size // patch_size, img_size // patch_size)

        # ── patch embedding ───────────────────────────────────────────
        self.patch_embed = PatchEmbed(3, embed_dims, patch_size)
        self.pos_embed   = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dims))
        self.drop_after_pos = nn.Dropout(p=0.0)

        # ── stochastic depth schedule ─────────────────────────────────
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # ── transformer blocks ────────────────────────────────────────
        self.layers = nn.ModuleList([
            Block(
                n_embd=embed_dims,
                n_layer=depth,
                layer_id=i,
                shift_mode=shift_mode,
                channel_gamma=channel_gamma,
                shift_pixel=shift_pixel,
                drop_path=dpr[i],
                hidden_rate=hidden_rate,
                init_mode="fancy",
                init_values=init_values,
                post_norm=post_norm,
                key_norm=False,
                with_cp=with_cp,
            )
            for i in range(depth)
        ])

        if final_norm:
            self.ln1 = nn.LayerNorm(embed_dims)

        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        # ── load weights then freeze ──────────────────────────────────
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        if frozen:
            self.freeze()

    def load_checkpoint(self, checkpoint: str) -> None:
        """Load from a local path, or download from HuggingFace."""
        if os.path.exists(checkpoint):
            print(f"[VRWKVEncoder] Loading checkpoint from {checkpoint} ...")
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        else:
            print(f"[VRWKVEncoder] Downloading checkpoint from HuggingFace ...")
            local_path = hf_hub_download(
                repo_id="OpenGVLab/Vision-RWKV",
                filename="upernet_vrwkv_adapter_base_512_160k_ade20k.pth",
            )
            ckpt = torch.load(local_path, map_location="cpu", weights_only=False)

        raw = ckpt.get("state_dict", ckpt)
        remapped = {}
        for k, v in raw.items():
            if not k.startswith("backbone."):
                continue
            k = k[len("backbone."):]
            if any(frag in k for frag in self._ADAPTER_KEY_FRAGMENTS):
                continue
            k = k.replace("patch_embed.projection.", "patch_embed.proj.")
            remapped[k] = v

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        print(
            f"[VRWKVEncoder] Loaded {len(remapped) - len(unexpected)} tensors. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}."
        )
        if missing:
            print(f"  Missing (first 5):    {missing[:5]}")
        if unexpected:
            print(f"  Unexpected (first 5): {unexpected[:5]}")

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            # Normalize [0,1] float input to ImageNet distribution
            x = x.clamp(0.0, 1.0)
            mean = self.pixel_mean.to(device=x.device, dtype=x.dtype)
            std = self.pixel_std.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
            x, patch_resolution = self.patch_embed(x)
            pos = resize_pos_embed(
                self.pos_embed,
                self._train_patch_resolution,
                patch_resolution,
                mode=self._interpolate_mode,
            )
            x = self.drop_after_pos(x + pos)
            for layer in self.layers:
                x = layer(x, patch_resolution)
            if self._final_norm:
                x = self.ln1(x)
        return x    # (B, N, C)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size
