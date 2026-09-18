########################################################################################################
# RWKV-7 Official Implementation
# Ported from: https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp
#
# Key components:
# - RWKV7_CLAMPW_CUDA_OP: Official training CUDA kernel (forward + backward)
# - RWKV_Tmix_x070: Official TimeMix with complete initialization
# - RWKV_CMix_x070: Official ChannelMix
# - Block: Official RWKV7 block
#
# Usage:
#   Set environment variable: RWKV_HEAD_SIZE=64
#   Import and use RWKV7_Tmix_x070, RWKV_CMix_x070, or Block directly
########################################################################################################

from __future__ import annotations

import os
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop

if os.environ.get("RWKV_JIT_ON", "0") == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel - rwkv7_clampw (Official Training Implementation)
########################################################################################################

HEAD_SIZE = int(os.environ.get("RWKV_HEAD_SIZE", "64"))
CHUNK_LEN = 16
_RWKV7_CLAMPW_AVAILABLE = False


def _load_rwkv7_clampw(head_size: int = HEAD_SIZE) -> bool:
    """Load the official RWKV7 clampw CUDA kernel."""
    global _RWKV7_CLAMPW_AVAILABLE

    if _RWKV7_CLAMPW_AVAILABLE:
        return True

    if not torch.cuda.is_available():
        return False

    # Find CUDA sources in project root
    _root = Path(__file__).resolve().parents[3]
    cuda_sources = [
        _root / "cuda" / "rwkv7_clampw.cu",
        _root / "cuda" / "rwkv7_clampw.cpp",
    ]

    # Check if files exist
    missing = [src for src in cuda_sources if not src.is_file()]
    if missing:
        print(f"[RWKV7Official] CUDA sources not found: {missing}")
        return False

    try:
        flags = [
            "-res-usage",
            f"-D_N_={head_size}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
        ]
        load(
            name="rwkv7_clampw",
            sources=[str(src) for src in cuda_sources],
            is_python_module=False,
            verbose=False,
            extra_cuda_cflags=flags,
        )
        _RWKV7_CLAMPW_AVAILABLE = True
        print(f"[RWKV7Official] CUDA kernel loaded (head_size={head_size})")
        return True
    except Exception as e:
        print(f"[RWKV7Official] Failed to load CUDA kernel: {e}")
        return False


# Try to load CUDA kernel at import time
_load_rwkv7_clampw(HEAD_SIZE)


class RWKV7_CLAMPW_CUDA_OP(torch.autograd.Function):
    """Official RWKV7 clampw CUDA op with forward + backward."""

    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        B, T, H, N = r.shape
        assert T % CHUNK_LEN == 0, f"T={T} must be divisible by CHUNK_LEN={CHUNK_LEN}"
        assert all(i.dtype == torch.bfloat16 for i in [r, w, k, v, a, b]), "All inputs must be bfloat16"
        assert all(i.is_contiguous() for i in [r, w, k, v, a, b]), "All inputs must be contiguous"

        y = torch.empty_like(v)
        s = torch.empty(B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)
        torch.ops.rwkv7_clampw.forward(r, w, k, v, a, b, y, s, sa)
        ctx.save_for_backward(r, w, k, v, a, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert dy.dtype == torch.bfloat16 and dy.is_contiguous()

        r, w, k, v, a, b, s, sa = ctx.saved_tensors
        dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in [r, w, k, v, a, b]]
        torch.ops.rwkv7_clampw.backward(r, w, k, v, a, b, dy, s, sa, dr, dw, dk, dv, da, db)
        return dr, dw, dk, dv, da, db


def RWKV7_CLAMPW_CUDA(r, w, k, v, a, b):
    """Wrapper for official RWKV7 clampw CUDA kernel.

    Args:
        r: (B, T, HC) receptance
        w: (B, T, HC) decay (before clamping)
        k: (B, T, HC) key
        v: (B, T, HC) value
        a: (B, T, HC) aaa
        b: (B, T, HC) aaa * k

    Returns:
        y: (B, T, HC) output
    """
    if not _RWKV7_CLAMPW_AVAILABLE:
        raise RuntimeError("RWKV7 clampw CUDA kernel is not available")

    B, T, HC = r.shape
    r, w, k, v, a, b = [x.view(B, T, HC // HEAD_SIZE, HEAD_SIZE) for x in (r, w, k, v, a, b)]
    return RWKV7_CLAMPW_CUDA_OP.apply(r, w, k, v, a, b).view(B, T, HC)


########################################################################################################
# PyTorch Reference Implementation (for fallback / CPU)
########################################################################################################

def RWKV7_PYTORCH(r, w, k, v, a, b, head_size: int = HEAD_SIZE):
    """PyTorch reference implementation of RWKV7.

    Note: This is slower than CUDA but works on CPU and supports autograd.
    """
    orig_dtype = r.dtype
    B, T, C = r.shape
    H = C // head_size
    N = head_size

    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    # w is clamped in CUDA: exp(-0.6065 / (1 + sigmoid(w)))
    w = torch.exp(-0.6065306597 / (1 + torch.sigmoid(w.view(B, T, H, N).float())))

    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float32)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float32)

    for t in range(T):
        kk = k[:, t].view(B, H, 1, N)
        rr = r[:, t].view(B, H, N, 1)
        vv = v[:, t].view(B, H, N, 1)
        aa = a[:, t].view(B, H, N, 1)
        bb = b[:, t].view(B, H, 1, N)

        state = state * w[:, t, :, None, :] + (state @ aa) @ bb + vv @ kk
        out[:, t] = (state @ rr).view(B, H, N)

    return out.view(B, T, C).to(dtype=orig_dtype)


def RUN_WKV7(r, w, k, v, a, b, use_cuda: bool = True):
    """Run RWKV7 with CUDA or PyTorch fallback."""
    if use_cuda and _RWKV7_CLAMPW_AVAILABLE and r.is_cuda:
        return RWKV7_CLAMPW_CUDA(r, w, k, v, a, b)
    return RWKV7_PYTORCH(r, w, k, v, a, b)


########################################################################################################
# Official RWKV7 TimeMix (x070)
########################################################################################################

class RWKV_Tmix_x070(MyModule):
    """Official RWKV-7 TimeMix from train_temp.

    Features:
    - Complete zigzag/linear modulation for initialization
    - Dynamic LoRA dimension calculation
    - Value residual (v_first) across layers
    """

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            D_DECAY_LORA = max(32, int(round((2.5 * (C**0.5)) / 32) * 32))
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

            D_AAA_LORA = max(32, int(round((2.5 * (C**0.5)) / 32) * 32))
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)

            D_MV_LORA = max(32, int(round((1.7 * (C**0.5)) / 32) * 32))
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

            D_GATE_LORA = max(32, int(round((5 * (C**0.5)) / 32) * 32))
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.key.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
            self.value.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2  # clamped in CUDA
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        # Use CUDA or PyTorch based on availability
        use_cuda = _RWKV7_CLAMPW_AVAILABLE and x.is_cuda
        if use_cuda:
            x = RWKV7_CLAMPW_CUDA(r, w, k, v, -kk, kk * a)
        else:
            x = RWKV7_PYTORCH(r, w, k, v, -kk, kk * a, head_size=self.head_size)

        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
            .sum(dim=-1, keepdim=True)
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


########################################################################################################
# Official RWKV7 ChannelMix (x070)
########################################################################################################

class RWKV_CMix_x070(MyModule):
    """Official RWKV-7 ChannelMix from train_temp."""

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5 / (args.n_embd**0.5), 0.5 / (args.n_embd**0.5))
        self.value.weight.data.zero_()

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x

        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)


########################################################################################################
# Official RWKV7 Block
########################################################################################################

class Block(nn.Module):
    """Official RWKV-7 Block from train_temp."""

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first


########################################################################################################
# Args Helper
########################################################################################################

class Args:
    """Simple args class for using RWKV7 modules standalone."""

    def __init__(
        self,
        n_layer: int = 6,
        n_embd: int = 384,
        dim_att: int | None = None,
        head_size: int = 64,
        my_testing: str = "x070",
    ):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.dim_att = dim_att if dim_att is not None else n_embd
        self.head_size = head_size
        self.my_testing = my_testing


__all__ = [
    "HEAD_SIZE",
    "CHUNK_LEN",
    "_RWKV7_CLAMPW_AVAILABLE",
    "_load_rwkv7_clampw",
    "RWKV7_CLAMPW_CUDA_OP",
    "RWKV7_CLAMPW_CUDA",
    "RWKV7_PYTORCH",
    "RUN_WKV7",
    "RWKV_Tmix_x070",
    "RWKV_CMix_x070",
    "Block",
    "Args",
]
