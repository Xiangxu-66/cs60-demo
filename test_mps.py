"""MPS compatibility test: DINOv2 encoder + CNN bottleneck + CNN decoder.

100 synthetic images, batch_size=4, image_size=224x224.
Tests both forward and backward passes.
"""
from __future__ import annotations

import time
import traceback
import torch

# ── Device ────────────────────────────────────────────────────────────────────
if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    raise SystemExit("MPS not available on this machine.")
device = torch.device("mps")
print(f"Device: {device}\n")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0  Isolate the failing op before running the full pipeline
# ══════════════════════════════════════════════════════════════════════════════
print("── Phase 0: isolate transpose+reshape backward ──")

def _test_op(label: str, fn):
    try:
        fn()
        print(f"  [PASS] {label}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}")
        print(f"         {type(e).__name__}: {e}")
        return False

# Decoder's exact pattern: transpose(1,2).reshape(...)
def _decoder_pattern():
    x = torch.randn(4, 256, 384, requires_grad=True, device=device)
    out = x.transpose(1, 2).reshape(4, 384, 16, 16)
    out.sum().backward()

# With .contiguous() fix
def _decoder_pattern_fixed():
    x = torch.randn(4, 256, 384, requires_grad=True, device=device)
    out = x.transpose(1, 2).contiguous().reshape(4, 384, 16, 16)
    out.sum().backward()

r0 = _test_op("decoder  transpose+reshape          backward (original)", _decoder_pattern)
r1 = _test_op("decoder  transpose+.contiguous()+reshape backward (fixed) ", _decoder_pattern_fixed)
print()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1  Build pipeline
# ══════════════════════════════════════════════════════════════════════════════
print("── Phase 1: build DINOv2 + CNN bottleneck + CNN decoder ──")
from src.models.encoders.dinov2 import DINOv2Encoder
from src.models.bottlenecks.cnn_bottleneck import CNNBottleneck
from src.models.decoders.cnn_decoder import CNNDecoder
from src.models.pipeline import ImageEnhancementPipeline

encoder    = DINOv2Encoder(pretrained=False, frozen=True, img_size=224, patch_size=14, embed_dim=768)
bottleneck = CNNBottleneck(input_dim=768, hidden_dim=384, num_layers=4)
decoder    = CNNDecoder(input_dim=384, patch_size=14, num_upsample_blocks=4, base_channels=64)
pipeline   = ImageEnhancementPipeline(encoder, bottleneck, decoder).to(device)
print("  Pipeline built.\n")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1b  Component-level backward isolation
# ══════════════════════════════════════════════════════════════════════════════
print("── Phase 1b: component backward isolation ──")

# Bottleneck only
def _test_bottleneck():
    bot = CNNBottleneck(input_dim=768, hidden_dim=384, num_layers=4).to(device).train()
    x = torch.randn(4, 256, 768, requires_grad=True, device=device)
    out = bot(x)
    out.sum().backward()

# Decoder only
def _test_decoder():
    dec = CNNDecoder(input_dim=384, patch_size=14, num_upsample_blocks=4, base_channels=64).to(device).train()
    x = torch.randn(4, 256, 384, requires_grad=True, device=device)
    out = dec(x, h=16, w=16)
    out.sum().backward()

# Bottleneck → Decoder
def _test_bottleneck_decoder():
    bot = CNNBottleneck(input_dim=768, hidden_dim=384, num_layers=4).to(device).train()
    dec = CNNDecoder(input_dim=384, patch_size=14, num_upsample_blocks=4, base_channels=64).to(device).train()
    x = torch.randn(4, 256, 768, requires_grad=True, device=device)
    feat = bot(x)
    out = dec(feat, h=16, w=16)
    out.sum().backward()

_test_op("bottleneck              backward", _test_bottleneck)
_test_op("decoder                 backward", _test_decoder)
_test_op("bottleneck → decoder    backward", _test_bottleneck_decoder)

# Drill into bottleneck: pinpoint which rearrange fails
print()
print("── Phase 1c: drill into bottleneck rearrange ops ──")
from einops import rearrange as einops_rearrange

def _rearrange_bhwc_to_bchw():
    # First rearrange in bottleneck: (b, n, c) → (b, c, h, w)  via proj_in output
    x = torch.randn(4, 256, 384, requires_grad=True, device=device)
    out = einops_rearrange(x, "b (h w) c -> b c h w", h=16, w=16)
    out.sum().backward()

def _rearrange_bchw_to_bnc_clean():
    # Second rearrange with contiguous input (e.g. fresh randn)
    x = torch.randn(4, 384, 16, 16, requires_grad=True, device=device)
    out = einops_rearrange(x, "b c h w -> b (h w) c")
    out.sum().backward()

def _rearrange_bchw_to_bnc_after_conv():
    # Second rearrange with non-contiguous input (output of conv → BN → GELU → Dropout2d)
    import torch.nn as nn
    block = nn.Sequential(
        nn.Conv2d(384, 384, 3, padding=1, bias=False),
        nn.BatchNorm2d(384),
        nn.GELU(),
        nn.Dropout2d(0.1),
    ).to(device).train()
    x = torch.randn(4, 384, 16, 16, requires_grad=True, device=device)
    h = block(x)
    out = einops_rearrange(h, "b c h w -> b (h w) c")
    out.sum().backward()

_test_op("rearrange  (b n c) → (b c h w)  backward", _rearrange_bhwc_to_bchw)
_test_op("rearrange  (b c h w) → (b n c)  clean input backward", _rearrange_bchw_to_bnc_clean)
_test_op("rearrange  (b c h w) → (b n c)  after conv+BN+GELU+Dropout2d backward", _rearrange_bchw_to_bnc_after_conv)

# Use anomaly detection to get the exact forward op that produces the bad grad
print()
print("── Phase 1d: anomaly detection on bottleneck ──")
with torch.autograd.set_detect_anomaly(True):
    try:
        bot = CNNBottleneck(input_dim=768, hidden_dim=384, num_layers=4).to(device).train()
        x   = torch.randn(4, 256, 768, requires_grad=True, device=device)
        out = bot(x)
        out.sum().backward()
        print("  [PASS] bottleneck with anomaly detection")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2  100-image forward + backward
# ══════════════════════════════════════════════════════════════════════════════
print("── Phase 2: 100 images, batch=4, 224×224 ──")
N_IMAGES, BATCH_SIZE, IMG_SIZE = 100, 4, 224
images    = torch.rand(N_IMAGES, 3, IMG_SIZE, IMG_SIZE)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(pipeline.trainable_parameters(), lr=1e-4)
pipeline.train()

batches_done, total_loss, errors = 0, 0.0, []
t0 = time.perf_counter()

for start in range(0, N_IMAGES, BATCH_SIZE):
    x      = images[start : start + BATCH_SIZE].to(device)
    target = torch.rand_like(x)
    try:
        optimizer.zero_grad()
        pred = pipeline(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss   += loss.item()
        batches_done += 1
        print(f"  batch {batches_done:3d}/{N_IMAGES // BATCH_SIZE}  loss={loss.item():.4f}")
    except Exception as e:
        errors.append((batches_done + 1, traceback.format_exc()))
        print(f"  batch {batches_done + 1:3d}  [ERROR] {type(e).__name__}: {e}")
        break

torch.mps.synchronize()
elapsed = time.perf_counter() - t0

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"  Phase 0 original pattern : {'PASS' if r0 else 'FAIL'}")
print(f"  Phase 0 fixed pattern    : {'PASS' if r1 else 'FAIL'}")
print(f"  Phase 2 batches done     : {batches_done}/{N_IMAGES // BATCH_SIZE}")
if batches_done:
    print(f"  Phase 2 avg loss         : {total_loss / batches_done:.4f}")
    print(f"  Phase 2 time             : {elapsed:.2f}s  ({elapsed/batches_done*1000:.1f}ms/batch)")
if errors:
    print(f"\n  First error (batch {errors[0][0]}):")
    for line in errors[0][1].splitlines():
        print(f"    {line}")
status = "ALL PASS" if (not errors and batches_done == N_IMAGES // BATCH_SIZE) else "FAIL"
print(f"\n  MPS result: {status}")
print('='*55)
