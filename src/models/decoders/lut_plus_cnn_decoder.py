"""LUT + CNN refinement decoder (PR3.2 of TokenLUT-BG plan).

Combines the global 3D LUT branch (Head A, same as PR1) with a CNN residual
refinement branch — implemented by reusing the existing baseline CNNDecoder.
The CNN's multi-scale image skip connections see the LUT-corrected image
(I_lut) instead of the raw input, so the CNN's job is simplified to
"refine the already globally-toned image" rather than "produce the entire
enhancement from scratch".

  out = clamp(I_lut + cnn_delta, 0, 1),  cnn_delta ∈ ±cnn_residual_scale

Motivation. PR3 (LUT + bilateral grid) reaches val PSNR 20.62 at ep 4
then collapses (val LPIPS doubles by ep 20). PR3.1 added TV regularization
on the bilateral grid, but with the conservative weight 1e-3 the penalty
stayed below 1% of the L1 loss and had no measurable effect — the val
trajectory was identical to PR3. PR3.2 replaces the unstable bilateral
grid with the baseline CNNDecoder, which is known to train stably for
32+ epochs on this pipeline (baseline reaches PSNR 20.76 without any
LPIPS regression). The CNN's 3×3 convolutions + tanh-bounded delta act
as built-in regularizers that the per-pixel bilateral affine grid lacks.

Output contract: full image (predict_residual=False), so the pipeline
must consume `clamp(decoder_output, 0, 1)` rather than `clamp(img + delta)`.

Init starts from identity:
  * basis LUTs are identity (Head A returns input image at step 0)
  * CNNDecoder's output head is zero-init (delta = 0 at step 0)
  * total: out = I_lut + 0 = input image
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.decoders.base import BaseDecoder
from src.models.decoders.cnn_decoder import CNNDecoder


# ----------------------------------------------------------------------
# LUT helper utilities
# ----------------------------------------------------------------------

def make_identity_lut(lut_dim: int) -> Tensor:
    """Return the identity 3D LUT, shape (3, D, D, D).

    LUT[c, i, j, k] outputs channel c when the input is
    (R = i / (D-1), G = j / (D-1), B = k / (D-1)).
    """
    coords = torch.linspace(0.0, 1.0, lut_dim)
    R, G, B = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([R, G, B], dim=0)  # (3, D, D, D)


def photometric_token(img: Tensor, n_bins: int = 6) -> Tensor:
    """Compute a (B, 6 + 3*n_bins) handcrafted photometric statistics token.

    Layout: [mean_R, mean_G, mean_B, std_R, std_G, std_B,
             hist_R(n_bins), hist_G(n_bins), hist_B(n_bins)].
    Each per-channel histogram is normalised to sum to 1 within that channel.
    """
    img_c = img.clamp(0.0, 1.0).float()
    B, C, H, W = img_c.shape

    mean = img_c.mean(dim=(2, 3))
    std = img_c.std(dim=(2, 3))

    bin_idx = (img_c * n_bins).long().clamp(max=n_bins - 1)
    bin_idx_flat = bin_idx.view(B, C, -1)
    hist = torch.zeros(B, C, n_bins, device=img.device, dtype=torch.float32)
    ones = torch.ones_like(bin_idx_flat, dtype=torch.float32)
    hist.scatter_add_(2, bin_idx_flat, ones)
    hist = (hist / float(H * W)).view(B, -1)

    return torch.cat([mean, std, hist], dim=-1)


def trilinear_lookup(lut: Tensor, img: Tensor) -> Tensor:
    """Apply per-sample 3D LUT to image via 5D grid_sample.

    Args:
        lut: (B, 3, D, D, D) LUT, axes ordered (R, G, B) along (D, H, W).
        img: (B, 3, H, W) image in [0, 1], channel order (R, G, B).
    """
    rgb = img.clamp(0.0, 1.0) * 2.0 - 1.0  # (B, 3, H, W)

    # grid_sample's last-dim order on a 5D input is (W, H, D). Our LUT axes
    # (R, G, B) are mapped to (D, H, W), so the sample point is (B, G, R).
    grid = torch.stack(
        [rgb[:, 2], rgb[:, 1], rgb[:, 0]], dim=-1
    ).unsqueeze(1)  # (B, 1, H, W, 3)

    out_5d = F.grid_sample(
        lut, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (B, 3, 1, H, W)
    return out_5d.squeeze(2)  # (B, 3, H, W)


class LUTPlusCNNDecoder(BaseDecoder):
    """3D LUT (global) + CNN (local refinement) hybrid decoder.

    Args:
        input_dim: Bottleneck output dim (injected by pipeline).
        patch_size: Encoder patch size.

      LUT branch (Head A):
        num_basis: Number of basis LUTs.
        lut_dim: LUT side length.
        mlp_hidden: Hidden dim for the basis-weight MLP.
        random_init_scale: Std of perturbation on non-anchor basis LUTs.
        use_photometric_token: Concat handcrafted RGB stats to global_vec.
        photo_n_bins: Histogram bins per channel for photometric token.

      CNN branch (Head R, passed straight through to CNNDecoder):
        cnn_num_upsample_blocks: Number of 2× PixelShuffle stages.
        cnn_base_channels: Channel count after the initial token projection.
            64 keeps the CNN smaller than the baseline (128) so it acts as a
            refiner rather than dominating Head A.
        cnn_residual_scale: Maximum |delta| produced by the CNN. Smaller
            than baseline (0.5) on purpose — Head A already does the heavy
            color lifting, the CNN only needs to clean up fine residuals.
        cnn_img_proj_channels: Channels for the CNN's per-stage img skip.
    """

    def __init__(
        self,
        input_dim: int,
        patch_size: int = 16,
        # LUT branch
        num_basis: int = 3,
        lut_dim: int = 17,
        mlp_hidden: int = 64,
        random_init_scale: float = 0.15,
        use_photometric_token: bool = False,
        photo_n_bins: int = 6,
        # CNN refinement branch
        cnn_num_upsample_blocks: int = 4,
        cnn_base_channels: int = 64,
        cnn_residual_scale: float = 0.3,
        cnn_img_proj_channels: int = 3,
    ) -> None:
        super().__init__()
        # Pipeline contract: full image, not residual delta.
        self.predict_residual = False

        self.patch_size = patch_size
        self.num_basis = num_basis
        self.lut_dim = lut_dim
        self.use_photometric_token = use_photometric_token
        self.photo_n_bins = photo_n_bins
        photo_dim = (6 + 3 * photo_n_bins) if use_photometric_token else 0

        # ---- Head A: LUT branch ----
        identity = make_identity_lut(lut_dim)
        basis = identity.unsqueeze(0).expand(num_basis, -1, -1, -1, -1).clone()
        if num_basis > 1:
            noise = torch.randn(num_basis - 1, *identity.shape) * random_init_scale
            basis[1:] = basis[1:] + noise
        self.basis_lut = nn.Parameter(basis)

        mlp_in_dim = input_dim + photo_dim
        self.weight_mlp = nn.Sequential(
            nn.LayerNorm(mlp_in_dim),
            nn.Linear(mlp_in_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_basis),
        )
        nn.init.zeros_(self.weight_mlp[-1].weight)
        nn.init.zeros_(self.weight_mlp[-1].bias)

        # ---- Head R: CNN refinement (reuses baseline CNNDecoder verbatim) ----
        # CNNDecoder already zero-inits its output head, so initial delta = 0
        # and total output equals I_lut at step 0.
        self.cnn = CNNDecoder(
            input_dim=input_dim,
            patch_size=patch_size,
            num_upsample_blocks=cnn_num_upsample_blocks,
            base_channels=cnn_base_channels,
            residual_scale=cnn_residual_scale,
            img_proj_channels=cnn_img_proj_channels,
        )
        assert self.cnn.predict_residual is True, (
            "Inner CNNDecoder must run in residual mode; the outer wrapper "
            "performs the addition and clamp."
        )

        # Keep the wrapper close to identity at init while still allowing
        # gradients to reach the CNN trunk on the very first update.
        nn.init.normal_(self.cnn.head[-1].weight, mean=0.0, std=1e-6)
        nn.init.zeros_(self.cnn.head[-1].bias)

    def forward(
        self, x: Tensor, h: int, w: int, img: Tensor | None = None
    ) -> Tensor:
        assert img is not None, "LUTPlusCNNDecoder requires the original image"
        B = x.shape[0]

        # === Head A: LUT branch ===
        global_vec = x.mean(dim=1).float()
        if self.use_photometric_token:
            photo = photometric_token(img, n_bins=self.photo_n_bins)
            global_vec = torch.cat([global_vec, photo.to(global_vec.dtype)], dim=-1)

        weights = F.softmax(self.weight_mlp(global_vec), dim=-1)
        fused_lut = (
            weights.view(B, self.num_basis, 1, 1, 1, 1)
            * self.basis_lut.unsqueeze(0)
        ).sum(dim=1)
        I_lut = trilinear_lookup(fused_lut, img.to(fused_lut.dtype))

        # === Head R: CNN residual on top of LUT output ===
        # Pass I_lut (already globally toned) as the CNN's image skip source.
        # CNN sees a smoothly color-corrected image at every scale, so its
        # only job is to add a small per-pixel correction.
        delta = self.cnn(x, h, w, img=I_lut)            # ±cnn_residual_scale

        return (I_lut + delta).clamp(0.0, 1.0)


__all__ = ["LUTPlusCNNDecoder"]
