"""CNN skip decoder with LUT-transformed image guidance skips."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.decoders.cnn_decoder_lut import GlobalLUTBranch
from src.models.decoders.cnn_skip_decoder import CNNSkipDecoder


class CNNLUTSkipDecoder(CNNSkipDecoder):
    """Use a learned LUT image as the decoder's multi-scale image skip.

    The decoder keeps the same residual CNN output path as ``CNNSkipDecoder``.
    The only architectural change is the image guidance branch: each upsample
    stage receives a projected ``LUT(img)`` feature instead of the raw image
    feature. This makes the LUT an intermediate skip signal rather than a
    separate final-output branch.
    """

    def __init__(
        self,
        input_dim: int,
        patch_size: int = 16,
        num_upsample_blocks: int = 4,
        base_channels: int = 64,
        residual_scale: float = 0.5,
        img_proj_channels: int = 3,
        predict_residual: bool = True,
        skip_enabled: bool = True,
        skip_fusion_mode: str = "gated_residual",
        skip_stages: list[int] = [1, 2, 3],
        img_skip_enabled: bool = True,
        img_skip_fusion_mode: str = "concat",
        color_naf_channels: list[int] = [64, 64, 128],
        film_stages: list[int] = [],
        film_cond_dim: int = 384,
        num_luts: int = 5,
        lut_size: int = 33,
        use_1d_lut: bool = True,
        lut_skip_blend: float = 1.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            patch_size=patch_size,
            num_upsample_blocks=num_upsample_blocks,
            base_channels=base_channels,
            residual_scale=residual_scale,
            img_proj_channels=img_proj_channels,
            predict_residual=predict_residual,
            skip_enabled=skip_enabled,
            skip_fusion_mode=skip_fusion_mode,
            skip_stages=skip_stages,
            img_skip_enabled=img_skip_enabled,
            img_skip_fusion_mode=img_skip_fusion_mode,
            color_naf_channels=color_naf_channels,
            film_stages=film_stages,
            film_cond_dim=film_cond_dim,
        )
        if not 0.0 <= lut_skip_blend <= 1.0:
            raise ValueError("lut_skip_blend must be in [0, 1]")

        self.global_lut_branch = GlobalLUTBranch(
            input_dim=input_dim,
            num_luts=num_luts,
            lut_size=lut_size,
            use_1d_lut=use_1d_lut,
        )
        self.lut_skip_blend = float(lut_skip_blend)

    def _make_skip_image(self, tokens: Tensor, img: Tensor) -> Tensor:
        lut_img = self.global_lut_branch(tokens, img).clamp(0.0, 1.0)
        if self.lut_skip_blend == 1.0:
            return lut_img
        if self.lut_skip_blend == 0.0:
            return img
        return (1.0 - self.lut_skip_blend) * img + self.lut_skip_blend * lut_img

    def forward(
        self,
        x: Tensor,
        h: int,
        w: int,
        img: Tensor | None = None,
        skip_dict: dict | None = None,
        hist_token: Tensor | None = None,
    ) -> Tensor:
        B = x.shape[0]
        scale = 2 ** len(self.upsample_blocks)
        if img is None:
            img = x.new_zeros(B, 3, h * scale, w * scale)
        _, _, H, W = img.shape

        skip_img = self._make_skip_image(x, img) if self.img_skip_enabled else img

        x = self.proj(x)
        x = x.transpose(1, 2).contiguous().reshape(B, -1, h, w)

        for i, block in enumerate(self.upsample_blocks):
            out_h = h * (2 ** (i + 1))
            out_w = w * (2 ** (i + 1))

            if self.img_skip_enabled:
                img_resized = F.interpolate(
                    skip_img,
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=False,
                )
                img_feat = self.img_proj(img_resized)
                x = block(x, img_feat)
            else:
                x = block(x)

            if self.skip_enabled and skip_dict is not None and i in self.skip_stages:
                if "color_naf" in skip_dict:
                    color_naf_skips = skip_dict["color_naf"]
                    stage_key = f"stage{i}"
                    fusion_key = f"color_naf_{i}"

                    if stage_key in color_naf_skips and fusion_key in self.skip_fusion_blocks:
                        skip_feat = color_naf_skips[stage_key]
                        if skip_feat.shape[-2:] != (out_h, out_w):
                            skip_feat = F.interpolate(
                                skip_feat,
                                size=(out_h, out_w),
                                mode="bilinear",
                                align_corners=False,
                            )
                        x = self.skip_fusion_blocks[fusion_key](x, [skip_feat], (out_h, out_w))

                elif "local" in skip_dict or "colour" in skip_dict:
                    fusion_key = f"legacy_{i}"
                    if fusion_key in self.skip_fusion_blocks:
                        skip_feats = []
                        if "local" in skip_dict:
                            local_dict = skip_dict["local"]
                            if i == 1 and "layer1" in local_dict:
                                skip_feats.append(local_dict["layer1"])
                            elif i == 2 and "layer0" in local_dict:
                                skip_feats.append(local_dict["layer0"])
                            elif i == 3 and "raw_input" in skip_dict:
                                skip_feats.append(skip_dict["raw_input"]["local"])

                        if "colour" in skip_dict:
                            colour_dict = skip_dict["colour"]
                            if i == 1 and "layer1" in colour_dict:
                                skip_feats.append(colour_dict["layer1"])
                            elif i == 2 and "layer0" in colour_dict:
                                skip_feats.append(colour_dict["layer0"])
                            elif i == 3 and "raw_input" in skip_dict:
                                skip_feats.append(skip_dict["raw_input"]["colour"])

                        if skip_feats:
                            x = self.skip_fusion_blocks[fusion_key](x, skip_feats, (out_h, out_w))

            if hist_token is not None and str(i) in self.film_blocks:
                x = self.film_blocks[str(i)](x, hist_token)

        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            x = x + self.align_refine(x)

        return torch.tanh(self.head(x)) * self.residual_scale


__all__ = ["CNNLUTSkipDecoder"]
