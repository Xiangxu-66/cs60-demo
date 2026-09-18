"""方案C: 神经 LUT 融合版图像增强 Decoder.

核心思想：模拟专业修图的物理过程
    1. 全局调色：通过可学习的 1D/3D LUT 实现色彩风格转换
    2. 局部修饰：通过空间权重图和残差实现区域化调整
    3. 自适应融合：根据图像内容动态混合全局和局部

Architecture:
                      ┌─────────────────┐
                      │   Input Image   │
                      └────────┬────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
        ┌─────────────────┐           ┌─────────────────┐
        │   DINOv3 Enc.   │           │     Input       │
        │  (Semantic)     │           │   (for LUT)      │
        └────────┬────────┘           └────────┬────────┘
                 │                             │
                 ▼                             ▼
        ┌─────────────────┐           ┌─────────────────┐
        │  RWKV Bottleneck│           │  LUT Fusion     │
        │   (B, N, C)     │           │   (Global)      │
        └────────┬────────┘           └────────┬────────┘
                 │                             │
                 │ α weights                   │ LUT_output
                 └──────────┬──────────────────┘
                            ▼
                  ┌─────────────────┐
                  │ Spatial Branch  │
                  │   CNN Decoder   │
                  │                 │
                  │ → M (Weight Map)│
                  │ → R (Residual)  │
                  └────────┬────────┘
                           │
                           ▼
             Output = (1-M)·LUT + M·(Input+R)

这种设计的优势：
    - 全局色调一致性好（LUT 保证）
    - 局部细节自适应（空间调制）
    - 高度可解释（可视化 LUT 和权重图）
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.decoders.base import BaseDecoder


class ChanLayerNorm(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps (B, C, H, W).

    Permutes to (B, H, W, C), applies LayerNorm(C), permutes back.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1).float()).to(x.dtype).permute(0, 3, 1, 2)


class FastLUTSampling(nn.Module):
    """快速 LUT 采样，支持 1D 和 3D LUT.

    对于 1D LUT：每通道独立进行线性插值
    对于 3D LUT：使用简化的三线性插值近似

    推荐使用 1D LUT（use_1d_lut=True），因为：
    1. 计算效率高
    2. 训练稳定
    3. 对于色彩增强任务效果足够好
    """

    def __init__(self, lut_size: int = 33) -> None:
        super().__init__()
        self.lut_size = lut_size

    def forward(self, rgb: Tensor, lut: Tensor) -> Tensor:
        """LUT 采样.

        Args:
            rgb: (B, 3, H, W) 输入图像 [0, 1]
            lut: (B, 3, D) 1D LUT 或 (B, D, D, D, 3) 3D LUT

        Returns:
            (B, 3, H, W) 映射后的图像
        """
        B, C, H, W = rgb.shape

        # 如果是 1D LUT (B, 3, D)
        if lut.dim() == 3:
            return self._sample_1d_lut(rgb, lut)

        # 如果是 3D LUT
        return self._sample_3d_lut_approx(rgb, lut)

    def _sample_1d_lut(self, rgb: Tensor, lut: Tensor) -> Tensor:
        """从 1D LUT 采样（每通道独立）。

        使用线性插值在离散的 LUT 点之间进行采样。
        """
        B, C, H, W = rgb.shape
        D = lut.shape[-1]

        # 缩放到索引空间 [0, D-1]
        idx = rgb * (D - 1)
        idx_floor = torch.floor(idx).long()
        idx_ceil = torch.ceil(idx).long()
        idx_floor = idx_floor.clamp(0, D - 1)
        idx_ceil = idx_ceil.clamp(0, D - 1)

        # 插值权重
        weight = idx - idx_floor.float()

        # 收集 LUT 值并插值
        output = []
        for c in range(C):
            lut_c = lut[:, c]  # (B, D)
            idx_floor_c = idx_floor[:, c].reshape(B, -1)  # (B, H*W)
            idx_ceil_c = idx_ceil[:, c].reshape(B, -1)

            val_floor = torch.gather(lut_c, 1, idx_floor_c)
            val_ceil = torch.gather(lut_c, 1, idx_ceil_c)

            # 线性插值
            weight_c = weight[:, c].reshape(B, -1)
            val = val_floor * (1 - weight_c) + val_ceil * weight_c

            output.append(val.reshape(B, 1, H, W))

        return torch.cat(output, dim=1)

    def _sample_3d_lut_approx(self, rgb: Tensor, lut: Tensor) -> Tensor:
        """真正的 3D LUT 采样，使用完整的三线性插值。

        3D LUT 格式：(B, D, D, D, 3)
        对于每个像素的 (R, G, B) 值，在 3D 网格中找到 8 个角点并插值。

        三线性插值公式：
            v = (1-dr)*(1-dg)*(1-db)*v000
              + dr*(1-dg)*(1-db)*v100
              + (1-dr)*dg*(1-db)*v010
              + dr*dg*(1-db)*v110
              + (1-dr)*(1-dg)*db*v001
              + dr*(1-dg)*db*v101
              + (1-dr)*dg*db*v011
              + dr*dg*db*v111
        其中 (dr, dg, db) 是小数部分，vijk 是 8 个角点的值。
        """
        B, C, H, W = rgb.shape
        D = self.lut_size

        # 确保格式为 (B, D, D, D, 3)
        if lut.shape[1] == 3:
            lut = lut.permute(0, 2, 3, 4, 1)

        # 将 RGB 坐标缩放到 LUT 网格空间 [0, D-1]
        coords = rgb * (D - 1)  # (B, 3, H, W)

        # 获取整数和小数部分
        coords_floor = torch.floor(coords).long().clamp(0, D - 1)
        coords_ceil = torch.ceil(coords).long().clamp(0, D - 1)

        # 小数部分，用于插值权重
        frac = coords - coords_floor.float()  # (B, 3, H, W)
        r_frac, g_frac, b_frac = frac[:, 0], frac[:, 1], frac[:, 2]

        # 展平空间维度以便索引
        HW = H * W
        coords_floor = coords_floor.reshape(B, 3, HW)  # (B, 3, HW)
        coords_ceil = coords_ceil.reshape(B, 3, HW)
        r_frac = r_frac.reshape(B, 1, HW)
        g_frac = g_frac.reshape(B, 1, HW)
        b_frac = b_frac.reshape(B, 1, HW)

        # 准备输出
        output = torch.zeros(B, 3, HW, device=rgb.device, dtype=rgb.dtype)

        # 对每个颜色通道进行 3D LUT 查询
        for c in range(3):
            lut_c = lut[:, :, :, :, c]  # (B, D, D, D)

            # 获取 8 个角点的索引
            r0, g0, b0 = coords_floor[:, 0], coords_floor[:, 1], coords_floor[:, 2]
            r1, g1, b1 = coords_ceil[:, 0], coords_ceil[:, 1], coords_ceil[:, 2]

            # 计算线性索引：idx = r * D * D + g * D + b
            D2 = D * D
            idx000 = r0 * D2 + g0 * D + b0
            idx001 = r0 * D2 + g0 * D + b1
            idx010 = r0 * D2 + g1 * D + b0
            idx011 = r0 * D2 + g1 * D + b1
            idx100 = r1 * D2 + g0 * D + b0
            idx101 = r1 * D2 + g0 * D + b1
            idx110 = r1 * D2 + g1 * D + b0
            idx111 = r1 * D2 + g1 * D + b1

            # 从展平的 LUT 中收集 8 个角点的值
            lut_flat = lut_c.reshape(B, -1)  # (B, D*D*D)

            v000 = torch.gather(lut_flat, 1, idx000.clamp(0, D*D*D-1))
            v001 = torch.gather(lut_flat, 1, idx001.clamp(0, D*D*D-1))
            v010 = torch.gather(lut_flat, 1, idx010.clamp(0, D*D*D-1))
            v011 = torch.gather(lut_flat, 1, idx011.clamp(0, D*D*D-1))
            v100 = torch.gather(lut_flat, 1, idx100.clamp(0, D*D*D-1))
            v101 = torch.gather(lut_flat, 1, idx101.clamp(0, D*D*D-1))
            v110 = torch.gather(lut_flat, 1, idx110.clamp(0, D*D*D-1))
            v111 = torch.gather(lut_flat, 1, idx111.clamp(0, D*D*D-1))

            # 三线性插值
            # 确保权重具有正确的形状: (B, 1, HW)
            r_frac = r_frac.unsqueeze(1) if r_frac.dim() == 2 else r_frac
            g_frac = g_frac.unsqueeze(1) if g_frac.dim() == 2 else g_frac
            b_frac = b_frac.unsqueeze(1) if b_frac.dim() == 2 else b_frac

            # 确保角点值具有正确的形状: (B, 1, HW)
            v000 = v000.unsqueeze(1) if v000.dim() == 2 else v000
            v001 = v001.unsqueeze(1) if v001.dim() == 2 else v001
            v010 = v010.unsqueeze(1) if v010.dim() == 2 else v010
            v011 = v011.unsqueeze(1) if v011.dim() == 2 else v011
            v100 = v100.unsqueeze(1) if v100.dim() == 2 else v100
            v101 = v101.unsqueeze(1) if v101.dim() == 2 else v101
            v110 = v110.unsqueeze(1) if v110.dim() == 2 else v110
            v111 = v111.unsqueeze(1) if v111.dim() == 2 else v111

            # 先在 R 方向插值
            v00 = v000 * (1 - r_frac) + v100 * r_frac
            v01 = v001 * (1 - r_frac) + v101 * r_frac
            v10 = v010 * (1 - r_frac) + v110 * r_frac
            v11 = v011 * (1 - r_frac) + v111 * r_frac

            # 再在 G 方向插值
            v0 = v00 * (1 - g_frac) + v10 * g_frac
            v1 = v01 * (1 - g_frac) + v11 * g_frac

            # 最后在 B 方向插值
            v = v0 * (1 - b_frac) + v1 * b_frac

            output[:, c, :] = v.squeeze(1)

        return output.reshape(B, 3, H, W)


class GlobalLUTBranch(nn.Module):
    """全局 LUT 分支.

    从 RWKV 特征预测 LUT 融合权重，应用一组基 LUT 实现全局调色。

    设计选择：
        1. 使用可学习的基 LUT（而非固定）
        2. RWKV → MLP → 融合权重 α
        3. LUT_fused = Σ α_i * LUT_i

    Args:
        input_dim: RWKV bottleneck 输出维度
        num_luts: 基 LUT 数量
        lut_size: LUT 网格尺寸
        hidden_dim: 权重预测器隐藏层维度
        use_1d_lut: 使用 1D LUT（推荐）还是 3D LUT
    """

    def __init__(
        self,
        input_dim: int,
        num_luts: int = 5,
        lut_size: int = 33,
        hidden_dim: int = 128,
        use_1d_lut: bool = True,
    ) -> None:
        super().__init__()
        self.num_luts = num_luts
        self.lut_size = lut_size
        self.use_1d_lut = use_1d_lut

        # RWKV 特征 → LUT 权重
        self.weight_predictor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_luts),
            nn.Softmax(dim=-1),  # 权重和为 1
        )

        # 初始化最后一层使初始权重接近均匀
        # 这样融合后的 LUT 更接近单位映射
        nn.init.zeros_(self.weight_predictor[-2].weight)
        nn.init.zeros_(self.weight_predictor[-2].bias)

        # 可学习的基 LUT
        if use_1d_lut:
            # 初始化为风格化曲线（鼓励更强的增强）
            base_luts = torch.zeros(num_luts, 3, lut_size)

            # 创建均匀分布的坐标
            coords = torch.linspace(0, 1, lut_size)

            for i in range(num_luts):
                # 为每个 LUT 创建不同的风格
                style_factor = i / max(num_luts - 1, 1)  # 0 到 1

                for c in range(3):
                    if c == 0:  # R 通道：略偏 S 曲线（增强对比）
                        base_luts[i, c] = coords + 0.1 * torch.sin(coords * 3.14159)
                    elif c == 1:  # G 通道：轻微调整
                        base_luts[i, c] = coords ** (0.8 + 0.4 * style_factor)
                    else:  # B 通道：更多变化
                        base_luts[i, c] = 1 - (1 - coords) ** (0.7 + 0.6 * style_factor)

            # 添加小的随机扰动
            base_luts += torch.randn_like(base_luts) * 0.03
            base_luts = base_luts.clamp(0, 1)

            self.base_luts = nn.Parameter(base_luts)
        else:
            # 3D LUT: 初始化策略
            # 第一个 LUT 是单位映射（identity），其他是小的残差变化
            coords = torch.linspace(0, 1, lut_size)
            grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'), dim=-1)
            # grid: (D, D, D, 3)

            # 初始化所有 LUT
            base_luts = torch.zeros(num_luts, lut_size, lut_size, lut_size, 3)

            for i in range(num_luts):
                if i == 0:
                    # 第一个 LUT：完全的单位映射
                    base_luts[i] = grid.clone()
                else:
                    # 其他 LUT：在单位映射基础上添加小的风格化变化
                    base_luts[i] = grid.clone()

                    # 为每个 LUT 添加不同的风格
                    style_factor = i / max(num_luts - 1, 1)

                    # S 曲线增强对比
                    s_curve = torch.sin(coords * 3.14159 / 2) ** 2

                    # 应用到每个通道，强度随 style_factor 变化
                    for c in range(3):
                        strength = 0.05 * style_factor * (1 if c == 0 else -1 if c == 2 else 0.5)
                        base_luts[i, :, :, :, c] += strength * s_curve.unsqueeze(-1)

                    # 添加非常小的随机扰动（保持稳定性）
                    base_luts[i] += torch.randn_like(base_luts[i]) * 0.005

            # Clamp 到有效范围
            base_luts = base_luts.clamp(0, 1)

            self.base_luts = nn.Parameter(base_luts)

            # 保存单位网格作为参考
            self.register_buffer('identity_grid', grid)  # (D, D, D, 3)

        self.sampler = FastLUTSampling(lut_size)

    def forward(self, bottleneck_feat: Tensor, img: Tensor) -> Tensor:
        """应用 LUT 全局调色.

        Args:
            bottleneck_feat: (B, N, C) RWKV 输出特征
            img: (B, 3, H, W) 输入图像

        Returns:
            (B, 3, H, W) LUT 调色后的图像
        """
        B = bottleneck_feat.shape[0]

        # 预测 LUT 权重（使用第一个 token 的特征）
        weights = self.weight_predictor(bottleneck_feat[:, 0])  # (B, num_luts)

        # 融合基 LUT
        if self.use_1d_lut:
            # base_luts: (num_luts, 3, D) → weighted: (B, 3, D)
            weights_expanded = weights.view(B, self.num_luts, 1, 1)
            fused_lut = (self.base_luts.unsqueeze(0) * weights_expanded).sum(dim=1)
        else:
            # base_luts: (num_luts, D, D, D, 3) → weighted: (B, D, D, D, 3)
            weights_expanded = weights.view(B, self.num_luts, 1, 1, 1, 1)
            fused_lut = (self.base_luts.unsqueeze(0) * weights_expanded).sum(dim=1)

        # 应用 LUT
        lut_output = self.sampler(img, fused_lut)

        return lut_output


class SkipAttentionFusion(nn.Module):
    """Skip 特征注意力融合.

    使用通道注意力自适应地融合 decoder 和 skip 特征。
    """

    def __init__(
        self,
        dec_ch: int,
        skip_ch: int,
        reduction: int = 8,
    ) -> None:
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_ch, dec_ch, 1, bias=True)

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dec_ch * 2, dec_ch * 2 // reduction, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(dec_ch * 2 // reduction, dec_ch, 1, bias=True),
            nn.Sigmoid(),
        )

        self.norm = ChanLayerNorm(dec_ch)
        self.fuse = nn.Conv2d(dec_ch, dec_ch, 3, padding=1, bias=True)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """融合 skip 特征.

        Args:
            x: Decoder features (B, C, H, W)
            skip: Skip features (B, C_skip, H_skip, W_skip)

        Returns:
            Fused features (B, C, H, W)
        """
        skip = self.skip_proj(skip)

        # 对齐空间尺寸
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(
                skip, size=x.shape[-2:], mode='bilinear', align_corners=False
            )

        # 通道注意力
        concat = torch.cat([x, skip], dim=1)
        att = self.channel_att(concat)

        # 融合
        fused = self.fuse(self.norm(x + skip * att))

        return fused


class ResidualRefineBlock(nn.Module):
    """残差细化块.

    使用可学习的增益参数控制残差强度。
    """

    def __init__(
        self,
        channels: int,
        residual_gain_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ChanLayerNorm(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )
        self.gain = nn.Parameter(torch.full((1, channels, 1, 1), residual_gain_init))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.gain.to(dtype=x.dtype) * self.net(x)


class SpatialBranch(nn.Module):
    """空间分支：预测空间权重图 M 和局部残差 R.

    使用 CNN Decoder + DINO skip 特征：
        - M: (B, 1, H, W) 空间权重图，决定全局 vs 局部
        - R: (B, 3, H, W) 局部残差修正

    Args:
        input_dim: 输入特征维度
        patch_size: Encoder patch size
        num_upsample_blocks: 上采样阶段数
        base_channels: 基础通道数
        skip_channels: Skip 通道数列表
        use_skip_attention: 是否使用 skip 注意力融合
    """

    def __init__(
        self,
        input_dim: int,
        patch_size: int = 16,
        num_upsample_blocks: int = 4,
        base_channels: int = 64,
        skip_channels: list[int] | None = None,
        use_skip_attention: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_upsample_blocks = num_upsample_blocks
        self.use_skip_attention = use_skip_attention

        # Token 投影
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, base_channels),
            nn.GELU(),
        )

        # 通道配置
        self.channels = self._build_channel_schedule(base_channels, num_upsample_blocks)

        # Skip 融合配置
        if skip_channels is None:
            skip_channels = [768] * num_upsample_blocks
        self.skip_channels = skip_channels

        # PixelShuffle 上采样块（不包括 skip 融合）
        num_pixelshuffle = num_upsample_blocks - 1
        self.pixelshuffle_upsample = nn.ModuleList()
        self.pixelshuffle_refine = nn.ModuleList()
        self.pixelshuffle_skip_fusion = nn.ModuleList()

        for i in range(num_pixelshuffle):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            skip_ch = skip_channels[i] if i < len(skip_channels) else None

            # 上采样
            self.pixelshuffle_upsample.append(
                nn.Sequential(
                    ChanLayerNorm(in_ch),
                    nn.Conv2d(in_ch, out_ch * 4, 3, padding=1, bias=True),
                    nn.PixelShuffle(2),
                    nn.GELU(),
                )
            )

            # Skip 融合模块
            if skip_ch is not None and use_skip_attention:
                self.pixelshuffle_skip_fusion.append(
                    SkipAttentionFusion(dec_ch=out_ch, skip_ch=skip_ch)
                )
            else:
                self.pixelshuffle_skip_fusion.append(None)

            # 残差细化
            self.pixelshuffle_refine.append(
                ResidualRefineBlock(out_ch, residual_gain_init=0.1)
            )

        # 最终块：投影 + 上采样到目标尺寸
        final_in_ch = self.channels[num_pixelshuffle]
        final_out_ch = self.channels[-1]

        self.final_proj = nn.Sequential(
            ChanLayerNorm(final_in_ch),
            nn.Conv2d(final_in_ch, final_out_ch, 3, padding=1, bias=True),
            nn.GELU(),
        )

        # 最终 skip 融合
        final_skip_ch = skip_channels[-1] if num_upsample_blocks > 0 else None
        if final_skip_ch is not None and use_skip_attention:
            self.final_skip_fusion = SkipAttentionFusion(
                dec_ch=final_out_ch, skip_ch=final_skip_ch
            )
        else:
            self.final_skip_fusion = None

        # 最终残差细化（两个块）
        self.final_refine = nn.Sequential(
            ResidualRefineBlock(final_out_ch, residual_gain_init=0.1),
            ResidualRefineBlock(final_out_ch, residual_gain_init=0.1),
        )

        # 输出：M (权重图) 和 R (残差)
        self.weight_head = nn.Sequential(
            ChanLayerNorm(self.channels[-1]),
            nn.Conv2d(self.channels[-1], self.channels[-1] // 2, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.channels[-1] // 2, 1, 1, bias=True),
            nn.Sigmoid(),  # M ∈ [0, 1]
        )

        self.residual_head = nn.Sequential(
            ChanLayerNorm(self.channels[-1]),
            nn.Conv2d(self.channels[-1], self.channels[-1], 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.channels[-1], 3, 1, bias=True),
        )
        # 零初始化 residual head，使训练开始时残差接近 0
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def _build_channel_schedule(base: int, num_blocks: int) -> list[int]:
        """构建通道配置."""
        channels = [base]
        for i in range(1, num_blocks + 1):
            ratio = 1.0 - (0.25 * i / num_blocks)
            ch = max(round(base * ratio), 32)
            channels.append(ch)
        return channels

    def forward(
        self,
        x: Tensor,
        h: int,
        w: int,
        vit_skips: list[Tensor] | None = None,
        vit_h: int | None = None,
        vit_w: int | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """预测空间权重图和残差.

        Args:
            x: (B, N, C) token features
            h, w: token grid size
            vit_skips: DINO skip features
            vit_h, vit_w: DINO feature size
            target_size: (H, W) 目标输出尺寸

        Returns:
            M: (B, 1, H, W) 空间权重图
            R: (B, 3, H, W) 局部残差
        """
        B = x.shape[0]
        vit_h = vit_h or h
        vit_w = vit_w or w

        # Token → feature map
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, -1, h, w)

        # 处理 skip
        if vit_skips is None:
            vit_skips = [None] * self.num_upsample_blocks

        # 适配 skip 数量
        if len(vit_skips) < self.num_upsample_blocks:
            vit_skips = list(vit_skips) + [None] * (self.num_upsample_blocks - len(vit_skips))
        elif len(vit_skips) > self.num_upsample_blocks:
            vit_skips = vit_skips[:self.num_upsample_blocks]

        # 归一化 skip 特征到 2D 格式
        normalized_skips = []
        for skip in vit_skips:
            if skip is None:
                normalized_skips.append(None)
            elif skip.dim() == 3:
                # (B, N, C) -> (B, C, H, W)
                C_skip = skip.shape[-1]
                N_skip = skip.shape[1]
                h_skip = w_skip = int(N_skip ** 0.5)
                if h_skip * w_skip != N_skip:
                    h_skip, w_skip = vit_h, vit_w
                normalized_skips.append(
                    skip.transpose(1, 2).reshape(B, -1, h_skip, w_skip)
                )
            else:
                normalized_skips.append(skip)

        # PixelShuffle 上采样
        for i in range(len(self.pixelshuffle_upsample)):
            # 上采样
            x = self.pixelshuffle_upsample[i](x)

            # Skip 融合
            skip_feat = normalized_skips[i] if i < len(normalized_skips) else None
            skip_fusion = self.pixelshuffle_skip_fusion[i]

            if skip_fusion is not None and skip_feat is not None:
                x = skip_fusion(x, skip_feat)

            # 残差细化
            x = self.pixelshuffle_refine[i](x)

        # 最终块：投影
        x = self.final_proj(x)

        # 上采样到目标尺寸
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        # 最终 skip 融合
        if self.final_skip_fusion is not None:
            skip_feat = normalized_skips[-1] if self.num_upsample_blocks > 0 else None
            if skip_feat is not None:
                x = self.final_skip_fusion(x, skip_feat)

        # 最终残差细化
        x = self.final_refine(x)

        # 预测 M 和 R
        M = self.weight_head(x)
        R = self.residual_head(x)

        return M, R


class CNNDecoderLUT(BaseDecoder):
    """LUT 融合版 Decoder.

    架构：
        1. Global Branch: RWKV → LUT weights → 全局调色
        2. Spatial Branch: CNN + DINO skip → 空间权重图 M + 残差 R
        3. 融合: Output = (1-M) * LUT_output + M * (Input + R)

    这种设计的优势：
        - 全局色调一致性好（LUT 保证）
        - 局部细节自适应（空间调制）
        - 高度可解释（可视化 LUT 和权重图）

    Args:
        input_dim: RWKV bottleneck 输出维度
        patch_size: Encoder patch size
        num_upsample_blocks: Spatial branch 上采样阶段数
        base_channels: Spatial branch 基础通道数
        skip_channels: DINO skip 通道数列表
        residual_scale: 残差输出缩放
        num_luts: 基 LUT 数量
        lut_size: LUT 网格尺寸
        use_1d_lut: 使用 1D LUT（推荐，更快）
        use_skip_attention: 是否使用 skip 注意力
    """

    def __init__(
        self,
        input_dim: int,
        patch_size: int = 16,
        num_upsample_blocks: int = 4,
        base_channels: int = 64,
        skip_channels: list[int] | None = None,
        residual_scale: float = 0.5,
        predict_residual: bool = True,
        # LUT 相关
        num_luts: int = 5,
        lut_size: int = 33,
        use_1d_lut: bool = True,
        use_skip_attention: bool = True,
    ) -> None:
        super().__init__()
        if not predict_residual:
            raise ValueError("CNNDecoderLUT only supports residual output mode")

        self.patch_size = patch_size
        self.residual_scale = residual_scale
        self.predict_residual = True
        self.num_luts = num_luts
        self.lut_size = lut_size

        # 全局 LUT 分支
        self.global_branch = GlobalLUTBranch(
            input_dim=input_dim,
            num_luts=num_luts,
            lut_size=lut_size,
            use_1d_lut=use_1d_lut,
        )

        # 空间分支
        self.spatial_branch = SpatialBranch(
            input_dim=input_dim,
            patch_size=patch_size,
            num_upsample_blocks=num_upsample_blocks,
            base_channels=base_channels,
            skip_channels=skip_channels,
            use_skip_attention=use_skip_attention,
        )

    def forward(
        self,
        x: Tensor,
        h: int,
        w: int,
        img: Tensor | None = None,
        vit_skips: list[Tensor] | None = None,
        vit_h: int | None = None,
        vit_w: int | None = None,
    ) -> Tensor:
        """解码到增强图像.

        注意：此 decoder 返回完整的增强图像减去输入图像的结果，
        即残差格式，以兼容 pipeline 的处理逻辑。

        最终输出：enhanced = (1-M) * LUT(img) + M * (img + R)
                 residual = enhanced - img

        Args:
            x: (B, N, C) token features from RWKV
            h, w: token grid size
            img: (B, 3, H, W) 原始输入图像（必需）
            vit_skips: DINO skip features
            vit_h, vit_w: DINO feature size

        Returns:
            (B, 3, H, W) 残差 delta（enhanced - img）
        """
        B = x.shape[0]

        if img is None:
            img = x.new_zeros(B, 3, h * self.patch_size, w * self.patch_size)
        _, _, H, W = img.shape

        # 全局 LUT 调色
        lut_output = self.global_branch(x, img)  # (B, 3, H, W)

        # 空间分支：预测权重图和残差
        M, R = self.spatial_branch(
            x, h, w,
            vit_skips=vit_skips,
            vit_h=vit_h,
            vit_w=vit_w,
            target_size=(H, W),
        )  # M: (B, 1, H, W), R: (B, 3, H, W)

        # 融合：enhanced = (1-M) * LUT_output + M * (img + R_scaled)
        R_scaled = torch.tanh(R) * self.residual_scale
        enhanced = (1 - M) * lut_output + M * (img + R_scaled)

        # Clamp 到有效范围
        enhanced = enhanced.clamp(0, 1)

        # 返回残差格式以兼容 pipeline
        return enhanced - img


__all__ = [
    "FastLUTSampling",
    "GlobalLUTBranch",
    "SkipAttentionFusion",
    "ResidualRefineBlock",
    "SpatialBranch",
    "CNNDecoderLUT",
]
