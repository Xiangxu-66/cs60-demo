# MS-SWC Loss：本次改动清单

本次任务：为 FiveK 色彩美学增强任务新增一个 loss 函数作为对比实验。

## 设计思路（简述）

**Multiscale Sliced-Wasserstein Contrastive Loss (MS-SWC)** — 当前实现为 **positive-only** 变体（不含对比项），等价于 ECCV 2024 MS-SWD 当作训练 loss 使用。

核心思想：把预测图和目标图都转到 CIELAB 色彩空间 → 构建 Gaussian 金字塔 → 在每个尺度上把像素看作 3D 点云 → 用 sliced 1-Wasserstein 距离（随机单位球方向投影 + 排序后 L1）对齐两者的色彩分布。

**与现有 VGG perceptual loss 的区别**：VGG 在 ImageNet 特征空间对齐，MS-SWC 在感知均匀的 LAB 色彩分布空间对齐。后者更贴合色彩 retouching 任务本质。

参考：He et al., "Multiscale Sliced Wasserstein Distances as Perceptual Color Difference Measures", ECCV 2024. ([arXiv:2407.10181](https://arxiv.org/abs/2407.10181))

---

## 新增文件

### 1. `src/losses/ms_swc_loss.py`
MS-SWC loss 主实现。包含：
- `rgb_to_lab(rgb)` — 可微的 sRGB→CIELAB 转换（D65 白点，标准 IEC 61966-2-1 矩阵）
- `_GaussianBlur` / `_gaussian_pyramid` — 多尺度金字塔构建
- `_sliced_wasserstein` — 单次 SWD 计算（随机投影 + 排序 L1）
- `MSSWCLoss` — 对外接口，签名与项目其他 loss 一致 `forward(pred, target) -> scalar`

**默认超参**：
| 参数 | 默认值 | 说明 |
|---|---|---|
| `weight` | 1.0 | 外部缩放 |
| `num_scales` | 5 | 金字塔尺度数 |
| `num_projections` | 128 | SWD 随机方向数（每次 forward 重采样）|
| `pyr_sigma` | 1.0 | 下采样前高斯模糊 sigma |
| `pyr_ksize` | 5 | 高斯 kernel 大小 |

### 2. `configs/loss/msswc_only.yaml`
实验 1 用的 Hydra config — 只用 MS-SWC（positive-only），weight=1.0，无 L1 无 SSIM。
纯诊断用途，用于：
- 确认 loss 能正常反传收敛
- 校准 loss 数值量级
- 验证 SWD "只管色彩分布不管位置" 的行为

### 3. `tests/test_ms_swc_loss.py`
6 个单元测试：
1. `test_rgb_to_lab_reference_values` — 纯黑/纯白的 LAB 参考值正确
2. `test_identical_inputs_give_zero_loss` — `loss(x, x) == 0`
3. `test_forward_returns_finite_scalar` — 输出是有限标量
4. `test_gradient_flows_to_pred` — 反向梯度存在且有限非零
5. `test_shifted_target_has_larger_loss_than_perturbation` — 色调偏离远 > 小扰动
6. `test_weight_scales_output` — `weight` 参数严格线性缩放

---

## 修改文件

### `src/losses/__init__.py`
添加 `MSSWCLoss` 的 re-export 到 `__all__`。**此改动非必须** — Hydra 的 `_target_` 直接 import 子模块，不经过 `__init__.py`。仅为了与现有 L1/SSIM/Perceptual 的导出约定保持一致，方便 notebook / 消融脚本里短写 `from src.losses import MSSWCLoss`。

---

## 服务器端使用

### 跑单元测试
```bash
python -m pytest tests/test_ms_swc_loss.py -v
```
预期 6 全绿。

### 实验 1：MS-SWC only（诊断）
```bash
python scripts/train.py \
  loss=msswc_only \
  bottleneck=v6_v2 \
  trainer.max_epochs=20
```
先 20 epoch 看趋势。

### 预期观察
| 指标 | 预期 | 解读 |
|---|---|---|
| train loss | 单调下降，量级 10–30 | SWD 在 LAB 单位上天然较大 |
| val/psnr | 📉 下降 1–3 dB | **预期** — 无 L1 锚点 |
| val/ssim | 📉 中等下降 | 无结构约束 |
| val/lpips | ➡️ 持平或微升 | SWD 对感知色彩对齐有效 |
| 可视化 | 色调贴 target，细节偏糊 | SWD 位置不敏感特性 |

---

## 后续实验计划

跑完实验 1 拿到 MS-SWC 的数值量级后再定权重：

| # | Config（待创建） | 组合 | 目的 |
|---|---|---|---|
| 1 | `msswc_only.yaml` ✅ | MS-SWC | 诊断、校准 |
| 2 | `l1_msswc.yaml` | L1 + MS-SWC | 看像素锚点救回多少 PSNR |
| 3 | `stage1_msswc.yaml` | L1 + SSIM + MS-SWC | 完整新组合，对比当前 baseline |
| 4 | `stage1.yaml` ✅（已存在）| L1 + SSIM + VGG | 当前 baseline |
| 5 *(可选)* | `msswc_contrastive.yaml` | 加三元对比项 | 带负样本的完整 MS-SWC |

权重待实验 1 结果出来后再调。

---

## 更新记录

- **2026-04-26**：V1 bottleneck 三组消融实验完成（FiveK-5k，32 epoch）。`stage1_msswc` 在全部 5 个测试指标上领先 `stage1` baseline（PSNR +0.05 dB，LPIPS −2.5%，NIMA +0.066）；`stage1_plus_msswc` 未超过 `stage1_msswc`，提示 Perceptual 与 ms_swc 存在结构性冲突。
