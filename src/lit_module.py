"""PyTorch Lightning module wrapping pipeline, losses, optimizer, and metrics."""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import Tensor

from src.data.transforms import to_image_range
from src.evaluation.experiment_record import ExperimentRecord
from src.evaluation.metrics import MetricCollection, compute_psnr
from src.evaluation.report_builder import build_experiment_dir
from src.evaluation.visualization import save_comparison_grid, to_display_u8
from src.models.pipeline import ImageEnhancementPipeline

logger = logging.getLogger(__name__)


class ColorEnhanceLitModule(pl.LightningModule):
    """Lightning module for image color enhancement training."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg

        encoder_kwargs = {}
        # Check for DINOv2 encoder specifically (requires img_size parameter)
        # DINOv3 encoders do NOT use img_size
        encoder_target = str(cfg.encoder.get("_target_", "")).lower()
        if "dinov2" in encoder_target:
            dinov2_img_size = cfg.data.get("crop_size") or cfg.data.get("image_size")
            encoder_kwargs["img_size"] = int(dinov2_img_size)

        encoder = hydra.utils.instantiate(cfg.encoder, **encoder_kwargs)
        main_encoder_proj_dim = cfg.model.get("main_encoder_proj_dim", None)
        bottleneck_input_dim = (
            int(main_encoder_proj_dim)
            if main_encoder_proj_dim is not None
            else encoder.embed_dim
        )
        bottleneck = hydra.utils.instantiate(
            cfg.bottleneck, input_dim=bottleneck_input_dim
        )
        decoder = hydra.utils.instantiate(
            cfg.decoder,
            input_dim=bottleneck.output_dim,
            patch_size=encoder.patch_size,
        )

        # Optional CNN encoder for local texture features
        cnn_encoder = None
        if cfg.get("cnn_encoder") is not None:
            cnn_encoder = hydra.utils.instantiate(cfg.cnn_encoder)


        # Get encoder skip layers from config (optional)
        encoder_skip_layers = cfg.model.get("encoder_skip_layers", None)
        fusion_mode = str(cfg.get("fusion_mode", "concat"))

        self.pipeline = ImageEnhancementPipeline(
            encoder,
            bottleneck,
            decoder,
            cnn_encoder=cnn_encoder,
            encoder_skip_layers=encoder_skip_layers,
            fusion_mode=fusion_mode,
            main_encoder_proj_dim=main_encoder_proj_dim,
        )

        for name, module in self.pipeline.named_children():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name}: total={total:,} trainable={trainable:,}")

        self.criterion = hydra.utils.instantiate(cfg.loss)
        self.metrics = MetricCollection(cfg)
        self.inputs_are_normalized = bool(
            cfg.data.get("normalize_to_neg_one_one", True)
        )

        # Per-test-run sample buffer (populated in test_step, consumed in on_test_end)
        self._test_samples: list[dict] = []
        self._experiment_record = ExperimentRecord(cfg)
        self._train_start_time: float | None = None
        self._reported_lora_backward = False
        self._optimizer_param_ids: set[int] | None = None

        self._log_lora_training_state("init")

    def _lora_named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        encoder = getattr(self.pipeline, "encoder", None)
        if encoder is None:
            return []
        return [
            (name, param)
            for name, param in encoder.named_parameters()
            if "lora" in name.lower()
        ]

    @staticmethod
    def _format_count(value: int) -> str:
        return f"{value:,}"

    def _log_lora_training_state(
        self,
        stage: str,
        *,
        optimizer_param_ids: set[int] | None = None,
        include_grads: bool = False,
    ) -> None:
        lora_params = self._lora_named_parameters()
        total_params = sum(param.numel() for _, param in lora_params)
        trainable_params = sum(
            param.numel() for _, param in lora_params if param.requires_grad
        )

        optimizer_params = None
        if optimizer_param_ids is not None:
            optimizer_params = sum(
                param.numel()
                for _, param in lora_params
                if id(param) in optimizer_param_ids
            )

        grad_tensors = None
        grad_params = None
        grad_norm = None
        if include_grads:
            grad_tensors = sum(1 for _, param in lora_params if param.grad is not None)
            grad_params = sum(
                param.numel() for _, param in lora_params if param.grad is not None
            )
            grad_norm_sq = 0.0
            for _, param in lora_params:
                if param.grad is not None:
                    grad_norm_sq += float(param.grad.detach().float().norm().item() ** 2)
            grad_norm = grad_norm_sq ** 0.5

        status = "NOT_TRAINING"
        if not lora_params:
            reason = "no LoRA parameters found on pipeline.encoder"
        elif trainable_params == 0:
            reason = "LoRA parameters exist but requires_grad=False"
        elif optimizer_params == 0:
            reason = "LoRA parameters require grad but are not in optimizer"
        elif include_grads and grad_tensors == 0:
            reason = "LoRA parameters received no gradients"
        elif include_grads and grad_tensors and grad_tensors > 0:
            status = "TRAINING"
            reason = "LoRA parameters received gradients"
        elif optimizer_params is not None and optimizer_params > 0:
            status = "OPTIMIZER_READY"
            reason = "LoRA parameters are included in optimizer"
        else:
            status = "REQUIRES_GRAD_ONLY"
            reason = "LoRA parameters require grad; optimizer not checked yet"

        fields = [
            f"stage={stage}",
            f"status={status}",
            f"reason={reason}",
            f"tensors={len(lora_params)}",
            f"params={self._format_count(total_params)}",
            f"requires_grad={self._format_count(trainable_params)}",
        ]
        if optimizer_params is not None:
            fields.append(f"in_optimizer={self._format_count(optimizer_params)}")
        if include_grads:
            fields.extend(
                [
                    f"grad_tensors={grad_tensors}",
                    f"grad_params={self._format_count(grad_params or 0)}",
                    f"grad_norm={grad_norm:.6e}" if grad_norm is not None else "grad_norm=n/a",
                ]
            )
        logger.info("LoRA training check: %s", " | ".join(fields))

    def _to_01(self, x: Tensor) -> Tensor:
        """Convert loader tensors to [0, 1] using explicit config."""
        return to_image_range(
            x,
            normalized_to_neg_one_one=self.inputs_are_normalized,
        )

    # Loss component name → corresponding metric gate kwarg.
    # "delta_e" is never a loss component, so its null fallback is always False;
    # it must be explicitly set to true in evaluation.test to be reported.
    _LOSS_TO_METRIC: dict[str, str] = {
        "lpips":      "enable_lpips",
        "nima":       "enable_nima",
        "clip":       "enable_clip",
        "perceptual": "enable_vgg_perceptual",
        "delta_e":    "enable_delta_e",
    }

    def _resolve_metric_gates(self, phase_cfg: object) -> dict[str, bool]:
        """Derive metric gates for a single eval phase.

        For each gate the resolution order is:
          1. Explicit value in phase_cfg (true / false)  → use it directly
          2. null in phase_cfg                           → follow active loss components
          3. Key absent from phase_cfg                   → same as null (follow loss)

        psnr and ssim are always enabled and not included in the returned dict
        (they are passed as positional defaults in _compute_metrics).

        Args:
            phase_cfg: OmegaConf node (or dict) for evaluation.val / evaluation.test.

        Returns:
            Dict of ``enable_*`` kwargs ready to be passed to ``_compute_metrics``.
        """
        active_loss_names: set[str] = set()
        if hasattr(self.criterion, "loss_fns"):
            active_loss_names = set(self.criterion.loss_fns.keys())

        gates: dict[str, bool] = {}
        for loss_name, gate_key in self._LOSS_TO_METRIC.items():
            raw = None
            if phase_cfg is not None:
                try:
                    raw = phase_cfg.get(gate_key)
                except Exception:
                    raw = None
            if raw is None:
                gates[gate_key] = loss_name in active_loss_names
            else:
                gates[gate_key] = bool(raw)
        return gates

    @staticmethod
    def _mean_tensor_dict(values: dict[str, list[Tensor]]) -> dict[str, Tensor]:
        return {
            name: torch.stack(items).mean()
            for name, items in values.items()
            if items
        }

    def _unpack_batch(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, list[str]],
    ) -> tuple[Tensor, Tensor, Tensor | None, list[str] | None]:
        if len(batch) == 2:
            input_img, target_img = batch
            return input_img, target_img, None, None
        if len(batch) == 3:
            input_img, target_img, sizes = batch
            return input_img, target_img, sizes, None
        if len(batch) == 4:
            input_img, target_img, sizes, names = batch
            return input_img, target_img, sizes, list(names)
        raise ValueError(f"Unexpected batch structure with length {len(batch)}")

    def _iter_valid_samples(
        self,
        pred: Tensor,
        target: Tensor,
        sizes: Tensor | None,
    ) -> list[tuple[Tensor, Tensor]]:
        if sizes is None:
            return [(pred, target)]
        samples: list[tuple[Tensor, Tensor]] = []
        for idx, size in enumerate(sizes):
            h = int(size[0].item())
            w = int(size[1].item())
            samples.append((
                pred[idx:idx + 1, :, :h, :w],
                target[idx:idx + 1, :, :h, :w],
            ))
        return samples

    def _crop_uniform_batch(
        self,
        pred: Tensor,
        target: Tensor,
        sizes: Tensor | None,
    ) -> tuple[Tensor, Tensor] | None:
        if sizes is None or sizes.numel() == 0:
            return pred, target
        first_h = int(sizes[0, 0].item())
        first_w = int(sizes[0, 1].item())
        if not bool(torch.all(sizes[:, 0] == first_h) and torch.all(sizes[:, 1] == first_w)):
            return None
        return (
            pred[:, :, :first_h, :first_w],
            target[:, :, :first_h, :first_w],
        )

    def _compute_loss(
        self,
        pred: Tensor,
        target: Tensor,
        sizes: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        raw_losses: dict[str, Tensor] = {}
        weighted_losses: dict[str, Tensor] = {}
        weights: dict[str, Tensor] = {}

        uniform_batch = self._crop_uniform_batch(pred, target, sizes)
        if uniform_batch is not None:
            pred, target = uniform_batch
            sizes = None

        if sizes is None:
            if hasattr(self.criterion, "compute"):
                return self.criterion.compute(
                    pred,
                    target,
                    update_stats=self.training,
                )
            if hasattr(self.criterion, "component_losses"):
                weighted_losses = self.criterion.component_losses(pred, target)
                loss = pred.new_zeros(())
                for value in weighted_losses.values():
                    loss = loss + value
                return loss, raw_losses, weighted_losses, weights
            return self.criterion(pred, target), raw_losses, weighted_losses, weights

        sample_pairs = self._iter_valid_samples(pred, target, sizes)
        if hasattr(self.criterion, "compute"):
            loss_items: list[Tensor] = []
            raw_acc: dict[str, list[Tensor]] = {}
            weighted_acc: dict[str, list[Tensor]] = {}
            weight_acc: dict[str, list[Tensor]] = {}
            for sample_pred, sample_target in sample_pairs:
                sample_loss, sample_raw, sample_weighted, sample_weights = self.criterion.compute(
                    sample_pred,
                    sample_target,
                    update_stats=self.training,
                )
                loss_items.append(sample_loss)
                for name, value in sample_raw.items():
                    raw_acc.setdefault(name, []).append(value)
                for name, value in sample_weighted.items():
                    weighted_acc.setdefault(name, []).append(value)
                for name, value in sample_weights.items():
                    weight_acc.setdefault(name, []).append(value)
            loss = torch.stack(loss_items).mean()
            return (
                loss,
                self._mean_tensor_dict(raw_acc),
                self._mean_tensor_dict(weighted_acc),
                self._mean_tensor_dict(weight_acc),
            )

        if hasattr(self.criterion, "component_losses"):
            loss_items: list[Tensor] = []
            weighted_acc: dict[str, list[Tensor]] = {}
            for sample_pred, sample_target in sample_pairs:
                sample_weighted = self.criterion.component_losses(
                    sample_pred,
                    sample_target,
                )
                sample_total = sample_pred.new_zeros(())
                for name, value in sample_weighted.items():
                    sample_total = sample_total + value
                    weighted_acc.setdefault(name, []).append(value)
                loss_items.append(sample_total)
            loss = torch.stack(loss_items).mean()
            return loss, raw_losses, self._mean_tensor_dict(weighted_acc), weights

        sample_losses = [
            self.criterion(sample_pred, sample_target)
            for sample_pred, sample_target in sample_pairs
        ]
        return torch.stack(sample_losses).mean(), raw_losses, weighted_losses, weights

    def _compute_metrics(
        self,
        pred: Tensor,
        target: Tensor,
        sizes: Tensor | None,
        *,
        enable_psnr: bool,
        enable_ssim: bool,
        enable_vgg_perceptual: bool,
        enable_lpips: bool,
        enable_delta_e: bool,
        enable_nima: bool,
        enable_clip: bool,
    ) -> dict[str, Tensor]:
        uniform_batch = self._crop_uniform_batch(pred, target, sizes)
        if uniform_batch is not None:
            pred, target = uniform_batch
            sizes = None
        if sizes is None:
            return self.metrics(
                pred,
                target,
                enable_psnr=enable_psnr,
                enable_ssim=enable_ssim,
                enable_vgg_perceptual=enable_vgg_perceptual,
                enable_lpips=enable_lpips,
                enable_delta_e=enable_delta_e,
                enable_nima=enable_nima,
                enable_clip=enable_clip,
            )

        metric_acc: dict[str, list[Tensor]] = {}
        for sample_pred, sample_target in self._iter_valid_samples(pred, target, sizes):
            sample_metrics = self.metrics(
                sample_pred,
                sample_target,
                enable_psnr=enable_psnr,
                enable_ssim=enable_ssim,
                enable_vgg_perceptual=enable_vgg_perceptual,
                enable_lpips=enable_lpips,
                enable_delta_e=enable_delta_e,
                enable_nima=enable_nima,
                enable_clip=enable_clip,
            )
            for name, value in sample_metrics.items():
                metric_acc.setdefault(name, []).append(value)
        return self._mean_tensor_dict(metric_acc)

    def _shared_step(
        self, batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, list[str]]
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
        list[str] | None,
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
    ]:
        """Run forward pass and loss computation for train/val/test steps."""
        input_img, target_img, sizes, names = self._unpack_batch(batch)
        input_01 = self._to_01(input_img)
        target_01 = self._to_01(target_img)
        pred = self.pipeline(input_01)
        loss, raw_losses, weighted_losses, weights = self._compute_loss(
            pred,
            target_01,
            sizes,
        )
        return loss, pred, target_01, sizes, names, raw_losses, weighted_losses, weights

    def _log_loss_breakdown(
        self,
        prefix: str,
        raw_losses: dict[str, Tensor],
        weighted_losses: dict[str, Tensor],
        weights: dict[str, Tensor],
        *,
        on_step: bool = False,
        on_epoch: bool = True,
    ) -> None:
        """Log weighted losses, raw losses, and adaptive weights."""
        if weighted_losses:
            self.log_dict(
                {
                    f"{prefix}/loss_{name}": value
                    for name, value in weighted_losses.items()
                },
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                sync_dist=True,
            )

        if prefix in {"val", "test"} and weights:
            self.log_dict(
                {
                    f"{prefix}/loss_weight_{name}": value
                    for name, value in weights.items()
                },
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                sync_dist=True,
            )

    def on_train_start(self) -> None:
        import time
        self._train_start_time = time.time()

    def on_after_backward(self) -> None:
        if not self._reported_lora_backward:
            self._log_lora_training_state(
                "after_backward",
                optimizer_param_ids=self._optimizer_param_ids,
                include_grads=True,
            )
            self._reported_lora_backward = True

    def training_step(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, list[str]],
        batch_idx: int,
    ) -> Tensor:
        loss, pred, target, _, _, raw_losses, weighted_losses, weights = self._shared_step(batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self._log_loss_breakdown(
            "train",
            raw_losses,
            weighted_losses,
            weights,
            on_epoch=True,
        )
        # Log train PSNR per epoch — required to distinguish underfitting from overfitting.
        # Computed on the padded batch tensor (padding is a consistent bias, acceptable
        # for monitoring). on_step=False avoids per-step noise in the CSV.
        with torch.no_grad():
            train_psnr = compute_psnr(pred, target)
        self.log("train/psnr", train_psnr, on_step=False, on_epoch=True,
                 prog_bar=False, sync_dist=True)
        return loss

    def _eval_step(
        self,
        phase: str,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, list[str]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, list[str] | None]:
        """Shared forward + loss + metric computation for val and test.

        Returns (loss, pred, target, sizes, names) so callers can do phase-specific
        work (e.g. saving visualizations in validation_step).
        """
        loss, pred, target, sizes, names, raw_losses, weighted_losses, weights = self._shared_step(batch)

        phase_cfg = self.cfg.evaluation.get(phase)
        gates = self._resolve_metric_gates(phase_cfg)
        metrics = self._compute_metrics(
            pred,
            target,
            sizes,
            enable_psnr=True,
            enable_ssim=True,
            **gates,
        )

        prog_bar = phase == "val"
        self.log(f"{phase}/loss", loss, on_epoch=True, sync_dist=True)
        self._log_loss_breakdown(phase, raw_losses, weighted_losses, weights, on_epoch=True)
        self.log_dict(
            {f"{phase}/{k}": v for k, v in metrics.items()},
            on_epoch=True,
            prog_bar=prog_bar,
            sync_dist=True,
        )
        return loss, pred, target, sizes, names

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, list[str]],
        batch_idx: int,
    ) -> None:
        _, pred, target, _, _ = self._eval_step("val", batch)

        save_every = self.cfg.evaluation.save_samples_every_n_epochs
        if batch_idx == 0 and (self.current_epoch % save_every == 0):
            input_img, _, _, _ = self._unpack_batch(batch)
            vis_inputs = self._to_01(input_img)
            n = min(
                self.cfg.evaluation.num_visualization_samples,
                pred.shape[0],
            )
            fig_dir = Path("outputs/figures") / self.cfg.experiment_name
            save_comparison_grid(
                vis_inputs[:n].cpu(),
                pred[:n].cpu(),
                target[:n].cpu(),
                save_path=fig_dir / f"epoch_{self.current_epoch:04d}.png",
            )

    @staticmethod
    def _enabled_metric_names(gates: dict[str, bool]) -> list[str]:
        """Return the metric columns that should be materialised for test samples."""
        names = ["psnr", "ssim"]
        optional = [
            ("enable_vgg_perceptual", "vgg_perceptual"),
            ("enable_lpips", "lpips"),
            ("enable_delta_e", "delta_e"),
            ("enable_nima", "nima"),
            ("enable_clip", "clip"),
        ]
        for gate_key, metric_name in optional:
            if gates.get(gate_key, False):
                names.append(metric_name)
        return names

    def _collect_test_samples(
        self,
        input_01: Tensor,
        pred: Tensor,
        target: Tensor,
        sizes: Tensor | None,
        names: list[str] | None,
        metric_gates: dict[str, bool],
    ) -> None:
        """Store per-sample metrics and display tensors for the post-test report."""
        B = pred.shape[0]
        metric_names = self._enabled_metric_names(metric_gates)
        with torch.no_grad():
            for i in range(B):
                if sizes is not None:
                    h = int(sizes[i, 0].item())
                    w = int(sizes[i, 1].item())
                else:
                    h, w = pred.shape[-2], pred.shape[-1]

                p = pred[i:i + 1, :, :h, :w]
                t = target[i:i + 1, :, :h, :w]
                inp = input_01[i, :, :h, :w]

                sample_metrics = self.metrics(
                    p,
                    t,
                    enable_psnr=True,
                    enable_ssim=True,
                    **metric_gates,
                )

                sample_index = len(self._test_samples)
                row = {
                    "sample_index": sample_index,
                    "file_name": names[i] if names is not None else str(sample_index),
                    "height": h,
                    "width": w,
                    "input_u8":  to_display_u8(inp),
                    "pred_u8":   to_display_u8(p[0]),
                    "target_u8": to_display_u8(t[0]),
                }
                for metric_name in metric_names:
                    row[metric_name] = float(sample_metrics[metric_name].detach().cpu())

                self._test_samples.append(row)

    def test_step(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, list[str]],
        batch_idx: int,
    ) -> None:
        _, pred, target, sizes, names = self._eval_step("test", batch)
        input_img, _, _, _ = self._unpack_batch(batch)
        test_gates = self._resolve_metric_gates(self.cfg.evaluation.get("test"))
        self._collect_test_samples(
            self._to_01(input_img),
            pred.detach(),
            target.detach(),
            sizes,
            names,
            test_gates,
        )

    def on_test_end(self) -> None:
        """Export all experiment artefacts to a unified self-contained directory."""
        logged = self.trainer.logged_metrics

        # 1. Finalise experiment record → writes experiment_record.json under outputs/logs/{name}/
        self._experiment_record.finalize(
            logged_metrics=logged,
            trainer=self.trainer,
            datamodule=self.trainer.datamodule,
            pipeline=self.pipeline,
            train_start_time=self._train_start_time,
        )

        # 2. Build the unified experiment directory under experiments/
        log_dir = Path(self.cfg.logging.log_dir) / self.cfg.experiment_name
        exp_dir = build_experiment_dir(
            record_data=self._experiment_record.data,
            test_samples=self._test_samples,
            log_dir=log_dir,
        )
        print(f"\nExperiment report → {exp_dir}")

    def configure_optimizers(self):
        """Build AdamW optimizer and LR scheduler."""
        trainable = list(self.pipeline.trainable_parameters())
        self._optimizer_param_ids = {id(param) for param in trainable}
        self._log_lora_training_state(
            "optimizer",
            optimizer_param_ids=self._optimizer_param_ids,
        )
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

        scheduler_name = self.cfg.training.get("scheduler", "cosine")
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.training.max_epochs,
                eta_min=1e-6,
            )
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=50, gamma=0.5
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=10
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/psnr",
                "interval": "epoch",
                "frequency": 1,
            },
        }
