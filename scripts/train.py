"""Hydra main training entry point.

Usage examples::

    # Official Stage 5 baseline: DINOv3 encoder + CNN bottleneck + CNN decoder
    python scripts/train.py experiment_name=dinov3_cnn_cnn

    # Ablation: remove perceptual / SSIM terms
    python scripts/train.py ablation=a_loss_l1_only experiment_name=a_loss_l1

    # Debug run (2 epochs, tiny data, no early stop)
    python scripts/train.py training.max_epochs=2 training.batch_size_per_device=2 \\
        data.train_subset_size=10 data.val_subset_size=5 \\
        data.test_subset_size=5 experiment_name=debug

    # Gradient accumulation: use smaller batch to save memory, accumulate steps
    # Effective batch size = batch_size_per_device * accumulate_grad_batches = 8 * 2 = 16
    python scripts/train.py training.batch_size_per_device=8 training.accumulate_grad_batches=2

    # Resume full training state from a saved checkpoint
    python scripts/train.py training.max_epochs=30 \\
        training.resume_from_checkpoint=outputs/checkpoints/model.ckpt

    # Load model weights from a checkpoint and continue with fresh optimiser
    python scripts/train.py training.max_epochs=10 \\
        training.init_from_checkpoint=outputs/checkpoints/model.ckpt

    # Final 4500/500 comparison run: merge train+val, disable validation
    python scripts/train.py family=color_naf_rwkv7 \\
        ablation/color_naf_rwkv7/film=on \\
        +run_mode=final_4500_no_val
"""
from __future__ import annotations

import logging
import warnings

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from src.lit_module import ColorEnhanceLitModule
from src.utils.checkpoint import allow_trusted_checkpoint_loading, load_checkpoint
from src.utils.seed import set_seed
from src.utils.config_snapshot import save_config_snapshot
from src.utils.device import get_accelerator
from src.utils.model_profiler import EncoderProfilerCallback
import torch
torch.use_deterministic_algorithms(False)


def _configure_console_logging(cfg: DictConfig) -> None:
    level_name = str(cfg.logging.get("console_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    allow_trusted_checkpoint_loading()
    _configure_console_logging(cfg)

    exp_suffix = str(cfg.get("experiment_name_suffix", "") or "")
    if exp_suffix and not str(cfg.experiment_name).endswith(exp_suffix):
        cfg.experiment_name = f"{cfg.experiment_name}{exp_suffix}"

    # 1. Reproducibility
    set_seed(cfg.seed)

    # 2. Persist full config to outputs/logs/{experiment_name}/config/
    save_config_snapshot(cfg)

    # 3. Model
    model = ColorEnhanceLitModule(cfg)
    init_ckpt = cfg.training.get("init_from_checkpoint")
    if init_ckpt:
        load_checkpoint(init_ckpt, model.pipeline, strict=False)

    # 4. Data
    datamodule = hydra.utils.instantiate(cfg.data)
    final_fit_no_val = bool(cfg.training.get("final_fit_no_val", False))

    # 5. Callbacks
    if final_fit_no_val:
        callbacks = [
            ModelCheckpoint(
                dirpath=cfg.checkpoint.dirpath,
                monitor=None,
                save_top_k=0,
                save_last=True,
                filename=f"{cfg.experiment_name}_{{epoch:03d}}",
                verbose=True,
                auto_insert_metric_name=False,
                every_n_epochs=1,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]
    else:
        callbacks = [
            ModelCheckpoint(
                dirpath=cfg.checkpoint.dirpath,
                monitor=cfg.checkpoint.monitor,
                mode=cfg.checkpoint.mode,
                save_top_k=cfg.checkpoint.save_top_k,
                save_last=False,  # 只保存最佳checkpoint，不保存最后一个epoch
                filename=f"{cfg.experiment_name}_{{epoch:03d}}-{{val/psnr:.2f}}",
                verbose=True,
                auto_insert_metric_name=False,  # PL 2.6+: 防止自动插入metric名称
                every_n_epochs=1,  # 每个epoch都检查是否为最佳模型
            ),
            EarlyStopping(
                monitor=cfg.training.early_stopping.monitor,
                patience=cfg.training.early_stopping.patience,
                mode=cfg.training.early_stopping.mode,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

    # 5a. Optional profiler (enable with profile=true on CLI)
    if cfg.get("profile", False):
        warnings.warn(
            "profile=true enables torch.profiler with shape/flops recording. "
            "The first few training batches will be much slower, so early ETA "
            "estimates are not representative of steady-state training speed.",
            stacklevel=2,
        )
        callbacks.append(
            EncoderProfilerCallback(
                wait=cfg.profiler.wait,
                warmup=cfg.profiler.warmup,
                active=cfg.profiler.active,
                log_dir=cfg.profiler.log_dir,
            )
        )

    # 6. Logger (pure local CSV — no cloud dependency)
    csv_logger = CSVLogger(
        save_dir=cfg.logging.log_dir,
        name=cfg.experiment_name,
    )

    # 7. Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=get_accelerator(),
        devices=1,
        callbacks=callbacks,
        logger=csv_logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        deterministic=False,
        log_every_n_steps=10,
        enable_progress_bar=True,
        limit_val_batches=0 if final_fit_no_val else None,
        num_sanity_val_steps=0 if final_fit_no_val else 2,
    )

    # 8. Fit
    resume_ckpt = cfg.training.get("resume_from_checkpoint")
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)

    # 9. Test with the selected checkpoint (also triggers on_test_end → JSON export)
    trainer.test(
        model,
        datamodule=datamodule,
        ckpt_path="last" if final_fit_no_val else "best",
    )


if __name__ == "__main__":
    main()
