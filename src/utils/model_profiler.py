# src/utils/model_profiler.py
from __future__ import annotations

import json
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.profiler import ProfilerActivity, profile, record_function, schedule


def _save_results(prof, log_dir: str, experiment_name: str) -> None:
    """Print a console summary and save results as JSON."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    avgs = prof.key_averages()

    # ── console summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PROFILER SUMMARY: {experiment_name}")
    print(f"{'='*60}")
    print(avgs.table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=20,
    ))

    # ── totals ────────────────────────────────────────────────────────
    total_flops   = sum(e.flops for e in avgs if e.flops)
    total_cuda_ms = sum(e.cuda_time for e in avgs) / 1e3
    total_cpu_ms  = sum(e.cpu_time for e in avgs) / 1e3
    peak_mem_mb   = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    print(f"\n  GFLOPs (profiler-visible): {total_flops/1e9:.2f}")
    print(f"  CUDA time per step:        {total_cuda_ms:.2f} ms")
    print(f"  CPU  time per step:        {total_cpu_ms:.2f} ms")
    print(f"  Peak VRAM:                 {peak_mem_mb:.1f} MB")

    # ── detailed JSON (one entry per op) ──────────────────────────────
    detailed_data = []
    for e in avgs:
        detailed_data.append({
            "op": e.key,
            "calls": e.count,
            "cpu_time_ms": e.cpu_time / 1e3,
            "cuda_time_ms": e.cuda_time / 1e3,
            "flops": e.flops,
        })

    json_path = log_path / f"{experiment_name}_profiler.json"
    with open(json_path, "w") as f:
        json.dump(detailed_data, f, indent=4)

    # ── summary JSON (append per experiment) ──────────────────────────
    summary_entry = {
        "experiment": experiment_name,
        "gflops": round(total_flops / 1e9, 2),
        "cuda_ms": round(total_cuda_ms, 2),
        "cpu_ms": round(total_cpu_ms, 2),
        "peak_vram_mb": round(peak_mem_mb, 1),
    }

    summary_path = log_path / "summary.json"

    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary_data = json.load(f)
    else:
        summary_data = []

    summary_data.append(summary_entry)

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"\n  Full JSON:    {json_path}")
    print(f"  Summary JSON: {summary_path}\n")


class EncoderProfilerCallback(pl.Callback):
    def __init__(
        self,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        log_dir: str = "./profiler_logs",
    ) -> None:
        self._wait    = wait
        self._warmup  = warmup
        self._active  = active
        self._log_dir = log_dir
        self._profiler = None
        self._step     = 0
        self._done     = False

    def on_train_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        self._experiment_name = trainer.logger.name if trainer.logger else "experiment"

        self._profiler = profile(
            activities=activities,
            schedule=schedule(
                wait=self._wait,
                warmup=self._warmup,
                active=self._active,
                repeat=1,
            ),
            with_flops=True,
            record_shapes=True,
        )
        self._profiler.start()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._done:
            return
        self._profiler.step()
        self._step += 1
        if self._step >= self._wait + self._warmup + self._active:
            self._profiler.stop()
            _save_results(self._profiler, self._log_dir, self._experiment_name)
            self._done = True

    def on_train_end(self, trainer, pl_module):
        if not self._done and self._profiler:
            self._profiler.stop()
            _save_results(self._profiler, self._log_dir, self._experiment_name)