"""One-step NaN debugger for DINOv2-B + vRWKV + CNN.

Runs a single batch through:
  input -> encoder -> bottleneck -> decoder -> loss -> backward -> optimizer step
and prints finite / min / max diagnostics at each stage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from src.data.transforms import to_image_range


def _stats(name: str, x: torch.Tensor) -> None:
    finite = bool(torch.isfinite(x).all().item())
    x_det = x.detach()
    x_min = float(x_det.min().cpu())
    x_max = float(x_det.max().cpu())
    print(f"{name:20s} finite={finite:<5} min={x_min: .6f} max={x_max: .6f}")


def _grad_stats(name: str, module: torch.nn.Module) -> None:
    total = 0
    bad = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        total += 1
        if not torch.isfinite(p.grad).all():
            bad += 1
    print(f"{name:20s} grad_bad={bad}/{total}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug NaN source for vRWKV training")
    p.add_argument(
        "--data-dir",
        default=None,
        help="Dataset root containing raw/ and c/ (or input/ and expertC/).",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--train-subset", type=int, default=8)
    p.add_argument("--use-l1-only", action="store_true")
    p.add_argument("--steps", type=int, default=1, help="How many optimization steps to run.")
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        return torch.device("mps")
    if choice == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device={device}")

    # Config pieces
    enc_cfg = OmegaConf.load("configs/encoder/dinov2_b.yaml")
    bn_cfg = OmegaConf.load("configs/bottleneck/vrwkv.yaml")
    dec_cfg = OmegaConf.load("configs/decoder/cnn.yaml")
    data_cfg = OmegaConf.load("configs/data/fivek.yaml")
    raw_loss_cfg = OmegaConf.load(
        "configs/ablation/a_loss_l1_only.yaml" if args.use_l1_only else "configs/loss/stage1.yaml"
    )
    loss_cfg = raw_loss_cfg.loss if "loss" in raw_loss_cfg else raw_loss_cfg

    data_cfg.batch_size = args.batch_size
    data_cfg.num_workers = args.num_workers
    data_cfg.train_subset_size = args.train_subset
    data_cfg.val_subset_size = 4
    data_cfg.test_subset_size = 4
    data_cfg.expert = "C"
    if args.data_dir:
        data_cfg.data_dir = str(Path(args.data_dir))

    # Data
    dm = hydra.utils.instantiate(data_cfg)
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    # Model pieces
    encoder = hydra.utils.instantiate(enc_cfg, img_size=int(data_cfg.crop_size)).to(device)
    bottleneck = hydra.utils.instantiate(bn_cfg, input_dim=encoder.embed_dim).to(device)
    decoder = hydra.utils.instantiate(
        dec_cfg, input_dim=bottleneck.output_dim, patch_size=encoder.patch_size
    ).to(device)
    criterion = hydra.utils.instantiate(loss_cfg).to(device)

    optim = torch.optim.AdamW(
        list(bottleneck.parameters()) + list(decoder.parameters()),
        lr=1e-5,
        weight_decay=1e-4,
    )

    step = 0
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break

            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x01 = to_image_range(x, normalized_to_neg_one_one=bool(data_cfg.normalize_to_neg_one_one))
            y01 = to_image_range(y, normalized_to_neg_one_one=bool(data_cfg.normalize_to_neg_one_one))

            print(f"\n--- step {step} ---")
            _stats("input_01", x01)
            _stats("target_01", y01)

            with torch.no_grad():
                feat = encoder(x01)
            _stats("encoder_out", feat)

            mid = bottleneck(feat)
            _stats("bottleneck_out", mid)

            h = x01.shape[-2] // encoder.patch_size
            w = x01.shape[-1] // encoder.patch_size
            delta = decoder(mid, h, w, img=x01)
            _stats("decoder_delta", delta)

            pred = (x01 + delta).clamp(0.0, 1.0)
            _stats("pred_01", pred)

            if hasattr(criterion, "component_losses"):
                parts = criterion.component_losses(pred, y01)
                total = pred.new_zeros(())
                for k, v in parts.items():
                    _stats(f"loss_{k}", v.unsqueeze(0))
                    total = total + v
                loss = total
            else:
                loss = criterion(pred, y01)
            _stats("loss_total", loss.unsqueeze(0))

            if not torch.isfinite(loss).all():
                print(f"non-finite loss detected at step {step} (before backward)")
                print("done")
                return

            optim.zero_grad(set_to_none=True)
            loss.backward()
            _grad_stats("bottleneck", bottleneck)
            _grad_stats("decoder", decoder)

            bad_grad = False
            for module in (bottleneck, decoder):
                for p in module.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_grad = True
                        break
                if bad_grad:
                    break
            if bad_grad:
                print(f"non-finite gradient detected at step {step}")
                print("done")
                return

            optim.step()

            with torch.no_grad():
                mid2 = bottleneck(feat)
                delta2 = decoder(mid2, h, w, img=x01)
                pred2 = (x01 + delta2).clamp(0.0, 1.0)
            _stats("post_bottleneck", mid2)
            _stats("post_delta", delta2)
            _stats("post_pred", pred2)

            if not (torch.isfinite(mid2).all() and torch.isfinite(delta2).all() and torch.isfinite(pred2).all()):
                print(f"non-finite activations detected at step {step} (after optimizer step)")
                print("done")
                return

            step += 1

    print("done")


if __name__ == "__main__":
    main()
