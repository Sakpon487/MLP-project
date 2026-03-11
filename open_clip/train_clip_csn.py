#!/usr/bin/env python3
"""
Train CLIP-frozen visual-only CSN pipeline.

- Superclass similarity on full projection embeddings.
- Category similarity on shared-mask CSN embeddings.
- Visual branch only: CLIP image encoder -> projection head -> shared CSN mask.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# local CLIP checkout
CLIP_DIR = Path(__file__).parent.parent / "CLIP"
if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

import clip  # type: ignore

from csn_pipeline.data import CSNMultiViewDataset, collate_csn_batch, create_or_load_split, load_csn_records
from csn_pipeline.losses import SupConLoss, two_view_supcon_loss
from csn_pipeline.model import ProjectionHead, SharedCSNMask


@dataclass
class LossBundle:
    total: float
    super_simclr: float
    cat_simclr: float


@dataclass
class TrainState:
    epoch: int
    best_metric: float
    best_epoch: int
    train_history: list[dict[str, Any]]
    test_history: list[dict[str, Any]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str, model_name: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            d = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            d = "mps"
        else:
            d = "cpu"
    else:
        d = device_arg

    if d == "mps" and (not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available()):
        print("Warning: MPS requested but unavailable. Falling back to CPU.")
        d = "cpu"

    if d == "mps" and model_name.strip().upper().startswith("RN"):
        print("Warning: ResNet models have MPS issues. Falling back to CPU.")
        d = "cpu"

    if d == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    return torch.device(d)


def freeze_clip_model(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def compute_losses(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    image_head: torch.nn.Module,
    csn_mask: torch.nn.Module,
    supcon_loss: SupConLoss,
    weights: dict[str, float],
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = next(image_head.parameters()).device

    images = torch.stack(
        [batch["image_view1"], batch["image_view2"], batch["image_view3"]],
        dim=1,
    ).to(device=device, dtype=torch.float32, non_blocking=True)

    labels_super = batch["label_view1_2"].to(device=device, non_blocking=True)
    labels_cat = batch["label_view1_3"].to(device=device, non_blocking=True)

    bsz, n_views, c, h, w = images.shape
    images_flat = images.view(bsz * n_views, c, h, w)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.no_grad():
            img_feat_flat = model.encode_image(images_flat).float()

        img_proj = image_head(img_feat_flat).view(bsz, n_views, -1)
        img_masked = csn_mask(img_proj.view(bsz * n_views, -1)).view(bsz, n_views, -1)

        l_super_simclr = two_view_supcon_loss(supcon_loss, img_proj[:, 0], img_proj[:, 1], labels_super)
        l_cat_simclr = two_view_supcon_loss(supcon_loss, img_masked[:, 0], img_masked[:, 2], labels_cat)

        total = (
            weights["w_super_simclr"] * l_super_simclr
            + weights["w_cat_simclr"] * l_cat_simclr
        )

    parts = {
        "super_simclr": l_super_simclr,
        "cat_simclr": l_cat_simclr,
    }
    return total, parts


def run_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    image_head: torch.nn.Module,
    csn_mask: torch.nn.Module,
    supcon_loss: SupConLoss,
    weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    train: bool,
    epoch: int,
    total_epochs: int,
) -> LossBundle:
    if train:
        image_head.train()
        csn_mask.train()
        mode = f"Train {epoch}/{total_epochs}"
    else:
        image_head.eval()
        csn_mask.eval()
        mode = f"Test {epoch}/{total_epochs}"

    running = {
        "total": 0.0,
        "super_simclr": 0.0,
        "cat_simclr": 0.0,
    }
    n_batches = 0

    pbar = tqdm(loader, desc=mode, total=len(loader))
    for batch in pbar:
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            total, parts = compute_losses(
                model=model,
                batch=batch,
                image_head=image_head,
                csn_mask=csn_mask,
                supcon_loss=supcon_loss,
                weights=weights,
                use_amp=use_amp,
            )

            if train:
                if use_amp:
                    scaler.scale(total).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total.backward()
                    optimizer.step()

        running["total"] += float(total.item())
        for k in ("super_simclr", "cat_simclr"):
            running[k] += float(parts[k].item())
        n_batches += 1

        pbar.set_postfix(total=f"{running['total']/n_batches:.4f}")

    if n_batches == 0:
        raise RuntimeError("No batches processed in epoch.")

    return LossBundle(
        total=running["total"] / n_batches,
        super_simclr=running["super_simclr"] / n_batches,
        cat_simclr=running["cat_simclr"] / n_batches,
    )


def save_checkpoint(
    out_dir: Path,
    epoch: int,
    args: argparse.Namespace,
    image_head: ProjectionHead,
    csn_mask: SharedCSNMask,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    train_state: TrainState,
    is_best: bool,
) -> None:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": int(epoch),
        "model_name": args.model,
        "visual_only": True,
        "image_head_state_dict": image_head.state_dict(),
        "csn_mask_state_dict": csn_mask.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "train_state": {
            "epoch": train_state.epoch,
            "best_metric": train_state.best_metric,
            "best_epoch": train_state.best_epoch,
            "train_history": train_state.train_history,
            "test_history": train_state.test_history,
        },
        "split_dir": str(args.split_dir) if args.split_dir else None,
        "seed": int(args.seed),
        "args": vars(args),
    }

    epoch_path = ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    latest_path = ckpt_dir / "latest_checkpoint.pt"
    torch.save(ckpt, epoch_path)
    torch.save(ckpt, latest_path)
    if is_best:
        best_path = ckpt_dir / "best_checkpoint.pt"
        torch.save(ckpt, best_path)


def save_loss_log(out_dir: Path, train_state: TrainState) -> None:
    log_path = out_dir / "loss_log.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "best_metric": train_state.best_metric,
                "best_epoch": train_state.best_epoch,
                "train_history": train_state.train_history,
                "test_history": train_state.test_history,
            },
            f,
            indent=2,
        )


def plot_loss_curves(out_dir: Path, train_history: list[dict[str, Any]], test_history: list[dict[str, Any]]) -> None:
    if not train_history:
        return

    plt.figure(figsize=(10, 6))
    train_epochs = [x["epoch"] for x in train_history]
    train_total = [x["total"] for x in train_history]
    plt.plot(train_epochs, train_total, label="train_total", linewidth=2)

    if test_history:
        test_epochs = [x["epoch"] for x in test_history]
        test_total = [x["total"] for x in test_history]
        plt.plot(test_epochs, test_total, label="test_total", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CSN training/test loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "loss_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen-CLIP visual-only CSN pipeline")

    parser.add_argument("--csv-file", type=str, required=True)
    parser.add_argument("--base-image-dir", type=str, default=None)
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--force-resplit", action="store_true")

    parser.add_argument("--model", type=str, default="ViT-B/32")
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--mask-init", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--linear-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for projection linear layers (image_head). Mask uses --weight-decay.",
    )
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--w-super-simclr", type=float, default=1.0)
    parser.add_argument("--w-super-it", type=float, default=1.0)
    parser.add_argument("--w-cat-simclr", type=float, default=1.0)
    parser.add_argument("--w-cat-it", type=float, default=1.0)

    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--output-dir", type=str, default="/Users/boud/mlpractical/final_project/open_clip/od_training")
    parser.add_argument("--experiment-name", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device, args.model)
    use_amp = bool(args.amp and device.type == "cuda")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.experiment_name:
        exp_dir = out_root / args.experiment_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = args.model.replace("/", "_").replace("@", "_")
        exp_dir = out_root / f"{model_safe}_csn_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.split_dir is None:
        args.split_dir = str(exp_dir / "splits")

    print(f"Output dir: {exp_dir}")
    print(f"Device: {device}  AMP(cuda-only): {use_amp}")

    # data
    records, data_stats = load_csn_records(args.csv_file, args.base_image_dir)
    train_idx, test_idx, split_meta = create_or_load_split(
        records=records,
        split_dir=args.split_dir,
        seed=args.seed,
        force_resplit=args.force_resplit,
        train_ratio=0.5,
    )
    print(f"Loaded records: {len(records)}  train={len(train_idx)} test={len(test_idx)}")

    # model
    model, preprocess = clip.load(args.model, device=device, jit=False)
    model = model.float()
    freeze_clip_model(model)

    with torch.no_grad():
        dummy_img = preprocess(Image.new("RGB", (224, 224))).unsqueeze(0).to(device=device, dtype=torch.float32)
        img_dim = int(model.encode_image(dummy_img).shape[-1])

    image_head = ProjectionHead(img_dim, args.hidden_dim, args.proj_dim).to(device).float()
    csn_mask = SharedCSNMask(args.proj_dim, mask_init=args.mask_init).to(device).float()

    if args.w_super_it != 0.0 or args.w_cat_it != 0.0:
        print("Note: --w-super-it and --w-cat-it are ignored in visual-only mode.")

    tokenizer = lambda texts: clip.tokenize(texts, truncate=True)

    train_ds = CSNMultiViewDataset(
        records=records,
        indices=train_idx,
        image_transform=preprocess,
        text_tokenize_fn=tokenizer,
        seed=args.seed,
    )
    test_ds = CSNMultiViewDataset(
        records=records,
        indices=test_idx,
        image_transform=preprocess,
        text_tokenize_fn=tokenizer,
        seed=args.seed + 1,
    )

    pin_memory = device.type == "cuda"
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_csn_batch,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_csn_batch,
    )

    supcon_loss = SupConLoss(temperature=args.temperature).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": list(image_head.parameters()), "weight_decay": float(args.linear_weight_decay)},
            {"params": list(csn_mask.parameters()), "weight_decay": float(args.weight_decay)},
        ],
        lr=args.lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    weights = {
        "w_super_simclr": float(args.w_super_simclr),
        "w_cat_simclr": float(args.w_cat_simclr),
    }

    state = TrainState(epoch=0, best_metric=float("inf"), best_epoch=0, train_history=[], test_history=[])
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        image_head.load_state_dict(ckpt["image_head_state_dict"], strict=True)
        csn_mask.load_state_dict(ckpt["csn_mask_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if use_amp and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        ts_data = ckpt.get("train_state", {})
        state = TrainState(
            epoch=int(ts_data.get("epoch", ckpt.get("epoch", 0))),
            best_metric=float(ts_data.get("best_metric", float("inf"))),
            best_epoch=int(ts_data.get("best_epoch", 0)),
            train_history=list(ts_data.get("train_history", [])),
            test_history=list(ts_data.get("test_history", [])),
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    # save static config now
    config = {
        "args": vars(args),
        "device": str(device),
        "amp_enabled": use_amp,
        "data_stats": data_stats,
        "split_meta": split_meta,
        "img_embed_dim": img_dim,
        "visual_only": True,
        "ignored_weights": {
            "w_super_it": float(args.w_super_it),
            "w_cat_it": float(args.w_cat_it),
        },
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(
            loader=train_loader,
            model=model,
            image_head=image_head,
            csn_mask=csn_mask,
            supcon_loss=supcon_loss,
            weights=weights,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            train=True,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        train_row = {"epoch": epoch, **asdict(train_loss)}
        state.train_history.append(train_row)

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        metric_for_best = train_loss.total
        if do_eval:
            test_loss = run_epoch(
                loader=test_loader,
                model=model,
                image_head=image_head,
                csn_mask=csn_mask,
                supcon_loss=supcon_loss,
                weights=weights,
                optimizer=None,
                scaler=scaler,
                use_amp=use_amp,
                train=False,
                epoch=epoch,
                total_epochs=args.epochs,
            )
            test_row = {"epoch": epoch, **asdict(test_loss)}
            state.test_history.append(test_row)
            metric_for_best = test_loss.total

        is_best = metric_for_best < state.best_metric
        if is_best:
            state.best_metric = metric_for_best
            state.best_epoch = epoch

        state.epoch = epoch

        if (epoch % args.save_every == 0) or (epoch == args.epochs) or is_best:
            save_checkpoint(
                out_dir=exp_dir,
                epoch=epoch,
                args=args,
                image_head=image_head,
                csn_mask=csn_mask,
                optimizer=optimizer,
                scaler=scaler if use_amp else None,
                train_state=state,
                is_best=is_best,
            )

        save_loss_log(exp_dir, state)
        plot_loss_curves(exp_dir, state.train_history, state.test_history)

        dt = time.perf_counter() - t0
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_total={train_loss.total:.4f} "
            f"best_metric={state.best_metric:.4f} "
            f"time={dt:.1f}s"
        )

    print("Training complete")
    print(f"Best epoch: {state.best_epoch}  best_metric={state.best_metric:.4f}")
    print(f"Artifacts: {exp_dir}")


if __name__ == "__main__":
    main()
