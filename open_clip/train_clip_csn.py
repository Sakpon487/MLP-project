#!/usr/bin/env python3
"""
Train CLIP-frozen visual-only CSN pipeline.

- Superclass similarity on full projection embeddings.
- Category similarity on shared-mask CSN embeddings.
- Visual branch only: CLIP image encoder -> projection head -> shared CSN mask.
"""

from __future__ import annotations

import argparse
import hashlib
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# local CLIP checkout
CLIP_DIR = Path(__file__).parent.parent / "CLIP"
if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

import clip  # type: ignore

from csn_pipeline.data import create_or_load_split, load_csn_records
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


class CSNIndexMultiViewDataset(Dataset):
    """Multi-view dataset returning global record indices for (anchor, super+, cat+)."""

    def __init__(self, records, indices: np.ndarray, seed: int = 0):
        self.records = records
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rng = random.Random(seed)

        self.super_to_local: dict[int, list[int]] = {}
        self.cat_to_local: dict[int, list[int]] = {}
        for local_pos, global_idx in enumerate(self.indices.tolist()):
            rec = self.records[int(global_idx)]
            self.super_to_local.setdefault(int(rec.superclass_id), []).append(local_pos)
            self.cat_to_local.setdefault(int(rec.category_id), []).append(local_pos)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _sample_positive_local(self, pool: list[int], anchor_local: int) -> int:
        candidates = [i for i in pool if i != anchor_local]
        if not candidates:
            return anchor_local
        return self.rng.choice(candidates)

    def __getitem__(self, local_idx: int) -> dict[str, int]:
        anchor_global = int(self.indices[local_idx])
        anchor = self.records[anchor_global]

        super_pool = self.super_to_local[int(anchor.superclass_id)]
        cat_pool = self.cat_to_local[int(anchor.category_id)]

        pos_super_local = self._sample_positive_local(super_pool, int(local_idx))
        pos_cat_local = self._sample_positive_local(cat_pool, int(local_idx))

        super_global = int(self.indices[pos_super_local])
        cat_global = int(self.indices[pos_cat_local])

        return {
            "idx_view1": anchor_global,
            "idx_view2": super_global,
            "idx_view3": cat_global,
            "label_view1_2": int(anchor.superclass_id),
            "label_view1_3": int(anchor.category_id),
        }


def collate_csn_index_batch(batch: list[dict[str, int]]) -> dict[str, torch.Tensor]:
    return {
        "idx_view1": torch.tensor([b["idx_view1"] for b in batch], dtype=torch.long),
        "idx_view2": torch.tensor([b["idx_view2"] for b in batch], dtype=torch.long),
        "idx_view3": torch.tensor([b["idx_view3"] for b in batch], dtype=torch.long),
        "label_view1_2": torch.tensor([b["label_view1_2"] for b in batch], dtype=torch.long),
        "label_view1_3": torch.tensor([b["label_view1_3"] for b in batch], dtype=torch.long),
    }


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


def default_split_dir_for_csv(csv_file: str | Path, seed: int, train_ratio: float) -> Path:
    csv_path = Path(csv_file).resolve()
    ratio_tag = int(round(float(train_ratio) * 100))
    return csv_path.parent / f"{csv_path.stem}_splits_seed{int(seed)}_tr{ratio_tag}"


def _records_path_hash(records) -> str:
    h = hashlib.sha1()
    for rec in records:
        h.update(str(rec.image_path).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_or_build_clip_image_cache(
    model: torch.nn.Module,
    preprocess,
    records,
    model_name: str,
    cache_path: Path,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path.with_suffix(".meta.json")

    expected = {
        "model": str(model_name),
        "num_records": int(len(records)),
        "records_path_hash": _records_path_hash(records),
    }

    if cache_path.exists() and meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            arr = np.load(cache_path)
            if (
                arr.ndim == 2
                and int(arr.shape[0]) == expected["num_records"]
                and str(meta.get("model")) == expected["model"]
                and str(meta.get("records_path_hash")) == expected["records_path_hash"]
            ):
                print(f"Loaded CLIP image cache: {cache_path}  shape={arr.shape}")
                return torch.from_numpy(np.asarray(arr, dtype=np.float32))
            print("CLIP image cache metadata mismatch; rebuilding cache.")
        except Exception as e:
            print(f"Failed to load CLIP image cache ({e}); rebuilding cache.")

    print(f"Building CLIP image cache for {len(records)} records...")
    feats_cpu: list[torch.Tensor] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(records), batch_size), desc="CLIP image cache"):
            batch_records = records[i : i + batch_size]
            imgs = []
            for rec in batch_records:
                try:
                    img = Image.open(rec.image_path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), color="black")
                imgs.append(preprocess(img))

            x = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)
            f = model.encode_image(x).float().cpu()
            feats_cpu.append(f)

    features = torch.cat(feats_cpu, dim=0).contiguous()
    np.save(cache_path, features.numpy())

    meta = {
        **expected,
        "feature_dim": int(features.shape[1]),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved CLIP image cache: {cache_path}  shape={tuple(features.shape)}")
    return features


def compute_losses(
    clip_feature_cache: torch.Tensor,
    batch: dict[str, torch.Tensor],
    image_head: torch.nn.Module,
    csn_mask: torch.nn.Module,
    supcon_loss: SupConLoss,
    weights: dict[str, float],
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = next(image_head.parameters()).device

    idx_views = torch.stack(
        [batch["idx_view1"], batch["idx_view2"], batch["idx_view3"]],
        dim=1,
    )
    labels_super = batch["label_view1_2"].to(device=device, non_blocking=True)
    labels_cat = batch["label_view1_3"].to(device=device, non_blocking=True)

    bsz, n_views = idx_views.shape
    idx_flat = idx_views.view(-1).cpu()

    with torch.cuda.amp.autocast(enabled=use_amp):
        img_feat_flat = clip_feature_cache.index_select(0, idx_flat).to(device=device, dtype=torch.float32, non_blocking=True)

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
    clip_feature_cache: torch.Tensor,
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
                clip_feature_cache=clip_feature_cache,
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

    train_ratio = 0.5
    if args.split_dir is None:
        args.split_dir = str(default_split_dir_for_csv(args.csv_file, args.seed, train_ratio))

    print(f"Output dir: {exp_dir}")
    print(f"Device: {device}  AMP(cuda-only): {use_amp}")
    print(f"Split dir: {args.split_dir}")

    # data
    records, data_stats = load_csn_records(args.csv_file, args.base_image_dir)
    train_idx, test_idx, split_meta = create_or_load_split(
        records=records,
        split_dir=args.split_dir,
        seed=args.seed,
        force_resplit=args.force_resplit,
        train_ratio=train_ratio,
    )
    print(f"Loaded records: {len(records)}  train={len(train_idx)} test={len(test_idx)}")

    # model and CLIP cache
    model, preprocess = clip.load(args.model, device=device, jit=False)
    model = model.float()
    freeze_clip_model(model)

    model_safe = args.model.replace("/", "_").replace("@", "_")
    split_dir_path = Path(args.split_dir)
    cache_path = split_dir_path / f"{model_safe}_clip_image_features.npy"

    clip_feature_cache = load_or_build_clip_image_cache(
        model=model,
        preprocess=preprocess,
        records=records,
        model_name=args.model,
        cache_path=cache_path,
        device=device,
        batch_size=max(int(args.batch_size), 64),
    )

    img_dim = int(clip_feature_cache.shape[1])
    if device.type == "cuda":
        clip_feature_cache = clip_feature_cache.pin_memory()

    # CLIP model no longer needed after feature cache is ready
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    image_head = ProjectionHead(img_dim, args.hidden_dim, args.proj_dim).to(device).float()
    csn_mask = SharedCSNMask(args.proj_dim, mask_init=args.mask_init).to(device).float()

    if args.w_super_it != 0.0 or args.w_cat_it != 0.0:
        print("Note: --w-super-it and --w-cat-it are ignored in visual-only mode.")

    train_ds = CSNIndexMultiViewDataset(records=records, indices=train_idx, seed=args.seed)
    test_ds = CSNIndexMultiViewDataset(records=records, indices=test_idx, seed=args.seed + 1)

    pin_memory = device.type == "cuda"
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_csn_index_batch,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_csn_index_batch,
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
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"Warning: could not load optimizer state from resume checkpoint ({e}); continuing with fresh optimizer.")
        if use_amp and ckpt.get("scaler_state_dict") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception as e:
                print(f"Warning: could not load AMP scaler state ({e}); continuing with fresh scaler.")

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
        "clip_image_cache_path": str(cache_path),
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
            clip_feature_cache=clip_feature_cache,
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
                clip_feature_cache=clip_feature_cache,
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
