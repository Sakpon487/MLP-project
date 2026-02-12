#!/usr/bin/env python3
"""
Finetune CLIP vision backbone with supervised contrastive (SupCon) loss.

Data format (same as SOP/.data/Ebay_train.txt):
  image_id class_id super_class_id path
  1 1 1 bicycle_final/111085122871_0.JPG

Training:
- Loads OpenAI CLIP via local `CLIP/` checkout (module name: `clip`)
- Finetunes ONLY the vision tower (image encoder) end-to-end (no projection head)
- Positives are other images with the same `super_class_id` within the batch
- Uses PK sampling to ensure each batch contains >=2 samples per class

Outputs:
- Checkpoints: checkpoint_epoch_XXXX.pt, best_checkpoint.pt, latest_checkpoint.pt
- loss_log.json and loss_curve.png
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


# ---- import OpenAI CLIP from local checkout ----
import sys

CLIP_DIR = Path(__file__).parent.parent / "CLIP"
if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

import clip  # type: ignore


# ---- dataset ----


class SuperClassDataset(Dataset):
    """Reads (image_path, super_class_id) from the SOP-format txt file."""

    def __init__(self, dataset_file: str | Path, base_image_dir: str | Path | None, transform):
        self.dataset_file = Path(dataset_file)
        self.base_image_dir = Path(base_image_dir) if base_image_dir else None
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        self.class_to_indices: Dict[int, List[int]] = {}

        with open(self.dataset_file, "r") as f:
            _header = f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                super_class_id = int(parts[2])
                rel_path = " ".join(parts[3:])
                full_path = (self.base_image_dir / rel_path) if self.base_image_dir else Path(rel_path)
                if not full_path.exists():
                    continue
                idx = len(self.samples)
                self.samples.append((str(full_path), super_class_id))
                self.class_to_indices.setdefault(super_class_id, []).append(idx)

        self.classes: List[int] = sorted(self.class_to_indices.keys())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, super_class_id = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        # SimCLR / SupCon style: two augmented views of the same image
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2, super_class_id


class PKBatchSampler(Sampler[List[int]]):
    """
    Yields batches built by sampling P classes and K samples per class (with replacement if needed).
    """

    def __init__(
        self,
        class_to_indices: Dict[int, Sequence[int]],
        classes_per_batch: int,
        samples_per_class: int,
        steps_per_epoch: int,
        seed: int = 0,
    ):
        self.class_to_indices = {c: list(idxs) for c, idxs in class_to_indices.items()}
        self.classes = list(self.class_to_indices.keys())
        self.P = classes_per_batch
        self.K = samples_per_class
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed

        if self.P <= 0 or self.K <= 1:
            raise ValueError("PK sampling requires classes_per_batch > 0 and samples_per_class >= 2.")
        if len(self.classes) < self.P:
            raise ValueError(f"Not enough classes ({len(self.classes)}) for classes_per_batch={self.P}.")

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(self.seed)
        for _ in range(self.steps_per_epoch):
            chosen_classes = rng.sample(self.classes, k=self.P)
            batch: List[int] = []
            for c in chosen_classes:
                idxs = self.class_to_indices[c]
                if len(idxs) >= self.K:
                    batch.extend(rng.sample(idxs, k=self.K))
                else:
                    # sample with replacement
                    batch.extend(rng.choices(idxs, k=self.K))
            yield batch


def collate_views_labels(batch):
    view1, view2, labels = zip(*batch)
    v1 = torch.stack(view1, dim=0)
    v2 = torch.stack(view2, dim=0)
    images = torch.stack([v1, v2], dim=1)  # [bsz, 2, C, H, W]
    return images, torch.tensor(labels, dtype=torch.long)


# ---- loss ----


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
    Also supports the unsupervised contrastive loss in SimCLR.

    This implementation matches the reference structure the user provided.
    In supervised mode (labels provided), we additionally set the diagonal of the
    (bsz x bsz) label mask to 0 so same-instance pairs (including other-view)
    are excluded as positives.
    """

    def __init__(self, temperature: float = 0.07, contrast_mode: str = "all", base_temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)
        self.contrast_mode = str(contrast_mode)
        self.base_temperature = float(base_temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None, mask: torch.Tensor | None = None):
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]; at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
            # exclude same-instance positives (including other-view)
            mask.fill_diagonal_(0.0)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        pos_counts = mask.sum(1)
        if torch.any(pos_counts == 0):
            raise RuntimeError(
                "SupConLoss: some anchors have zero positives in-batch. "
                "Increase --samples-per-class / adjust PK sampling."
            )
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_counts

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


# ---- checkpointing / plotting ----


def save_checkpoint(
    out_dir: Path,
    epoch: int,
    model_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    loss_value: float,
    is_best: bool,
    args_dict: dict,
):
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_name": model_name,
        "loss": loss_value,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": args_dict,
    }

    epoch_path = ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(ckpt, epoch_path)

    latest_path = ckpt_dir / "latest_checkpoint.pt"
    torch.save(ckpt, latest_path)

    if is_best:
        best_path = ckpt_dir / "best_checkpoint.pt"
        torch.save(ckpt, best_path)


def plot_loss_curve(losses: Sequence[float], save_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SupCon Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---- training ----


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(preprocess, image_size: int):
    """
    Build a simple augmentation pipeline that stays compatible with CLIP normalization.
    We keep CLIP mean/std but use RandomResizedCrop + HorizontalFlip.
    """
    from torchvision.transforms import Compose, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Normalize

    # Extract mean/std from CLIP preprocess Normalize
    norm = None
    for t in preprocess.transforms:
        if isinstance(t, Normalize):
            norm = t
            break
    if norm is None:
        # fallback to OpenAI CLIP mean/std
        norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    return Compose(
        [
            RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            norm,
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", type=str, required=True)
    parser.add_argument("--base-image-dir", type=str, default=None)

    parser.add_argument("--model", type=str, default="ViT-B/32", help="Default: smallest ViT CLIP variant")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5, help="Backbone finetune LR (start small)")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--contrast-mode", type=str, default="all", choices=["all", "one"])
    parser.add_argument(
        "--base-temperature",
        type=float,
        default=None,
        help="If unset, defaults to --temperature",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=None)

    parser.add_argument("--classes-per-batch", type=int, default=16, help="P in PK sampling")
    parser.add_argument("--samples-per-class", type=int, default=2, help="K in PK sampling (>=2)")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="If unset, computed from dataset size")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/boud/mlpractical/final_project/open_clip/od_training",
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoints every N epochs")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    # output folder
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.experiment_name:
        out_dir = out_root / args.experiment_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = args.model.replace("/", "_").replace("@", "_")
        out_dir = out_root / f"{model_safe}_supcon_backbone_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    # load CLIP
    model, preprocess = clip.load(args.model, device=device, jit=False)
    # ensure float32 weights for finetuning (AMP handles mixed precision)
    model = model.float()
    model.train()

    # freeze text tower explicitly (safety)
    for p in model.transformer.parameters():
        p.requires_grad = False
    for p in model.token_embedding.parameters():
        p.requires_grad = False
    for p in model.ln_final.parameters():
        p.requires_grad = False
    if hasattr(model, "text_projection") and isinstance(model.text_projection, torch.Tensor):
        # text_projection is a Parameter in OpenAI CLIP
        model.text_projection.requires_grad = False  # type: ignore

    # only train vision
    for p in model.visual.parameters():
        p.requires_grad = True

    # train transform
    image_size = getattr(model.visual, "input_resolution", 224)
    train_transform = build_train_transform(preprocess, int(image_size))

    dataset = SuperClassDataset(args.dataset_file, args.base_image_dir, transform=train_transform)
    if len(dataset) == 0:
        raise ValueError("Dataset contains 0 valid images. Check paths/base-image-dir.")

    batch_size = args.classes_per_batch * args.samples_per_class
    steps_per_epoch = args.steps_per_epoch or max(1, len(dataset) // batch_size)
    sampler = PKBatchSampler(
        dataset.class_to_indices,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        steps_per_epoch=steps_per_epoch,
        seed=args.seed,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_views_labels,
    )

    base_temp = float(args.temperature if args.base_temperature is None else args.base_temperature)
    loss_fn = SupConLoss(
        temperature=float(args.temperature),
        contrast_mode=args.contrast_mode,
        base_temperature=base_temp,
    ).to(device)

    # optimizer only on vision params
    optim_params = [p for p in model.visual.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    losses: List[float] = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", total=len(loader))
        for images, labels in pbar:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # images: [bsz, 2, C, H, W] -> flatten views for encoding
                bsz, n_views, c, h, w = images.shape
                x = images.view(bsz * n_views, c, h, w)
                feats_flat = model.encode_image(x).float()
                feats_flat = F.normalize(feats_flat, dim=-1)
                feats = feats_flat.view(bsz, n_views, -1)  # [bsz, 2, dim]
                loss = loss_fn(feats, labels=labels)

            if use_amp:
                scaler.scale(loss).backward()
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(optim_params, args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(optim_params, args.grad_clip_norm)
                optimizer.step()

            running += float(loss.item())
            count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running/max(1,count):.4f}")

        avg_loss = running / max(1, count)
        losses.append(avg_loss)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        # write loss log each epoch
        with open(out_dir / "loss_log.json", "w") as f:
            json.dump(
                {
                    "epochs": list(range(1, epoch + 1)),
                    "losses": losses,
                    "best_loss": best_loss,
                    "best_epoch": int(np.argmin(np.array(losses)) + 1),
                    "batch_size": batch_size,
                    "classes_per_batch": args.classes_per_batch,
                    "samples_per_class": args.samples_per_class,
                        "contrast_mode": args.contrast_mode,
                        "temperature": args.temperature,
                        "base_temperature": base_temp,
                },
                f,
                indent=2,
            )

        if epoch % args.save_every == 0 or epoch == args.epochs or is_best:
            save_checkpoint(
                out_dir=out_dir,
                epoch=epoch,
                model_name=args.model,
                model=model,
                optimizer=optimizer,
                scaler=scaler if use_amp else None,
                loss_value=avg_loss,
                is_best=is_best,
                args_dict=vars(args),
            )

        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f} (best={best_loss:.4f})")

    # plot
    plot_loss_curve(losses, out_dir / "loss_curve.png")

    # save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "temperature": args.temperature,
                "contrast_mode": args.contrast_mode,
                "base_temperature": base_temp,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "amp": use_amp,
                "device": str(device),
                "batch_size": batch_size,
                "classes_per_batch": args.classes_per_batch,
                "samples_per_class": args.samples_per_class,
                "steps_per_epoch": steps_per_epoch,
                "seed": args.seed,
                "final_loss": losses[-1] if losses else None,
                "best_loss": best_loss,
            },
            f,
            indent=2,
        )

    print("Done.")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()

