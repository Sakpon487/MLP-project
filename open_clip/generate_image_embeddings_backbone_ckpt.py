#!/usr/bin/env python3
"""
Generate image embeddings using a finetuned CLIP backbone checkpoint.

This script is paired with:
  - train_clip_supcon_backbone.py

Supports --device cuda, mps (Apple Silicon), or cpu. ResNet models
automatically use CPU when MPS is requested (MPS has compatibility issues).

Input txt format:
  image_id class_id super_class_id path

Outputs:
  - <prefix>_embeddings.npy (N x D)
  - <prefix>_super_class_ids.npy (N,)
  - <prefix>_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import sys

CLIP_DIR = Path(__file__).parent.parent / "CLIP"
if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

import clip  # type: ignore


def load_dataset(dataset_file: str | Path, base_image_dir: str | Path | None) -> List[Tuple[str, int]]:
    dataset_file = Path(dataset_file)
    base_image_dir = Path(base_image_dir) if base_image_dir else None

    samples: List[Tuple[str, int]] = []
    with open(dataset_file, "r") as f:
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
            full_path = (base_image_dir / rel_path) if base_image_dir else Path(rel_path)
            if not full_path.exists():
                continue
            samples.append((str(full_path), super_class_id))
    return samples


def load_backbone_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint missing 'model_state_dict': {checkpoint_path}")
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    info = {
        "checkpoint_path": str(checkpoint_path),
        "epoch": ckpt.get("epoch", None),
        "loss": ckpt.get("loss", None),
        "model_name_in_ckpt": ckpt.get("model_name", None),
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys_sample": missing[:20],
        "unexpected_keys_sample": unexpected[:20],
    }
    return info


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from finetuned CLIP backbone checkpoint")
    parser.add_argument("--dataset-file", type=str, required=True)
    parser.add_argument("--base-image-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="ViT-B/32")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to *.pt checkpoint from train_clip_supcon_backbone.py")
    parser.add_argument("--batch-size", type=int, default=64)
    def _default_device():
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device (default: cuda if available, else mps, else cpu). ResNet models auto-use CPU when --device mps.",
    )
    parser.add_argument("--output-dir", type=str, default="./embeddings")
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--normalize", action="store_true", default=True, help="L2-normalize embeddings (default: True)")
    args = parser.parse_args()

    if args.device is None:
        args.device = _default_device()

    # MPS fallback if unavailable
    if args.device == "mps" and (not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available()):
        print("Warning: MPS requested but not available, using CPU.")
        args.device = "cpu"

    # ResNet models have fatal MPS compatibility issues; use CPU before loading
    if args.device == "mps" and args.model.strip().upper().startswith("RN"):
        print("Warning: ResNet models have MPS compatibility issues. Using CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    dataset_file = Path(args.dataset_file)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    samples = load_dataset(dataset_file, args.base_image_dir)
    if not samples:
        raise ValueError("No valid samples found. Check dataset paths/base-image-dir.")

    # load base CLIP model & preprocess
    model, preprocess = clip.load(args.model, device=device, jit=False)
    model = model.float()  # ensure checkpoint (float32) loads cleanly and inference is stable
    model.eval()

    ckpt_info = load_backbone_checkpoint(model, checkpoint_path)

    # Determine embedding dimension
    with torch.no_grad():
        dummy = preprocess(Image.new("RGB", (224, 224))).unsqueeze(0).to(device=device, dtype=torch.float32)
        emb = model.encode_image(dummy).float()
        if args.normalize:
            emb = F.normalize(emb, dim=-1)
        embed_dim = emb.shape[1]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_prefix is None:
        model_safe = args.model.replace("/", "_").replace("@", "_")
        args.output_prefix = f"{model_safe}_ckpt"

    all_embeddings: List[np.ndarray] = []
    all_super: List[int] = []

    for i in tqdm(range(0, len(samples), args.batch_size), desc="Embedding batches"):
        batch = samples[i : i + args.batch_size]
        imgs = []
        supers = []
        for path, sid in batch:
            try:
                img = Image.open(path).convert("RGB")
                imgs.append(preprocess(img))
                supers.append(sid)
            except Exception:
                continue
        if not imgs:
            continue

        x = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            feats = model.encode_image(x).float()
            if args.normalize:
                feats = F.normalize(feats, dim=-1)
        all_embeddings.append(feats.cpu().numpy())
        all_super.extend(supers)

    embeddings = np.vstack(all_embeddings)
    super_class_ids = np.array(all_super, dtype=np.int64)

    emb_path = out_dir / f"{args.output_prefix}_embeddings.npy"
    sup_path = out_dir / f"{args.output_prefix}_super_class_ids.npy"
    meta_path = out_dir / f"{args.output_prefix}_metadata.json"

    np.save(emb_path, embeddings)
    np.save(sup_path, super_class_ids)

    metadata = {
        "dataset_file": str(dataset_file),
        "num_images": int(embeddings.shape[0]),
        "embedding_dim": int(embed_dim),
        "model": args.model,
        "device": str(device),
        "normalized": bool(args.normalize),
        "checkpoint": ckpt_info,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved embeddings: {emb_path}  shape={embeddings.shape}")
    print(f"Saved super_class_ids: {sup_path}  shape={super_class_ids.shape}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()

