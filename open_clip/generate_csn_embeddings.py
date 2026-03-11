#!/usr/bin/env python3
"""Generate CSN embeddings from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

CLIP_DIR = Path(__file__).parent.parent / "CLIP"
if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

import clip  # type: ignore

from csn_pipeline.data import load_csn_records
from csn_pipeline.model import ProjectionHead, SharedCSNMask


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
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(d)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate image/text CSN embeddings from checkpoint")
    parser.add_argument("--csv-file", type=str, required=True)
    parser.add_argument("--base-image-dir", type=str, default=None)
    parser.add_argument("--split-indices", type=str, required=True, help="Path to test_indices.npy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--output-dir", type=str, default="./embeddings")
    parser.add_argument("--output-prefix", type=str, default="csn_test")
    parser.add_argument("--normalize", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device, args.model)

    records, _ = load_csn_records(args.csv_file, args.base_image_dir)
    split_indices = np.load(args.split_indices).astype(np.int64)
    if split_indices.size == 0:
        raise ValueError("split_indices is empty")

    max_idx = len(records) - 1
    if int(split_indices.max()) > max_idx:
        raise ValueError("split_indices contains index beyond record count")

    model, preprocess = clip.load(args.model, device=device, jit=False)
    model = model.float().eval()

    with torch.no_grad():
        dummy_img = preprocess(Image.new("RGB", (224, 224))).unsqueeze(0).to(device=device, dtype=torch.float32)
        dummy_txt = clip.tokenize(["a photo"]).to(device)
        img_dim = int(model.encode_image(dummy_img).shape[-1])
        txt_dim = int(model.encode_text(dummy_txt).shape[-1])

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)

    ckpt_args = ckpt.get("args", {})
    hidden_dim = int(ckpt_args.get("hidden_dim", 512))
    proj_dim = int(ckpt_args.get("proj_dim", 128))
    mask_init = float(ckpt_args.get("mask_init", 0.0))

    image_head = ProjectionHead(img_dim, hidden_dim, proj_dim).to(device).float().eval()
    text_head = ProjectionHead(txt_dim, hidden_dim, proj_dim).to(device).float().eval()
    csn_mask = SharedCSNMask(proj_dim, mask_init=mask_init).to(device).float().eval()

    image_head.load_state_dict(ckpt["image_head_state_dict"], strict=True)
    text_head.load_state_dict(ckpt["text_head_state_dict"], strict=True)
    csn_mask.load_state_dict(ckpt["csn_mask_state_dict"], strict=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_super_all = []
    image_cat_all = []
    text_super_all = []
    text_cat_all = []
    superclass_ids = []
    category_ids = []
    paths = []

    for i in tqdm(range(0, split_indices.size, args.batch_size), desc="Embedding batches"):
        batch_idxs = split_indices[i : i + args.batch_size]
        imgs = []
        txts = []
        sup = []
        cat = []
        pths = []

        for idx in batch_idxs.tolist():
            r = records[int(idx)]
            try:
                img = Image.open(r.image_path).convert("RGB")
                img_t = preprocess(img)
            except Exception:
                img_t = preprocess(Image.new("RGB", (224, 224), color="black"))
            imgs.append(img_t)
            txts.append(r.clip_text)
            sup.append(int(r.superclass_id))
            cat.append(int(r.category_id))
            pths.append(r.image_path)

        x_img = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)
        x_txt = clip.tokenize(txts, truncate=True).to(device)

        with torch.no_grad():
            f_img = model.encode_image(x_img).float()
            f_txt = model.encode_text(x_txt).float()

            z_img = image_head(f_img)
            z_txt = text_head(f_txt)

            z_img_cat = csn_mask(z_img)
            z_txt_cat = csn_mask(z_txt)

            if args.normalize:
                z_img = F.normalize(z_img, dim=-1)
                z_txt = F.normalize(z_txt, dim=-1)
                z_img_cat = F.normalize(z_img_cat, dim=-1)
                z_txt_cat = F.normalize(z_txt_cat, dim=-1)

        image_super_all.append(z_img.cpu().numpy())
        image_cat_all.append(z_img_cat.cpu().numpy())
        text_super_all.append(z_txt.cpu().numpy())
        text_cat_all.append(z_txt_cat.cpu().numpy())
        superclass_ids.extend(sup)
        category_ids.extend(cat)
        paths.extend(pths)

    image_super = np.vstack(image_super_all)
    image_cat = np.vstack(image_cat_all)
    text_super = np.vstack(text_super_all)
    text_cat = np.vstack(text_cat_all)

    superclass_ids_np = np.asarray(superclass_ids, dtype=np.int64)
    category_ids_np = np.asarray(category_ids, dtype=np.int64)
    paths_np = np.asarray(paths)

    prefix = args.output_prefix
    out_files = {
        "image_super": out_dir / f"{prefix}_image_super_embeddings.npy",
        "image_category": out_dir / f"{prefix}_image_category_embeddings.npy",
        "text_super": out_dir / f"{prefix}_text_super_embeddings.npy",
        "text_category": out_dir / f"{prefix}_text_category_embeddings.npy",
        "superclass_ids": out_dir / f"{prefix}_superclass_ids.npy",
        "category_ids": out_dir / f"{prefix}_category_ids.npy",
        "paths": out_dir / f"{prefix}_paths.npy",
        "metadata": out_dir / f"{prefix}_metadata.json",
    }

    np.save(out_files["image_super"], image_super)
    np.save(out_files["image_category"], image_cat)
    np.save(out_files["text_super"], text_super)
    np.save(out_files["text_category"], text_cat)
    np.save(out_files["superclass_ids"], superclass_ids_np)
    np.save(out_files["category_ids"], category_ids_np)
    np.save(out_files["paths"], paths_np)

    metadata = {
        "csv_file": str(Path(args.csv_file).resolve()),
        "split_indices": str(Path(args.split_indices).resolve()),
        "checkpoint": str(ckpt_path.resolve()),
        "model": args.model,
        "device": str(device),
        "normalize": bool(args.normalize),
        "num_samples": int(image_super.shape[0]),
        "proj_dim": int(image_super.shape[1]),
        "outputs": {k: str(v) for k, v in out_files.items()},
    }
    with open(out_files["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)

    for name, p in out_files.items():
        print(f"Saved {name}: {p}")


if __name__ == "__main__":
    main()
