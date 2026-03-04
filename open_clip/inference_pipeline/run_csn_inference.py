#!/usr/bin/env python3
"""Metrics-only evaluation for CSN embeddings (recall@K and precision@K)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_array(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if len(arr.files) != 1:
            raise ValueError(f"Expected one array in npz: {path}")
        return np.asarray(arr[arr.files[0]])
    return np.asarray(arr)


def compute_retrieval_metrics_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_list: list[int],
    device: torch.device,
    batch_size: int = 512,
) -> tuple[dict[int, float], dict[int, float], list[int]]:
    emb = torch.from_numpy(embeddings).float().to(device)
    emb = torch.nn.functional.normalize(emb, dim=1)
    labels_t = torch.from_numpy(labels).to(device)

    n = emb.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples for retrieval metrics")

    k_req = sorted(set(int(k) for k in k_list if int(k) > 0))
    if not k_req:
        raise ValueError("rank-k must include positive integers")

    max_allowed = n - 1
    max_k = max(min(k, max_allowed) for k in k_req)

    recall_hits = {k: 0 for k in k_req}
    precision_sum = {k: 0.0 for k in k_req}

    for i in tqdm(range(0, n, batch_size), desc="recall/precision"):
        end = min(i + batch_size, n)
        q = emb[i:end]
        q_labels = labels_t[i:end]
        sim = q @ emb.T
        for j in range(sim.shape[0]):
            sim[j, i + j] = float("-inf")
        top_idx = torch.topk(sim, max_k, dim=1).indices
        retrieved = labels_t[top_idx]
        same = retrieved == q_labels.unsqueeze(1)
        for k in k_req:
            kk = min(k, max_allowed)
            s = same[:, :kk]
            recall_hits[k] += int(s.any(dim=1).sum().item())
            precision_sum[k] += float(s.float().mean(dim=1).sum().item())

    recall = {k: recall_hits[k] / n for k in k_req}
    precision = {k: precision_sum[k] / n for k in k_req}
    clipped = sorted([k for k in k_req if k > max_allowed])
    return recall, precision, clipped


def evaluate_one(
    embeddings_path: Path,
    labels_path: Path,
    rank_k: list[int],
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    tag: str,
) -> None:
    embeddings = load_array(embeddings_path).astype(np.float32)
    labels = load_array(labels_path).astype(np.int64).flatten()

    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Shape mismatch for {tag}: embeddings={embeddings.shape[0]}, labels={labels.shape[0]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    recall, precision, clipped = compute_retrieval_metrics_at_k(
        embeddings=embeddings,
        labels=labels,
        k_list=rank_k,
        device=device,
        batch_size=batch_size,
    )

    recall_path = output_dir / "recall_at_k.txt"
    precision_path = output_dir / "precision_at_k.txt"
    metrics_path = output_dir / "metrics.json"

    with open(recall_path, "w") as f:
        for k in rank_k:
            f.write(f"Recall@{k}: {recall[k]:.4f}\n")

    with open(precision_path, "w") as f:
        for k in rank_k:
            f.write(f"Precision@{k}: {precision[k]:.4f}\n")

    payload = {
        "tag": tag,
        "embeddings": str(embeddings_path.resolve()),
        "labels": str(labels_path.resolve()),
        "num_samples": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "rank_k": rank_k,
        "recall": {str(k): float(v) for k, v in recall.items()},
        "precision": {str(k): float(v) for k, v in precision.items()},
        "clipped_k": clipped,
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[{tag}] saved recall: {recall_path}")
    print(f"[{tag}] saved precision: {precision_path}")
    print(f"[{tag}] saved metrics: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metrics-only CSN embedding evaluation")

    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument(
        "--prefix-metadata",
        type=Path,
        default=None,
        help="Metadata json from generate_csn_embeddings.py; evaluates default 4 spaces.",
    )

    parser.add_argument("--rank-k", type=int, nargs="+", default=[1, 5, 10, 100, 1000])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--output-dir", type=Path, default=Path("./csn_inference_output"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.prefix_metadata is None and (args.embeddings is None or args.labels is None):
        raise ValueError("Provide either --prefix-metadata OR both --embeddings and --labels")

    if args.prefix_metadata is not None:
        with open(args.prefix_metadata.resolve(), "r") as f:
            meta = json.load(f)

        outputs = meta.get("outputs", {})
        paths = {
            "image_super": Path(outputs["image_super"]),
            "image_category": Path(outputs["image_category"]),
            "text_super": Path(outputs["text_super"]),
            "text_category": Path(outputs["text_category"]),
            "superclass_ids": Path(outputs["superclass_ids"]),
            "category_ids": Path(outputs["category_ids"]),
        }

        matrix = [
            ("image_super", paths["image_super"], paths["superclass_ids"]),
            ("image_category", paths["image_category"], paths["category_ids"]),
            ("text_super", paths["text_super"], paths["superclass_ids"]),
            ("text_category", paths["text_category"], paths["category_ids"]),
        ]

        for tag, emb_path, lbl_path in matrix:
            evaluate_one(
                embeddings_path=emb_path,
                labels_path=lbl_path,
                rank_k=args.rank_k,
                batch_size=args.batch_size,
                device=device,
                output_dir=args.output_dir / tag,
                tag=tag,
            )
    else:
        evaluate_one(
            embeddings_path=args.embeddings.resolve(),
            labels_path=args.labels.resolve(),
            rank_k=args.rank_k,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_dir,
            tag="single",
        )

    print("Done")


if __name__ == "__main__":
    main()
