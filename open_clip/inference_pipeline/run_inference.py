#!/usr/bin/env python3
"""
Inference script: load embeddings + labels, compute rank-X recall,
match vs non-match similarity distribution, and t-SNE visualization.
"""
from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm


def load_data(embeddings_path: Path, labels_path: Path):
    """Load embeddings and super-class IDs from .npy (or .npz) files."""
    emb_load = np.load(embeddings_path, allow_pickle=True)
    if isinstance(emb_load, np.lib.npyio.NpzFile):
        keys = sorted(emb_load.files)
        embeddings = np.stack([emb_load[k] for k in keys]).astype(np.float32)
    else:
        embeddings = np.asarray(emb_load).astype(np.float32)
    labels = np.load(labels_path)
    labels = np.asarray(labels).flatten()
    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"Shape mismatch: embeddings {embeddings.shape[0]}, labels {labels.shape[0]}"
        )
    return embeddings, labels


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)


def compute_retrieval_metrics_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_list: list[int],
    device: torch.device,
    batch_size: int = 512,
) -> tuple[dict[int, float], dict[int, float], list[int]]:
    """
    Compute recall@k and precision@k for each k in k_list (cosine retrieval, self excluded).

    Recall@k: fraction of queries with at least one same-label item in top-k.
    Precision@k: mean fraction of top-k items that share the query label.
    """
    emb = torch.from_numpy(embeddings).float().to(device)
    emb = torch.nn.functional.normalize(emb, dim=1)
    labels_t = torch.from_numpy(labels).to(device)
    N = emb.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 embeddings to compute retrieval metrics.")

    requested_k = sorted(set(int(k) for k in k_list if int(k) > 0))
    if not requested_k:
        raise ValueError("k_list must contain at least one positive integer.")
    max_allowed_k = N - 1
    effective_k = [min(k, max_allowed_k) for k in requested_k]
    max_k = max(effective_k)

    recall_hits = {k: 0 for k in k_list}
    precision_sum = {k: 0.0 for k in k_list}

    for i in tqdm(range(0, N, batch_size), desc="Recall/Precision@k"):
        end = min(i + batch_size, N)
        query = emb[i:end]
        query_labels = labels_t[i:end]
        sim = query @ emb.T
        # Remove self-match
        for j in range(sim.shape[0]):
            sim[j, i + j] = float("-inf")
        topk_idx = torch.topk(sim, max_k, dim=1).indices
        retrieved_labels = labels_t[topk_idx]
        same_label = retrieved_labels == query_labels.unsqueeze(1)
        for k_req in k_list:
            k = min(k_req, max_allowed_k)
            topk_same = same_label[:, :k]
            recall_hits[k_req] += topk_same.any(dim=1).sum().item()
            precision_sum[k_req] += topk_same.float().mean(dim=1).sum().item()

    recall = {k: recall_hits[k] / N for k in k_list}
    precision = {k: precision_sum[k] / N for k in k_list}
    clipped_k = sorted({k for k in k_list if k > max_allowed_k})
    return recall, precision, clipped_k


def compute_match_nonmatch_distribution(
    embeddings: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    block_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise cosine similarity: return (match_sims, nonmatch_sims) (upper triangle only)."""
    emb = torch.from_numpy(embeddings).float().to(device)
    emb = torch.nn.functional.normalize(emb, dim=1)
    N = emb.shape[0]
    cosine_vals = []
    match_flags = []

    for i in tqdm(range(0, N, block_size), desc="Match/non-match similarity"):
        end_i = min(i + block_size, N)
        block_i = emb[i:end_i]
        for j in range(i, N, block_size):
            end_j = min(j + block_size, N)
            block_j = emb[j:end_j]
            sim = block_i @ block_j.T
            if i == j:
                mask = torch.triu(torch.ones_like(sim), diagonal=1).bool()
            else:
                mask = torch.ones_like(sim).bool()
            sim_vals = sim[mask].cpu().numpy()
            cosine_vals.append(sim_vals)
            ids_i = labels[i:end_i]
            ids_j = labels[j:end_j]
            label_block = ids_i[:, None] == ids_j[None, :]
            match_flags.append(label_block[mask.cpu().numpy()])

    cosine_vals = np.concatenate(cosine_vals)
    match_flags = np.concatenate(match_flags)
    return cosine_vals[match_flags], cosine_vals[~match_flags]


def plot_match_nonmatch_distribution(
    match_sims: np.ndarray,
    nonmatch_sims: np.ndarray,
    out_path: Path,
    bins: int = 50,
) -> None:
    """Plot and save match vs non-match similarity distribution."""
    plt.figure(figsize=(8, 4))
    plt.hist(match_sims, bins=bins, alpha=0.5, density=True, label="Match")
    plt.hist(nonmatch_sims, bins=bins, alpha=0.5, density=True, label="Non-match")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title("Match vs Non-match Similarity Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def run_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    subsample: int | None = 5000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run t-SNE on embeddings. Optionally subsample for speed.
    Returns (tsne_2d, labels_subset, indices_used).
    """
    N = embeddings.shape[0]
    if subsample is not None and N > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, size=subsample, replace=False)
        X = embeddings[idx]
        y = labels[idx]
    else:
        idx = np.arange(N)
        X = embeddings
        y = labels

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    X_2d = tsne.fit_transform(X)
    return X_2d, y, idx


def plot_tsne(
    X_2d: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    tail_mask: np.ndarray | None = None,
    title: str = "t-SNE of Embeddings (colored by super-class)",
) -> None:
    """Scatter plot of t-SNE with points colored by label; optional star markers for tail samples."""
    plt.figure(figsize=(10, 8))
    cmap = "tab20" if np.unique(labels).size <= 20 else "viridis"

    if tail_mask is None:
        tail_mask = np.zeros((labels.shape[0],), dtype=bool)
    else:
        tail_mask = np.asarray(tail_mask, dtype=bool).flatten()
        if tail_mask.shape[0] != labels.shape[0]:
            raise ValueError("tail_mask length must match labels length for t-SNE plotting.")

    normal_mask = ~tail_mask
    scatter_base = None

    if normal_mask.any():
        scatter_base = plt.scatter(
            X_2d[normal_mask, 0],
            X_2d[normal_mask, 1],
            c=labels[normal_mask],
            s=8,
            alpha=0.55,
            marker="o",
            cmap=cmap,
        )

    if tail_mask.any():
        scatter_tail = plt.scatter(
            X_2d[tail_mask, 0],
            X_2d[tail_mask, 1],
            c=labels[tail_mask],
            s=70,
            alpha=0.95,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            cmap=cmap,
            label="Tail samples",
        )
        if scatter_base is None:
            scatter_base = scatter_tail
        plt.legend(loc="best")

    if scatter_base is not None:
        plt.colorbar(scatter_base, label="Super-class ID")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def load_image_paths_file(path: Path) -> list[str]:
    """
    Load image paths aligned with embeddings:
    - .npy/.npz containing 1D string array
    - .txt with one path per line
    """
    path = path.resolve()
    if path.suffix.lower() in {".npy", ".npz"}:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            if len(arr.files) != 1:
                raise ValueError(f"Expected exactly one array in npz: {path}")
            key = arr.files[0]
            arr = arr[key]
        arr = np.asarray(arr).flatten()
        return [str(x) for x in arr.tolist()]

    image_paths = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip()
            if p:
                image_paths.append(p)
    return image_paths


def load_image_paths_from_dataset(dataset_file: Path, base_image_dir: Path | None) -> list[str]:
    """Load image paths from SOP-format dataset txt in-order."""
    image_paths: list[str] = []
    with open(dataset_file, "r") as f:
        _header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            rel_path = " ".join(parts[3:])
            full_path = (base_image_dir / rel_path) if base_image_dir else Path(rel_path)
            if full_path.exists():
                image_paths.append(str(full_path))
    return image_paths


def compute_class_centers(emb_norm: np.ndarray, labels: np.ndarray):
    """Return (unique_classes, class_centers, own_class_similarity_per_sample)."""
    classes = np.unique(labels)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    D = emb_norm.shape[1]
    centers = np.zeros((classes.shape[0], D), dtype=np.float32)
    own_class_sim = np.zeros((emb_norm.shape[0],), dtype=np.float32)

    for c in classes:
        idx = np.where(labels == c)[0]
        class_emb = emb_norm[idx]
        center = class_emb.mean(axis=0)
        center = center / max(np.linalg.norm(center), 1e-12)
        ci = class_to_idx[c]
        centers[ci] = center.astype(np.float32)
        own_class_sim[idx] = class_emb @ centers[ci]

    return classes, centers, own_class_sim


def get_tail_sample_indices(
    labels: np.ndarray,
    own_class_sim: np.ndarray,
    classes: np.ndarray,
    tail_samples_per_class: int,
) -> np.ndarray:
    """Return unique global indices for per-class tail samples (lowest own-center similarity)."""
    picked = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        scores = own_class_sim[idx]
        n_pick = min(int(tail_samples_per_class), idx.size)
        order = np.argsort(scores)[:n_pick]
        picked.extend(idx[order].tolist())
    if not picked:
        return np.array([], dtype=np.int64)
    return np.unique(np.asarray(picked, dtype=np.int64))


def save_intra_class_stats(
    labels: np.ndarray,
    own_class_sim: np.ndarray,
    classes: np.ndarray,
    out_path: Path,
) -> None:
    """Save per-class intra-class similarity stats (using similarity to class center)."""
    rows = []
    for c in classes:
        idx = np.where(labels == c)[0]
        sims = own_class_sim[idx]
        rows.append(
            {
                "class_id": int(c),
                "count": int(idx.shape[0]),
                "mean_own_center_sim": float(np.mean(sims)),
                "std_own_center_sim": float(np.std(sims)),
                "min_own_center_sim": float(np.min(sims)),
                "max_own_center_sim": float(np.max(sims)),
            }
        )
    rows.sort(key=lambda x: x["class_id"])
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "count",
                "mean_own_center_sim",
                "std_own_center_sim",
                "min_own_center_sim",
                "max_own_center_sim",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_path}")


def _safe_open_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color="black")


def _format_path_for_overlay(path: str, width: int = 55) -> str:
    """Wrap long paths to keep overlay text readable."""
    return "\n".join(textwrap.wrap(path, width=width)) if path else "N/A"


def generate_tail_sample_analysis(
    emb_norm: np.ndarray,
    labels: np.ndarray,
    image_paths: list[str],
    classes: np.ndarray,
    class_centers: np.ndarray,
    own_class_sim: np.ndarray,
    out_dir: Path,
    tail_samples_per_class: int = 20,
) -> None:
    """
    For each class:
    - select tail samples (lowest similarity to own class center)
    - create 3-panel visualization per tail sample
    - save a summary csv of all tail samples
    """
    if len(image_paths) != emb_norm.shape[0]:
        raise ValueError(
            f"image_paths length mismatch: paths={len(image_paths)}, embeddings={emb_norm.shape[0]}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    class_to_idx = {int(c): i for i, c in enumerate(classes)}

    for c in tqdm(classes, desc="Tail analysis (per class)"):
        c_int = int(c)
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue

        class_scores = own_class_sim[idx]
        order = np.argsort(class_scores)  # ascending => tail first
        n_pick = min(tail_samples_per_class, idx.size)
        tail_indices = idx[order[:n_pick]]

        class_dir = out_dir / f"class_{c_int}"
        class_dir.mkdir(parents=True, exist_ok=True)

        for rank_in_tail, q_idx in enumerate(tail_indices, start=1):
            q_emb = emb_norm[q_idx]
            sim_to_all = q_emb @ emb_norm.T
            sim_to_all[q_idx] = -np.inf
            nn_idx = int(np.argmax(sim_to_all))
            nn_sim = float(sim_to_all[nn_idx])

            center_sims = q_emb @ class_centers.T
            pred_center_class = int(classes[int(np.argmax(center_sims))])
            own_center = float(own_class_sim[q_idx])

            q_label = int(labels[q_idx])
            nn_label = int(labels[nn_idx])

            q_img = _safe_open_image(image_paths[q_idx])
            nn_img = _safe_open_image(image_paths[nn_idx])

            fig, axes = plt.subplots(
                1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [1.0, 1.0, 1.6]}
            )

            # Left: query (tail sample)
            axes[0].imshow(q_img)
            axes[0].axis("off")
            axes[0].set_title("Tail Sample")
            axes[0].text(
                0.02,
                0.02,
                (
                    f"class={q_label}\n"
                    f"nearest-sim={nn_sim:.4f}\n"
                    f"own-center-sim={own_center:.4f}\n"
                    f"path:\n{_format_path_for_overlay(image_paths[q_idx])}"
                ),
                transform=axes[0].transAxes,
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.65, pad=6),
            )

            # Middle: nearest neighbor
            axes[1].imshow(nn_img)
            axes[1].axis("off")
            axes[1].set_title("Nearest Neighbor")
            axes[1].text(
                0.02,
                0.02,
                (
                    f"class={nn_label}\n"
                    f"sim-to-tail={nn_sim:.4f}\n"
                    f"path:\n{_format_path_for_overlay(image_paths[nn_idx])}"
                ),
                transform=axes[1].transAxes,
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.65, pad=6),
            )

            # Right: similarity vs class centers
            order_desc = np.argsort(center_sims)[::-1]
            sorted_scores = center_sims[order_desc]
            sorted_classes = classes[order_desc]
            y_pos = np.arange(sorted_scores.shape[0])
            bar_colors = np.array(["#5f6c7b"] * sorted_scores.shape[0], dtype=object)
            bar_colors[np.where(sorted_classes == q_label)[0]] = "#2ca02c"
            bar_colors[np.where(sorted_classes == pred_center_class)[0]] = "#d62728"
            axes[2].barh(y_pos, sorted_scores, color=bar_colors)
            axes[2].invert_yaxis()
            axes[2].set_xlabel("Cosine similarity")
            axes[2].set_title("Similarity to Each Class Center")
            if sorted_scores.shape[0] <= 60:
                axes[2].set_yticks(y_pos)
                axes[2].set_yticklabels([str(int(x)) for x in sorted_classes], fontsize=8)
                axes[2].set_ylabel("Class ID")
            else:
                axes[2].set_yticks([])
            axes[2].axvline(x=0.0, color="black", linewidth=0.8, alpha=0.5)

            fig.suptitle(
                f"class={q_label} tail-rank={rank_in_tail} idx={q_idx} "
                f"(pred-center-class={pred_center_class})",
                fontsize=12,
            )
            fig.tight_layout()

            out_path = class_dir / f"tail_{rank_in_tail:02d}_idx_{q_idx}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            summary_rows.append(
                {
                    "class_id": q_label,
                    "tail_rank_within_class": rank_in_tail,
                    "sample_index": int(q_idx),
                    "sample_image_path": image_paths[q_idx],
                    "own_center_similarity": own_center,
                    "nearest_index": nn_idx,
                    "nearest_image_path": image_paths[nn_idx],
                    "nearest_class_id": nn_label,
                    "nearest_similarity": nn_sim,
                    "pred_center_class": pred_center_class,
                    "figure_path": str(out_path),
                }
            )

    summary_path = out_dir / "tail_samples_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "tail_rank_within_class",
                "sample_index",
                "sample_image_path",
                "own_center_similarity",
                "nearest_index",
                "nearest_image_path",
                "nearest_class_id",
                "nearest_similarity",
                "pred_center_class",
                "figure_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inference: recall/precision@K, match/non-match distribution, t-SNE, "
            "and tail-sample class-cluster error analysis"
        )
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Path to embeddings .npy file",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to super-class IDs .npy file (same order as embeddings)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./inference_output"),
        help="Directory to save plots and optional recall summary",
    )
    parser.add_argument(
        "--rank-k",
        type=int,
        nargs="+",
        default=[1, 5, 10, 100, 1000],
        help="Recall@k values to compute",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=2000,
        help="Block size for pairwise similarity (memory)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for recall@k computation",
    )
    parser.add_argument(
        "--tsne-subsample",
        type=int,
        default=5000,
        help="Max number of points for t-SNE (0 = use all; can be slow)",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, mps, or cpu (default: auto)",
    )
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Skip t-SNE (slow for large N)",
    )
    parser.add_argument(
        "--skip-distribution",
        action="store_true",
        help="Skip match/non-match distribution computation (can be slow)",
    )
    parser.add_argument(
        "--skip-tail-analysis",
        action="store_true",
        help="Skip per-class tail-sample error analysis",
    )
    parser.add_argument(
        "--tail-samples-per-class",
        type=int,
        default=20,
        help="Number of tail samples per class for error analysis",
    )
    parser.add_argument(
        "--image-paths",
        type=Path,
        default=None,
        help=(
            "Optional paths file aligned to embeddings (.npy/.npz/.txt). "
            "Used for tail-sample image visualizations."
        ),
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help=(
            "Optional SOP-format dataset txt (image_id class_id super_class_id path). "
            "Used to build image paths in-order for tail analysis."
        ),
    )
    parser.add_argument(
        "--base-image-dir",
        type=Path,
        default=None,
        help="Base directory for relative paths in --dataset-file",
    )
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    args.embeddings = args.embeddings.resolve()
    args.labels = args.labels.resolve()
    if not args.embeddings.is_file():
        print(f"Error: embeddings file not found: {args.embeddings}", file=sys.stderr)
        sys.exit(1)
    if not args.labels.is_file():
        print(f"Error: labels file not found: {args.labels}", file=sys.stderr)
        sys.exit(1)

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings and labels...")
    embeddings, labels = load_data(args.embeddings, args.labels)
    N, D = embeddings.shape
    print(f"Embeddings shape: ({N}, {D})")
    print(f"Labels shape: {labels.shape}")

    precompute_tail = (not args.skip_tsne) or (not args.skip_tail_analysis)
    emb_norm = None
    classes = None
    class_centers = None
    own_class_sim = None
    tail_indices_global = np.array([], dtype=np.int64)
    if precompute_tail:
        emb_norm = _normalize_embeddings(embeddings)
        classes, class_centers, own_class_sim = compute_class_centers(emb_norm, labels)
        tail_indices_global = get_tail_sample_indices(
            labels=labels,
            own_class_sim=own_class_sim,
            classes=classes,
            tail_samples_per_class=args.tail_samples_per_class,
        )

    # ---- Rank-X recall / precision ----
    print("\nComputing recall@k and precision@k...")
    recall, precision, clipped_k = compute_retrieval_metrics_at_k(
        embeddings, labels, args.rank_k, device, batch_size=args.batch_size
    )
    if clipped_k:
        print(
            f"Warning: requested k values {clipped_k} exceed N-1={N-1}; "
            f"they were clipped to {N-1}."
        )
    for k in args.rank_k:
        print(f"  Recall@{k}: {recall[k]:.4f}")
        print(f"  Precision@{k}: {precision[k]:.4f}")
    recall_path = args.output_dir / "recall_at_k.txt"
    precision_path = args.output_dir / "precision_at_k.txt"
    with open(recall_path, "w") as f:
        for k in args.rank_k:
            f.write(f"Recall@{k}: {recall[k]:.4f}\n")
    with open(precision_path, "w") as f:
        for k in args.rank_k:
            f.write(f"Precision@{k}: {precision[k]:.4f}\n")
    print(f"Saved: {recall_path}")
    print(f"Saved: {precision_path}")

    # ---- Match vs non-match distribution ----
    if not args.skip_distribution:
        print("\nComputing match vs non-match similarity distribution...")
        match_sims, nonmatch_sims = compute_match_nonmatch_distribution(
            embeddings, labels, device, block_size=args.block_size
        )
        plot_match_nonmatch_distribution(
            match_sims,
            nonmatch_sims,
            args.output_dir / "match_vs_nonmatch_distribution.png",
        )
    else:
        print("\nSkipping match/non-match distribution (--skip-distribution).")

    # ---- t-SNE ----
    if not args.skip_tsne:
        subsample = None if args.tsne_subsample <= 0 else args.tsne_subsample
        n_tsne = min(N, subsample) if subsample else N
        print(f"\nRunning t-SNE (n={n_tsne})...")
        X_2d, y_sub, idx_used = run_tsne(
            embeddings,
            labels,
            perplexity=args.tsne_perplexity,
            subsample=subsample,
        )
        tail_mask_sub = np.isin(idx_used, tail_indices_global)
        print(f"t-SNE tail markers: {int(tail_mask_sub.sum())}/{len(idx_used)} points")
        plot_tsne(
            X_2d,
            y_sub,
            args.output_dir / "tsne_embeddings.png",
            tail_mask=tail_mask_sub,
            title="t-SNE of Embeddings (tail samples shown as stars)",
        )
    else:
        print("\nSkipping t-SNE (--skip-tsne).")

    # ---- Per-class tail-sample error analysis ----
    if not args.skip_tail_analysis:
        if args.tail_samples_per_class <= 0:
            raise ValueError("--tail-samples-per-class must be > 0")

        image_paths = None
        if args.image_paths is not None:
            image_paths = load_image_paths_file(args.image_paths)
            print(f"Loaded image paths from: {args.image_paths}")
        elif args.dataset_file is not None:
            base_dir = args.base_image_dir.resolve() if args.base_image_dir else None
            image_paths = load_image_paths_from_dataset(args.dataset_file.resolve(), base_dir)
            print(f"Loaded image paths from dataset file: {args.dataset_file}")

        if image_paths is None:
            print(
                "\nSkipping tail analysis: no image paths source was provided. "
                "Use --image-paths OR --dataset-file [--base-image-dir]."
            )
        else:
            print("\nComputing class centers and per-class tail sample analysis...")
            if emb_norm is None or classes is None or class_centers is None or own_class_sim is None:
                emb_norm = _normalize_embeddings(embeddings)
                classes, class_centers, own_class_sim = compute_class_centers(emb_norm, labels)

            intra_class_stats_path = args.output_dir / "intra_class_similarity_stats.csv"
            save_intra_class_stats(labels, own_class_sim, classes, intra_class_stats_path)

            tail_out_dir = args.output_dir / "tail_analysis"
            generate_tail_sample_analysis(
                emb_norm=emb_norm,
                labels=labels,
                image_paths=image_paths,
                classes=classes,
                class_centers=class_centers,
                own_class_sim=own_class_sim,
                out_dir=tail_out_dir,
                tail_samples_per_class=args.tail_samples_per_class,
            )
    else:
        print("\nSkipping tail analysis (--skip-tail-analysis).")

    print("\nDone.")


if __name__ == "__main__":
    main()
