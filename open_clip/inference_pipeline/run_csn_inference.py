#!/usr/bin/env python3
"""CSN inference: recall/precision, optional distributions, t-SNE, and tail-sample analysis.

Supports:
- Single-space mode via --embeddings + --labels
- Multi-space CSN mode via --prefix-metadata from generate_csn_embeddings.py
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
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
        arr = arr[arr.files[0]]
    return np.asarray(arr)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)


def compute_bits_left_stats(
    embeddings: np.ndarray,
    eps_list: tuple[float, ...] = (1e-2, 1e-3, 1e-4),
) -> dict[str, float | int | dict[str, float | int]]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [N, D]")

    abs_emb = np.abs(embeddings)
    l2_norms = np.linalg.norm(embeddings, axis=1)

    per_sample_active: dict[str, dict[str, float | int]] = {}
    for eps in eps_list:
        counts = np.sum(abs_emb > eps, axis=1).astype(np.int64)
        key = f"eps_{eps:g}"
        per_sample_active[key] = {
            "mean": float(np.mean(counts)),
            "median": float(np.median(counts)),
            "min": int(np.min(counts)),
            "max": int(np.max(counts)),
        }

    dim_std = np.std(embeddings, axis=0)
    globally_active_dims = int(np.sum(dim_std > 1e-6))

    return {
        "dim_total": int(embeddings.shape[1]),
        "embedding_l2_norm": {
            "mean": float(np.mean(l2_norms)),
            "std": float(np.std(l2_norms)),
            "min": float(np.min(l2_norms)),
            "max": float(np.max(l2_norms)),
        },
        "per_sample_active_dims": per_sample_active,
        "globally_active_dims_std_gt_1e-6": globally_active_dims,
        "globally_active_frac_std_gt_1e-6": float(globally_active_dims / max(embeddings.shape[1], 1)),
    }


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

    requested_k = sorted(set(int(k) for k in k_list if int(k) > 0))
    if not requested_k:
        raise ValueError("k_list must include at least one positive integer")

    max_allowed_k = n - 1
    max_k = max(min(k, max_allowed_k) for k in requested_k)

    recall_hits = {k: 0 for k in requested_k}
    precision_sum = {k: 0.0 for k in requested_k}

    for i in tqdm(range(0, n, batch_size), desc="Recall/Precision@k"):
        end = min(i + batch_size, n)
        query = emb[i:end]
        query_labels = labels_t[i:end]
        sim = query @ emb.T
        for j in range(sim.shape[0]):
            sim[j, i + j] = float("-inf")
        topk_idx = torch.topk(sim, max_k, dim=1).indices
        retrieved_labels = labels_t[topk_idx]
        same = retrieved_labels == query_labels.unsqueeze(1)

        for k_req in requested_k:
            kk = min(k_req, max_allowed_k)
            top_same = same[:, :kk]
            recall_hits[k_req] += int(top_same.any(dim=1).sum().item())
            precision_sum[k_req] += float(top_same.float().mean(dim=1).sum().item())

    recall = {k: recall_hits[k] / n for k in requested_k}
    precision = {k: precision_sum[k] / n for k in requested_k}
    clipped_k = sorted(k for k in requested_k if k > max_allowed_k)
    return recall, precision, clipped_k


def compute_match_nonmatch_distribution(
    embeddings: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    block_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    emb = torch.from_numpy(embeddings).float().to(device)
    emb = torch.nn.functional.normalize(emb, dim=1)
    n = emb.shape[0]
    cosine_vals = []
    match_flags = []

    for i in tqdm(range(0, n, block_size), desc="Match/non-match similarity"):
        end_i = min(i + block_size, n)
        block_i = emb[i:end_i]
        for j in range(i, n, block_size):
            end_j = min(j + block_size, n)
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


def plot_match_nonmatch_distribution(match_sims: np.ndarray, nonmatch_sims: np.ndarray, out_path: Path, bins: int = 50) -> None:
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
    n = embeddings.shape[0]
    if subsample is not None and n > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=subsample, replace=False)
        x = embeddings[idx]
        y = labels[idx]
    else:
        idx = np.arange(n)
        x = embeddings
        y = labels

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    x_2d = tsne.fit_transform(x)
    return x_2d, y, idx


def plot_tsne(
    x_2d: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    tail_mask: np.ndarray | None = None,
    center_labels: np.ndarray | None = None,
    center_label_name: str = "Center IDs",
    contour_labels: np.ndarray | None = None,
    contour_label_name: str = "Superclass 90% contours",
    title: str = "t-SNE of Embeddings",
) -> None:
    plt.figure(figsize=(10, 8))
    cmap = "tab20" if np.unique(labels).size <= 20 else "viridis"

    if tail_mask is None:
        tail_mask = np.zeros((labels.shape[0],), dtype=bool)
    else:
        tail_mask = np.asarray(tail_mask, dtype=bool).flatten()
        if tail_mask.shape[0] != labels.shape[0]:
            raise ValueError("tail_mask length must match labels length")

    normal_mask = ~tail_mask
    scatter_base = None

    if normal_mask.any():
        scatter_base = plt.scatter(
            x_2d[normal_mask, 0],
            x_2d[normal_mask, 1],
            c=labels[normal_mask],
            s=8,
            alpha=0.55,
            marker="o",
            cmap=cmap,
        )

    if tail_mask.any():
        scatter_tail = plt.scatter(
            x_2d[tail_mask, 0],
            x_2d[tail_mask, 1],
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
        plt.colorbar(scatter_base, label="Class ID")

    if center_labels is not None:
        center_labels = np.asarray(center_labels).flatten()
        if center_labels.shape[0] != labels.shape[0]:
            raise ValueError("center_labels length must match labels length")

        center_classes = np.unique(center_labels)
        centers = np.zeros((center_classes.shape[0], 2), dtype=np.float32)
        for i, c in enumerate(center_classes):
            idx = np.where(center_labels == c)[0]
            centers[i] = x_2d[idx].mean(axis=0).astype(np.float32)

        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            s=130,
            marker="X",
            c="none",
            edgecolors="black",
            linewidths=1.0,
            label=center_label_name,
            zorder=5,
        )
        for i, c in enumerate(center_classes):
            plt.annotate(
                str(int(c)),
                (centers[i, 0], centers[i, 1]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color="black",
                zorder=6,
            )

        handles, legend_labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(loc="best")

    if contour_labels is not None:
        contour_labels = np.asarray(contour_labels).flatten()
        if contour_labels.shape[0] != labels.shape[0]:
            raise ValueError("contour_labels length must match labels length")

        # 90% probability mass for 2D Gaussian => chi-square(df=2, p=0.90) ≈ 4.605
        chi2_q_90_df2 = 4.605170186
        contour_classes = np.unique(contour_labels)
        cmap_contour = plt.cm.get_cmap("tab20", max(int(contour_classes.shape[0]), 1))
        contour_count = 0
        first_label = True
        for i, c in enumerate(contour_classes):
            idx = np.where(contour_labels == c)[0]
            if idx.size < 3:
                continue

            pts = x_2d[idx]
            mu = pts.mean(axis=0)
            cov = np.cov(pts.T)
            if not np.all(np.isfinite(cov)):
                continue

            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, a_min=1e-9, a_max=None)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            width = 2.0 * np.sqrt(chi2_q_90_df2 * eigvals[0])
            height = 2.0 * np.sqrt(chi2_q_90_df2 * eigvals[1])
            angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))

            ell = Ellipse(
                xy=(float(mu[0]), float(mu[1])),
                width=float(width),
                height=float(height),
                angle=angle,
                fill=False,
                edgecolor=cmap_contour(i),
                linewidth=1.1,
                alpha=0.75,
                label=contour_label_name if first_label else None,
                zorder=4,
            )
            plt.gca().add_patch(ell)
            first_label = False
            contour_count += 1

        if contour_count > 0:
            print(f"Plotted {contour_count} superclass 90% contours.")
            handles, legend_labels = plt.gca().get_legend_handles_labels()
            if handles:
                plt.legend(loc="best")

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def compute_class_centers(emb_norm: np.ndarray, labels: np.ndarray):
    classes = np.unique(labels)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    d = emb_norm.shape[1]
    centers = np.zeros((classes.shape[0], d), dtype=np.float32)
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


def get_tail_sample_indices(labels: np.ndarray, own_class_sim: np.ndarray, classes: np.ndarray, tail_samples_per_class: int) -> np.ndarray:
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


def save_intra_class_stats(labels: np.ndarray, own_class_sim: np.ndarray, classes: np.ndarray, out_path: Path) -> None:
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
    if len(image_paths) != emb_norm.shape[0]:
        raise ValueError(
            f"image_paths length mismatch: paths={len(image_paths)}, embeddings={emb_norm.shape[0]}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for c in tqdm(classes, desc="Tail analysis (per class)"):
        c_int = int(c)
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue

        class_scores = own_class_sim[idx]
        order = np.argsort(class_scores)
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

            fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [1.0, 1.0, 1.6]})

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


def evaluate_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    tag: str,
    image_paths: list[str] | None,
    center_overlay_labels: np.ndarray | None = None,
    center_overlay_name: str = "Centers",
    contour_overlay_labels: np.ndarray | None = None,
    contour_overlay_name: str = "Superclass 90% contours",
    embedding_mode: str = "unknown",
) -> None:
    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError(f"Shape mismatch for {tag}: embeddings={embeddings.shape[0]}, labels={labels.shape[0]}")
    if center_overlay_labels is not None and center_overlay_labels.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"Shape mismatch for {tag}: center overlay labels={center_overlay_labels.shape[0]}, "
            f"embeddings={embeddings.shape[0]}"
        )
    if contour_overlay_labels is not None and contour_overlay_labels.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"Shape mismatch for {tag}: contour overlay labels={contour_overlay_labels.shape[0]}, "
            f"embeddings={embeddings.shape[0]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{tag}] Computing recall@k and precision@k...")
    recall, precision, clipped_k = compute_retrieval_metrics_at_k(
        embeddings=embeddings,
        labels=labels,
        k_list=args.rank_k,
        device=device,
        batch_size=args.batch_size,
    )

    if clipped_k:
        print(f"[{tag}] Warning: requested k {clipped_k} exceed N-1={embeddings.shape[0]-1}; clipped.")

    recall_path = output_dir / "recall_at_k.txt"
    precision_path = output_dir / "precision_at_k.txt"
    metrics_path = output_dir / "metrics.json"

    with open(recall_path, "w") as f:
        for k in args.rank_k:
            f.write(f"Recall@{k}: {recall[k]:.4f}\n")

    with open(precision_path, "w") as f:
        for k in args.rank_k:
            f.write(f"Precision@{k}: {precision[k]:.4f}\n")

    payload = {
        "tag": tag,
        "embedding_mode": embedding_mode,
        "num_samples": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "rank_k": args.rank_k,
        "recall": {str(k): float(v) for k, v in recall.items()},
        "precision": {str(k): float(v) for k, v in precision.items()},
        "clipped_k": clipped_k,
        "bits_left": compute_bits_left_stats(embeddings),
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[{tag}] Saved: {recall_path}")
    print(f"[{tag}] Saved: {precision_path}")
    print(f"[{tag}] Saved: {metrics_path}")

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

    if not args.skip_distribution:
        print(f"\n[{tag}] Computing match/non-match distribution...")
        match_sims, nonmatch_sims = compute_match_nonmatch_distribution(
            embeddings=embeddings,
            labels=labels,
            device=device,
            block_size=args.block_size,
        )
        plot_match_nonmatch_distribution(match_sims, nonmatch_sims, output_dir / "match_vs_nonmatch_distribution.png")
    else:
        print(f"\n[{tag}] Skipping match/non-match distribution (--skip-distribution).")

    if not args.skip_tsne:
        subsample = None if args.tsne_subsample <= 0 else args.tsne_subsample
        n_tsne = min(embeddings.shape[0], subsample) if subsample else embeddings.shape[0]
        print(f"\n[{tag}] Running t-SNE (n={n_tsne})...")
        x_2d, y_sub, idx_used = run_tsne(
            embeddings,
            labels,
            perplexity=args.tsne_perplexity,
            subsample=subsample,
        )
        tail_mask_sub = np.isin(idx_used, tail_indices_global)
        print(f"[{tag}] t-SNE tail markers: {int(tail_mask_sub.sum())}/{len(idx_used)} points")

        center_labels_sub = None
        if center_overlay_labels is not None:
            center_labels_sub = center_overlay_labels[idx_used]
            print(
                f"[{tag}] Overlaying {np.unique(center_labels_sub).size} {center_overlay_name.lower()} on t-SNE."
            )

        contour_labels_sub = None
        if contour_overlay_labels is not None:
            contour_labels_sub = contour_overlay_labels[idx_used]
            print(
                f"[{tag}] Overlaying {np.unique(contour_labels_sub).size} {contour_overlay_name.lower()} on t-SNE."
            )

        plot_tsne(
            x_2d,
            y_sub,
            output_dir / "tsne_embeddings.png",
            tail_mask=tail_mask_sub,
            center_labels=center_labels_sub,
            center_label_name=center_overlay_name,
            contour_labels=contour_labels_sub,
            contour_label_name=contour_overlay_name,
            title=f"t-SNE ({tag}) tail samples as stars",
        )
    else:
        print(f"\n[{tag}] Skipping t-SNE (--skip-tsne).")

    if not args.skip_tail_analysis:
        if args.tail_samples_per_class <= 0:
            raise ValueError("--tail-samples-per-class must be > 0")

        if image_paths is None:
            print(f"\n[{tag}] Skipping tail analysis: no image paths available.")
        else:
            print(f"\n[{tag}] Computing per-class tail sample analysis...")
            if emb_norm is None or classes is None or class_centers is None or own_class_sim is None:
                emb_norm = _normalize_embeddings(embeddings)
                classes, class_centers, own_class_sim = compute_class_centers(emb_norm, labels)

            intra_path = output_dir / "intra_class_similarity_stats.csv"
            save_intra_class_stats(labels, own_class_sim, classes, intra_path)

            generate_tail_sample_analysis(
                emb_norm=emb_norm,
                labels=labels,
                image_paths=image_paths,
                classes=classes,
                class_centers=class_centers,
                own_class_sim=own_class_sim,
                out_dir=output_dir / "tail_analysis",
                tail_samples_per_class=args.tail_samples_per_class,
            )
    else:
        print(f"\n[{tag}] Skipping tail analysis (--skip-tail-analysis).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSN inference with retrieval metrics and advanced analysis")

    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument(
        "--prefix-metadata",
        type=Path,
        default=None,
        help="Metadata json from generate_csn_embeddings.py; evaluates 4 spaces by default.",
    )

    parser.add_argument("--image-paths", type=Path, default=None, help="Optional .npy/.npz/.txt aligned image paths")

    parser.add_argument("--rank-k", type=int, nargs="+", default=[1, 5, 10, 100, 1000])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=2000)

    parser.add_argument("--tsne-subsample", type=int, default=5000, help="0 => all points")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)

    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--skip-distribution", action="store_true")
    parser.add_argument("--skip-tail-analysis", action="store_true")
    parser.add_argument("--tail-samples-per-class", type=int, default=20)

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
            "paths": Path(outputs["paths"]) if "paths" in outputs else None,
        }

        image_paths = None
        if args.image_paths is not None:
            image_paths = [str(x) for x in load_array(args.image_paths).flatten().tolist()] if args.image_paths.suffix.lower() in {".npy", ".npz"} else [ln.strip() for ln in args.image_paths.read_text().splitlines() if ln.strip()]
        elif paths["paths"] is not None and paths["paths"].exists():
            image_paths = [str(x) for x in load_array(paths["paths"]).flatten().tolist()]

        superclass_ids = load_array(paths["superclass_ids"]).astype(np.int64).flatten()
        category_ids = load_array(paths["category_ids"]).astype(np.int64).flatten()

        embedding_spaces = [
            ("image_super", paths["image_super"]),
            ("image_category", paths["image_category"]),
            ("text_super", paths["text_super"]),
            ("text_category", paths["text_category"]),
        ]
        label_sets = [
            ("superclass_ids", superclass_ids),
            ("category_ids", category_ids),
        ]

        for emb_tag, emb_path in embedding_spaces:
            embeddings = load_array(emb_path).astype(np.float32)
            embedding_mode = "masked" if emb_tag.endswith("category") else "unmasked"
            for label_tag, labels in label_sets:
                tag = f"{emb_tag}__eval_{label_tag}"
                center_overlay_labels = category_ids
                center_overlay_name = "Category Centers"
                contour_overlay_labels = None
                contour_overlay_name = "Superclass 90% contours"

                # For category-id evaluation views, show superclass mass regions.
                if label_tag == "category_ids":
                    contour_overlay_labels = superclass_ids

                evaluate_space(
                    embeddings=embeddings,
                    labels=labels,
                    args=args,
                    device=device,
                    output_dir=args.output_dir / tag,
                    tag=tag,
                    image_paths=image_paths,
                    center_overlay_labels=center_overlay_labels,
                    center_overlay_name=center_overlay_name,
                    contour_overlay_labels=contour_overlay_labels,
                    contour_overlay_name=contour_overlay_name,
                    embedding_mode=embedding_mode,
                )
    else:
        embeddings = load_array(args.embeddings.resolve()).astype(np.float32)
        labels = load_array(args.labels.resolve()).astype(np.int64).flatten()

        image_paths = None
        if args.image_paths is not None:
            if args.image_paths.suffix.lower() in {".npy", ".npz"}:
                image_paths = [str(x) for x in load_array(args.image_paths).flatten().tolist()]
            else:
                image_paths = [ln.strip() for ln in args.image_paths.read_text().splitlines() if ln.strip()]

        evaluate_space(
            embeddings=embeddings,
            labels=labels,
            args=args,
            device=device,
            output_dir=args.output_dir,
            tag="single",
            image_paths=image_paths,
            embedding_mode="unknown",
        )

    print("Done")


if __name__ == "__main__":
    main()
