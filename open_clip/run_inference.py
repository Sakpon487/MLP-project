#!/usr/bin/env python3
"""
Inference script: load embeddings + labels, compute rank-X recall,
match vs non-match similarity distribution, and t-SNE visualization.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def compute_recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_list: list[int],
    device: torch.device,
    batch_size: int = 512,
) -> dict[int, float]:
    """Compute recall@k for each k in k_list (retrieval by cosine similarity)."""
    emb = torch.from_numpy(embeddings).float().to(device)
    emb = torch.nn.functional.normalize(emb, dim=1)
    labels_t = torch.from_numpy(labels).to(device)
    N = emb.shape[0]
    max_k = max(k_list)
    recall_hits = {k: 0 for k in k_list}

    for i in tqdm(range(0, N, batch_size), desc="Recall@k"):
        end = min(i + batch_size, N)
        query = emb[i:end]
        query_labels = labels_t[i:end]
        sim = query @ emb.T
        # Remove self-match
        for j in range(sim.shape[0]):
            sim[j, i + j] = -1.0
        topk_idx = torch.topk(sim, max_k, dim=1).indices
        retrieved_labels = labels_t[topk_idx]
        for k in k_list:
            correct = (retrieved_labels[:, :k] == query_labels.unsqueeze(1)).any(dim=1)
            recall_hits[k] += correct.sum().item()

    return {k: recall_hits[k] / N for k in k_list}


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
    title: str = "t-SNE of Embeddings (colored by super-class)",
) -> None:
    """Scatter plot of t-SNE with points colored by label."""
    plt.figure(figsize=(10, 8))
    # Use a colormap; many classes -> scatter with c=labels
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        s=5,
        alpha=0.6,
        cmap="tab20" if np.unique(labels).size <= 20 else "viridis",
    )
    plt.colorbar(scatter, label="Super-class ID")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference: rank-X recall, match/non-match distribution, t-SNE"
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

    # ---- Rank-X recall ----
    print("\nComputing recall@k...")
    recall = compute_recall_at_k(
        embeddings, labels, args.rank_k, device, batch_size=args.batch_size
    )
    for k in args.rank_k:
        print(f"  Recall@{k}: {recall[k]:.4f}")
    recall_path = args.output_dir / "recall_at_k.txt"
    with open(recall_path, "w") as f:
        for k in args.rank_k:
            f.write(f"Recall@{k}: {recall[k]:.4f}\n")
    print(f"Saved: {recall_path}")

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
        X_2d, y_sub, _ = run_tsne(
            embeddings,
            labels,
            perplexity=args.tsne_perplexity,
            subsample=subsample,
        )
        plot_tsne(
            X_2d,
            y_sub,
            args.output_dir / "tsne_embeddings.png",
        )
    else:
        print("\nSkipping t-SNE (--skip-tsne).")

    print("\nDone.")


if __name__ == "__main__":
    main()
