from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class CSNRecord:
    image_path: str
    clip_text: str
    superclass_id: int
    category_id: int


def _resolve_image_path(raw_path: str, base_image_dir: Path | None) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    if base_image_dir is not None:
        return base_image_dir / raw_path
    return p


def load_csn_records(csv_file: str | Path, base_image_dir: str | Path | None = None) -> tuple[list[CSNRecord], dict[str, int]]:
    csv_file = Path(csv_file)
    base_dir = Path(base_image_dir) if base_image_dir else None

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        required_any_text = ("clip_text" in cols) or ("clip_txt" in cols)
        required = {"image_path", "superclass_id", "category_id"}
        missing = sorted(required - cols)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        if not required_any_text:
            raise ValueError("CSV must contain 'clip_text' or 'clip_txt'")

        text_col = "clip_text" if "clip_text" in cols else "clip_txt"

        records: list[CSNRecord] = []
        stats = {
            "rows_total": 0,
            "rows_missing_image": 0,
            "rows_bad_label": 0,
            "rows_kept": 0,
        }

        for row in reader:
            stats["rows_total"] += 1
            try:
                image_path = str(_resolve_image_path(str(row["image_path"]), base_dir))
                superclass_id = int(row["superclass_id"])
                category_id = int(row["category_id"])
            except Exception:
                stats["rows_bad_label"] += 1
                continue

            if not Path(image_path).exists():
                stats["rows_missing_image"] += 1
                continue

            clip_text = str(row.get(text_col, "")).strip()
            if not clip_text:
                clip_text = ""

            records.append(
                CSNRecord(
                    image_path=image_path,
                    clip_text=clip_text,
                    superclass_id=superclass_id,
                    category_id=category_id,
                )
            )

        stats["rows_kept"] = len(records)

    if not records:
        raise ValueError("No valid records after filtering missing/bad rows.")

    return records, stats


def create_stratified_split_indices(
    records: list[CSNRecord],
    seed: int,
    train_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")

    by_super: dict[int, list[int]] = {}
    for idx, r in enumerate(records):
        by_super.setdefault(r.superclass_id, []).append(idx)

    rng = random.Random(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    singleton_supers = 0

    for super_id, idxs in sorted(by_super.items(), key=lambda x: x[0]):
        idxs_local = list(idxs)
        rng.shuffle(idxs_local)

        if len(idxs_local) == 1:
            train_idx.extend(idxs_local)
            singleton_supers += 1
            continue

        n_train = int(round(len(idxs_local) * train_ratio))
        n_train = max(1, min(len(idxs_local) - 1, n_train))

        train_idx.extend(idxs_local[:n_train])
        test_idx.extend(idxs_local[n_train:])

    train_idx = np.array(sorted(train_idx), dtype=np.int64)
    test_idx = np.array(sorted(test_idx), dtype=np.int64)

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            f"Empty split generated (train={train_idx.size}, test={test_idx.size}). Check dataset distribution."
        )

    meta = {
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "num_records": int(len(records)),
        "num_train": int(train_idx.size),
        "num_test": int(test_idx.size),
        "num_superclasses": int(len(by_super)),
        "singleton_superclasses": int(singleton_supers),
    }
    return train_idx, test_idx, meta


def create_or_load_split(
    records: list[CSNRecord],
    split_dir: str | Path,
    seed: int,
    force_resplit: bool = False,
    train_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train_indices.npy"
    test_path = split_dir / "test_indices.npy"
    meta_path = split_dir / "split_metadata.json"

    if train_path.exists() and test_path.exists() and meta_path.exists() and not force_resplit:
        train_idx = np.load(train_path)
        test_idx = np.load(test_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)

        max_idx = len(records) - 1
        if train_idx.size == 0 or test_idx.size == 0:
            raise ValueError("Existing split files contain empty indices.")
        if int(train_idx.max()) > max_idx or int(test_idx.max()) > max_idx:
            raise ValueError("Existing split indices exceed current record count. Use --force-resplit.")

        return train_idx.astype(np.int64), test_idx.astype(np.int64), meta

    train_idx, test_idx, meta = create_stratified_split_indices(records, seed=seed, train_ratio=train_ratio)
    np.save(train_path, train_idx)
    np.save(test_path, test_idx)

    meta = {
        **meta,
        "train_indices_path": str(train_path),
        "test_indices_path": str(test_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return train_idx, test_idx, meta


class CSNMultiViewDataset(Dataset):
    """Returns multi-view image/text tuples for CSN training.

    Output dict keys:
      image_view1/2/3, text_view1/2/3, label_view1_2 (super), label_view1_3 (category), path_view1
    """

    def __init__(
        self,
        records: list[CSNRecord],
        indices: np.ndarray,
        image_transform,
        text_tokenize_fn,
        seed: int = 0,
    ):
        self.records = records
        self.indices = np.asarray(indices, dtype=np.int64)
        self.image_transform = image_transform
        self.text_tokenize_fn = text_tokenize_fn
        self.rng = random.Random(seed)

        self.super_to_local: dict[int, list[int]] = {}
        self.cat_to_local: dict[int, list[int]] = {}
        for local_pos, global_idx in enumerate(self.indices.tolist()):
            r = self.records[int(global_idx)]
            self.super_to_local.setdefault(r.superclass_id, []).append(local_pos)
            self.cat_to_local.setdefault(r.category_id, []).append(local_pos)

        self.super_singleton_fallbacks = 0
        self.cat_singleton_fallbacks = 0

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _load_image(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color="black")
        return self.image_transform(img)

    def _sample_positive_local(self, pool: list[int], anchor_local: int, mode: str) -> int:
        candidates = [i for i in pool if i != anchor_local]
        if not candidates:
            if mode == "super":
                self.super_singleton_fallbacks += 1
            else:
                self.cat_singleton_fallbacks += 1
            return anchor_local
        return self.rng.choice(candidates)

    def __getitem__(self, local_idx: int) -> dict[str, Any]:
        anchor_global = int(self.indices[local_idx])
        anchor = self.records[anchor_global]

        super_pool = self.super_to_local[anchor.superclass_id]
        cat_pool = self.cat_to_local[anchor.category_id]

        pos_super_local = self._sample_positive_local(super_pool, local_idx, mode="super")
        pos_cat_local = self._sample_positive_local(cat_pool, local_idx, mode="cat")

        super_global = int(self.indices[pos_super_local])
        cat_global = int(self.indices[pos_cat_local])

        rec_super = self.records[super_global]
        rec_cat = self.records[cat_global]

        image_view1 = self._load_image(anchor.image_path)
        image_view2 = self._load_image(rec_super.image_path)
        image_view3 = self._load_image(rec_cat.image_path)

        texts = [anchor.clip_text, rec_super.clip_text, rec_cat.clip_text]
        tokens = self.text_tokenize_fn(texts)
        text_view1, text_view2, text_view3 = tokens[0], tokens[1], tokens[2]

        return {
            "image_view1": image_view1,
            "image_view2": image_view2,
            "image_view3": image_view3,
            "text_view1": text_view1,
            "text_view2": text_view2,
            "text_view3": text_view3,
            "label_view1_2": int(anchor.superclass_id),
            "label_view1_3": int(anchor.category_id),
            "path_view1": anchor.image_path,
        }


def collate_csn_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    def stack_tensor(key: str):
        return torch.stack([b[key] for b in batch], dim=0)

    out = {
        "image_view1": stack_tensor("image_view1"),
        "image_view2": stack_tensor("image_view2"),
        "image_view3": stack_tensor("image_view3"),
        "text_view1": stack_tensor("text_view1"),
        "text_view2": stack_tensor("text_view2"),
        "text_view3": stack_tensor("text_view3"),
        "label_view1_2": torch.tensor([b["label_view1_2"] for b in batch], dtype=torch.long),
        "label_view1_3": torch.tensor([b["label_view1_3"] for b in batch], dtype=torch.long),
        "path_view1": [b["path_view1"] for b in batch],
    }
    return out
