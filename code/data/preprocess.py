"""
code/data/preprocess.py
=======================
STEP 2 — Feature Extraction + Train/Val/Test Splits

Walks the raw dataset, extracts 63-dim feature vectors from MediaPipe JSON
files, and creates participant-level train / val / test splits.

No semi-supervised subsets. Every sample is labeled. All splits are fully
supervised — 100% of training data is used in training.

What this produces
------------------
  <dataset_root>/splits/
      train.txt              — training set (70%)
      val.txt                — validation set (15%)
      test.txt               — test set (15%)
      train_features.npz     — precomputed (features, labels) for fast loading
      val_features.npz
      test_features.npz

Run
---
    python code/data/preprocess.py --task hgrd
    python code/data/preprocess.py --task custom
    python code/data/preprocess.py --task both
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

from sklearn.model_selection import StratifiedShuffleSplit

from utils.config import (
    get_dataset_root, DATA_SPLITS, NUM_CLASSES, GESTURE_NAMES,
)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def _load_landmark_json(path: Path) -> np.ndarray:
    """
    Load a MediaPipe landmark JSON → 63-dim float32 vector.

    JSON format (written by collector.py):
      {"landmarks": [{"id":0,"x":0.5,"y":0.3,"z":-0.01}, ...21 entries...]}

    Returns None if file is missing, malformed, or has wrong landmark count.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        lms = data["landmarks"]
        if len(lms) != 21:
            return None
        vec = np.array(
            [[lm["x"], lm["y"], lm["z"]] for lm in lms],
            dtype=np.float32,
        ).flatten()   # (63,)
        return vec
    except Exception:
        return None


def normalise_landmarks(vec: np.ndarray) -> np.ndarray:
    """
    Make the landmark vector invariant to hand position in the frame.

    Steps:
      1. Subtract wrist (landmark 0) — centres hand at origin
      2. Divide by max absolute value — scales into [-1, 1]

    Result: the model learns hand SHAPE (relative finger positions),
    not absolute hand position on screen.

    This function is also called during inference in realtime/test/server.
    """
    pts   = vec.reshape(21, 3)
    pts   = pts - pts[0]           # translate: wrist → origin
    scale = np.max(np.abs(pts)) + 1e-8
    return (pts / scale).flatten().astype(np.float32)


# =============================================================================
# DATASET SCANNER
# =============================================================================

def _scan_dataset(root: Path) -> List[Dict]:
    """
    Scan dataset directory for all (landmark_path, gesture_id) pairs.

    Expected directory structure (created by collector.py):
      <root>/raw/<participant>/<session>/<gesture_id>/
          frame_000001.jpg
          landmarks_000001.json

    Returns list of sample dicts:
      {landmark_path, image_path, gesture_id, participant, session}
    """
    samples = []
    raw_root = root / "raw"

    # ── Standard structure from collector.py ──────────────────────────────────
    if raw_root.exists():
        for participant in sorted(raw_root.iterdir()):
            if not participant.is_dir() or participant.name.startswith("."):
                continue
            for session in sorted(participant.iterdir()):
                if not session.is_dir():
                    continue
                for gesture_dir in sorted(session.iterdir()):
                    if not gesture_dir.is_dir():
                        continue
                    try:
                        gid = int(gesture_dir.name)
                    except ValueError:
                        continue
                    if gid not in GESTURE_NAMES:
                        continue
                    for lm_path in sorted(gesture_dir.glob("landmarks_*.json")):
                        img_path = gesture_dir / lm_path.name.replace(
                            "landmarks_", "frame_").replace(".json", ".jpg")
                        samples.append({
                            "landmark_path": lm_path,
                            "image_path":    img_path if img_path.exists() else None,
                            "gesture_id":    gid,
                            "participant":   participant.name,
                            "session":       session.name,
                        })
        return samples

    # ── Legacy flat structure: <root>/<gesture_id>/*.json ─────────────────────
    for gesture_dir in sorted(root.iterdir()):
        if not gesture_dir.is_dir():
            continue
        try:
            gid = int(gesture_dir.name)
        except ValueError:
            continue
        if gid not in GESTURE_NAMES:
            continue
        for lm_path in sorted(gesture_dir.glob("*.json")):
            img_path = lm_path.with_suffix(".jpg")
            samples.append({
                "landmark_path": lm_path,
                "image_path":    img_path if img_path.exists() else None,
                "gesture_id":    gid,
                "participant":   "p001",
                "session":       "default",
            })
    return samples


# =============================================================================
# PARTICIPANT-LEVEL SPLIT
# =============================================================================

def _participant_split(
    samples: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split at participant level so no participant appears in both train and test.
    This prevents data leakage: the model cannot memorise a specific person's
    hand shape. It must generalise to new participants.

    Strategy:
      1. Get unique participants
      2. StratifiedShuffleSplit participants (stratified by dominant gesture)
      3. All samples from each participant go into one split only

    Returns (train_samples, val_samples, test_samples)
    """
    by_participant: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_participant[s["participant"]].append(s)

    participants = list(by_participant.keys())

    # Fewer than 3 participants → fall back to sample-level split
    if len(participants) < 3:
        print("  [WARN] Fewer than 3 participants — using sample-level split.")
        rng  = np.random.default_rng(42)
        idx  = rng.permutation(len(samples))
        n    = len(samples)
        n_tr = int(n * DATA_SPLITS["train_ratio"])
        n_va = int(n * DATA_SPLITS["val_ratio"])
        return (
            [samples[i] for i in idx[:n_tr]],
            [samples[i] for i in idx[n_tr:n_tr + n_va]],
            [samples[i] for i in idx[n_tr + n_va:]],
        )

    # Dominant gesture per participant for stratification
    p_labels = []
    for p in participants:
        counts = defaultdict(int)
        for s in by_participant[p]:
            counts[s["gesture_id"]] += 1
        p_labels.append(max(counts, key=counts.get))

    p_arr = np.array(participants)
    y_arr = np.array(p_labels)

    # Split off test set
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=DATA_SPLITS["test_ratio"], random_state=42)
    train_val_idx, test_idx = next(sss1.split(p_arr, y_arr))

    # Split train_val into train and val
    val_frac = DATA_SPLITS["val_ratio"] / (
        DATA_SPLITS["train_ratio"] + DATA_SPLITS["val_ratio"])
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=42)
    tr_sub, va_sub = next(
        sss2.split(p_arr[train_val_idx], y_arr[train_val_idx]))

    train_p = set(p_arr[train_val_idx][tr_sub])
    val_p   = set(p_arr[train_val_idx][va_sub])
    test_p  = set(p_arr[test_idx])

    return (
        [s for s in samples if s["participant"] in train_p],
        [s for s in samples if s["participant"] in val_p],
        [s for s in samples if s["participant"] in test_p],
    )


# =============================================================================
# WRITE / CACHE
# =============================================================================

def _write_split_file(samples: List[Dict], path: Path) -> None:
    """Write split .txt — one line per sample: <landmark_path>\t<gesture_id>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(f"{s['landmark_path']}\t{s['gesture_id']}\n")
    print(f"  Wrote {len(samples):5d} samples → {path.name}")


def _build_feature_cache(samples: List[Dict], out_path: Path) -> None:
    """
    Pre-extract and normalise all 63-dim vectors → .npz cache.
    The DataLoader reads this file directly for fast epoch iteration.

    Arrays:
      features : (N, 63)  float32 — normalised landmark vectors
      labels   : (N,)     int32   — gesture class IDs
      paths    : (N,)     str     — source landmark file paths
    """
    feats, labels_list, paths_list = [], [], []
    skipped = 0

    for s in samples:
        vec = _load_landmark_json(s["landmark_path"])
        if vec is None:
            skipped += 1
            continue
        vec = normalise_landmarks(vec)
        feats.append(vec)
        labels_list.append(s["gesture_id"])
        paths_list.append(str(s["landmark_path"]))

    if not feats:
        print(f"  [WARN] No valid landmarks found — cache not written: {out_path}")
        return

    np.savez_compressed(
        out_path,
        features=np.array(feats,       dtype=np.float32),
        labels=np.array(labels_list,   dtype=np.int32),
        paths=np.array(paths_list),
    )
    print(f"  Cache {len(feats)} samples (skipped {skipped}) → {out_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def preprocess(task: str) -> None:
    root = get_dataset_root(task)
    print(f"\n[Preprocess] Dataset: {task}  root={root}")

    samples = _scan_dataset(root)
    if not samples:
        print(f"  [ERROR] No samples found in {root}")
        print("  Run:  bash collect.sh -p p001   first")
        return

    n_participants = len({s["participant"] for s in samples})
    print(f"  Found {len(samples)} samples  |  {n_participants} participants")

    # Class distribution
    by_class = defaultdict(int)
    for s in samples:
        by_class[s["gesture_id"]] += 1
    print("  Class distribution:")
    for gid in range(NUM_CLASSES):
        cnt = by_class.get(gid, 0)
        bar = "█" * (cnt // 10)
        print(f"    [{gid:02d}] {GESTURE_NAMES[gid]:22s}: {cnt:4d}  {bar}")

    # Participant-level split
    train_s, val_s, test_s = _participant_split(samples)
    print(f"\n  Split → train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")

    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Write split files
    print()
    _write_split_file(train_s, splits_dir / "train.txt")
    _write_split_file(val_s,   splits_dir / "val.txt")
    _write_split_file(test_s,  splits_dir / "test.txt")

    # Build feature caches
    print()
    _build_feature_cache(train_s, splits_dir / "train_features.npz")
    _build_feature_cache(val_s,   splits_dir / "val_features.npz")
    _build_feature_cache(test_s,  splits_dir / "test_features.npz")

    print(f"\n[Preprocess] Done — splits saved to {splits_dir}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--task", required=True,
                    choices=["hgrd", "custom", "both"])
    args = ap.parse_args()
    tasks = ["hgrd", "custom"] if args.task == "both" else [args.task]
    for t in tasks:
        preprocess(t)
