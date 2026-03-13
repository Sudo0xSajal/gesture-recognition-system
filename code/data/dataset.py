"""
code/data/dataset.py
====================
PyTorch Dataset and DataLoader factory — fully supervised.

Three dataset modes matching Ma'am's three AI model types:
  "cnn"      — 224×224 RGB image tensor             → GestureCNN
  "lstm"     — (seq_len, 63) temporal sequence       → LSTMModel
  "svm"      — returns raw (features, labels) arrays → SVMClassifier (numpy)

All splits are fully supervised: train / val / test only.
No unlabeled dataset, no contrastive pairs, no semi-supervised subsets.

Usage
-----
    from data.dataset import create_dataloaders
    loaders = create_dataloaders(task="hgrd", model_type="cnn", batch_size=32)
    for images, labels in loaders["train"]:
        ...

    # SVM — returns numpy arrays directly (not DataLoaders)
    from data.dataset import load_svm_data
    X_train, y_train = load_svm_data(task="hgrd", split="train")
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils.config import (
    get_dataset_root, get_split_file,
    NUM_CLASSES, TRAINING_CONFIG, MODEL_CONFIG,
)


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def normalise_landmarks(vec: np.ndarray) -> np.ndarray:
    """Centre on wrist and scale to [-1, 1]. Same function as in preprocess.py."""
    pts   = vec.reshape(21, 3)
    pts   = pts - pts[0]
    scale = np.max(np.abs(pts)) + 1e-8
    return (pts / scale).flatten().astype(np.float32)


def _load_vec_from_json(path: Path) -> np.ndarray:
    """Load MediaPipe JSON → normalised 63-dim vector, or zeros on error."""
    try:
        with open(path) as f:
            data = json.load(f)
        lms = data["landmarks"]
        vec = np.array([[lm["x"], lm["y"], lm["z"]] for lm in lms],
                       dtype=np.float32).flatten()
        return normalise_landmarks(vec)
    except Exception:
        return np.zeros(63, dtype=np.float32)


def _read_split_file(split_file: Path) -> list:
    """Read split .txt → list of (landmark_path, gesture_id) tuples."""
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Run: python code/data/preprocess.py --task <task>"
        )
    samples = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            samples.append((Path(parts[0]), int(parts[1])))
    return samples


# =============================================================================
# IMAGE AUGMENTATION PIPELINES  (for CNN)
# =============================================================================

def _train_aug(size: int = 224) -> A.Compose:
    """Strong augmentation for training — robust to lighting and hand position."""
    return A.Compose([
        A.Resize(size, size),
        A.Rotate(limit=25, p=0.6),
        A.Affine(scale=(0.75, 1.25), translate_percent=0.15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.GaussNoise(var_limit=(5, 30), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _val_aug(size: int = 224) -> A.Compose:
    """Minimal augmentation for validation and test sets."""
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# =============================================================================
# 1. CNN DATASET  — image classification
# =============================================================================
class CNNDataset(Dataset):
    """
    Loads 224×224 RGB images for GestureCNN training.
    Uses strong augmentation on training set, minimal on val/test.

    Each sample: (image_tensor: Tensor[3,224,224], label: int)
    """

    def __init__(self, split_file: Path, augment: bool = False,
                 input_size: int = 224):
        self.samples   = _read_split_file(split_file)
        self.transform = _train_aug(input_size) if augment else _val_aug(input_size)

    def __len__(self) -> int:
        return len(self.samples)

    def get_labels(self) -> list:
        return [s[1] for s in self.samples]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        lm_path, label = self.samples[idx]

        # Derive image path from landmark path.
        # Primary convention:   landmarks_<stem>.json → frame_<stem>.jpg
        # Fallback: look for any image with the same stem (downloaded datasets
        # may not use the frame_ prefix convention).
        stem     = lm_path.stem.replace("landmarks_", "")
        parent   = lm_path.parent
        img_path = parent / f"frame_{stem}.jpg"

        if not img_path.exists():
            # Try same stem with common image extensions
            found = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                candidate = parent / (stem + ext)
                if candidate.exists():
                    found = candidate
                    break
            img_path = found or img_path   # keep original path as fallback

        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Fallback: blank image (shouldn't happen after proper collection)
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        img_t = self.transform(image=img)["image"]   # (3, H, W) float tensor
        return img_t, label


# =============================================================================
# 2. LSTM DATASET  — temporal sequences (RNN)
# =============================================================================
class LSTMDataset(Dataset):
    """
    Builds temporal sequences of consecutive landmark vectors for RNN/LSTM.

    Used for gestures that involve motion over time:
      Hand Shake (#19), Both Hands Raise (#18), Hand Wave (#3), Circle Motion (#14)

    Each sample: (sequence: Tensor[seq_len, 63], label: int)
    Sequences are built by sliding a window of size seq_len over each session's
    frames for the same gesture. Shorter sessions are zero-padded at the start.

    Parameters
    ----------
    split_file : Path
    seq_len    : int  — number of time steps (default MODEL_CONFIG['lstm_seq_len'] = 15)
    augment    : bool — apply small random jitter to landmark coordinates
    """

    def __init__(self, split_file: Path, seq_len: int = None,
                 augment: bool = False):
        self.seq_len = seq_len or MODEL_CONFIG["lstm_seq_len"]
        self.augment = augment

        raw = _read_split_file(split_file)

        # Group by (gesture_id, session) to build temporal windows
        by_session: dict = {}
        for path, gid in raw:
            key = (gid, str(path.parent))
            by_session.setdefault(key, []).append(path)

        self.sequences: list = []
        for (gid, _), paths in by_session.items():
            paths_sorted = sorted(paths)
            n = len(paths_sorted)
            if n < self.seq_len:
                # Pad by repeating first frame
                pad     = [paths_sorted[0]] * (self.seq_len - n)
                windows = [pad + paths_sorted]
            else:
                # Sliding window with 50% overlap
                step    = max(1, self.seq_len // 2)
                windows = [
                    paths_sorted[i:i + self.seq_len]
                    for i in range(0, n - self.seq_len + 1, step)
                ]
            for w in windows:
                self.sequences.append((w, gid))

    def __len__(self) -> int:
        return len(self.sequences)

    def get_labels(self) -> list:
        return [s[1] for s in self.sequences]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        paths, label = self.sequences[idx]
        seq = np.stack([_load_vec_from_json(p) for p in paths])  # (seq_len, 63)

        if self.augment:
            seq = seq + np.random.normal(0, 0.008, seq.shape).astype(np.float32)

        return torch.from_numpy(seq), label


# =============================================================================
# 3. SVM DATA LOADER  — returns numpy arrays directly (no PyTorch DataLoader)
# =============================================================================
def load_svm_data(task: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load precomputed features for SVM training / evaluation.

    Reads from the .npz cache built by preprocess.py for fast loading.
    Falls back to reading JSON files directly if cache doesn't exist.

    Parameters
    ----------
    task  : 'hgrd' | 'custom'
    split : 'train' | 'val' | 'test'

    Returns
    -------
    X : (N, 63) float32 — normalised landmark feature vectors
    y : (N,)    int32   — gesture class labels
    """
    root       = get_dataset_root(task)
    cache_path = root / "splits" / f"{split}_features.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        X    = data["features"].astype(np.float32)
        y    = data["labels"].astype(np.int32)
        print(f"[SVM data] Loaded {len(X)} samples from cache ({split})")
        return X, y

    # No cache — read from split file and JSON files directly
    print(f"[SVM data] No cache found — reading from JSON files ({split})")
    split_file = get_split_file(task, split)
    samples    = _read_split_file(split_file)
    feats, labels_list = [], []
    for path, gid in samples:
        vec = _load_vec_from_json(path)
        feats.append(vec)
        labels_list.append(gid)
    return (np.array(feats, dtype=np.float32),
            np.array(labels_list, dtype=np.int32))


# =============================================================================
# WEIGHTED RANDOM SAMPLER
# =============================================================================
def _make_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """
    Gives every gesture class equal probability of appearing in each batch.
    Without this, common gestures dominate and emergency gestures are rarely seen.
    """
    counts  = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    sample_weights = [weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights),
                                 replacement=True)


# =============================================================================
# DATALOADER FACTORY
# =============================================================================
def create_dataloaders(
    task:       str,
    model_type: str,
    batch_size: int = None,
    num_workers: int = None,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders for CNN or LSTM training.
    For SVM, use load_svm_data() directly — it returns numpy arrays.

    Parameters
    ----------
    task       : 'hgrd' | 'custom'
    model_type : 'cnn' | 'lstm'
    batch_size : overrides TRAINING_CONFIG if given
    num_workers: overrides TRAINING_CONFIG if given

    Returns
    -------
    dict with keys: "train", "val", "test"
    """
    if model_type not in ("cnn", "lstm"):
        raise ValueError(
            f"create_dataloaders() is for 'cnn' or 'lstm' only.\n"
            f"For SVM, use: load_svm_data(task, split)"
        )

    bs  = batch_size  or TRAINING_CONFIG["batch_size"]
    nw  = num_workers or TRAINING_CONFIG["num_workers"]
    pin = TRAINING_CONFIG["pin_memory"]

    def _make_ds(split: str, augment: bool) -> Dataset:
        sf = get_split_file(task, split)
        if model_type == "cnn":
            return CNNDataset(sf, augment=augment)
        else:
            return LSTMDataset(sf, augment=augment)

    train_ds = _make_ds("train", augment=True)
    val_ds   = _make_ds("val",   augment=False)
    test_ds  = _make_ds("test",  augment=False)

    train_sampler = _make_weighted_sampler(train_ds.get_labels())

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=train_sampler,
                              num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin)

    print(f"[DataLoaders] task={task}  model={model_type}")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}
