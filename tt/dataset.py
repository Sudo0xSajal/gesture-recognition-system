"""
dataset.py — PyTorch Dataset & DataLoader factory
==================================================
Works directly with the folder structure produced by preprocess.py:

    preprocessed_dataset/
        train/
            0/  1.jpg  10.jpg  …
            1/  …
            …
        val/
            0/  …
        test/
            0/  …

Everything is read from config.py (GestureConfig).
No external split files, no MediaPipe JSON, no .npz caches needed.

Usage
-----
    from dataset import create_dataloaders

    loaders = create_dataloaders()
    for images, labels in loaders["train"]:
        # images : Tensor[B, 3, 224, 224]
        # labels : Tensor[B]   (integer class index)
        ...

    # Override batch size or get a single split:
    loaders = create_dataloaders(batch_size=64)
    val_loader = loaders["val"]

    # Quick sanity-check (run this file directly):
    python dataset.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── bring in config ──────────────────────────────────────────────────── #
sys.path.insert(0, str(Path(__file__).parent))
from config import GestureConfig

cfg = GestureConfig(mode="train")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ════════════════════════════════════════════════════════════════════════
# AUGMENTATION PIPELINES
# ════════════════════════════════════════════════════════════════════════

def _train_transforms(size: Tuple[int, int]) -> A.Compose:
    """
    Strong augmentation applied only during training.
    Helps the model generalise to different hand positions,
    lighting conditions, and backgrounds.
    """
    h, w = size
    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=25, p=0.6),
        A.Affine(scale=(0.80, 1.20), translate_percent=0.10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            p=0.2,
        ),
        A.Normalize(
            mean=cfg.normalize_mean,
            std=cfg.normalize_std,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),   # HWC → CHW, numpy → torch.Tensor
    ])


def _val_transforms(size: Tuple[int, int]) -> A.Compose:
    """
    Minimal pipeline for validation and test sets.
    Only resize + normalise — no random operations.
    """
    h, w = size
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(
            mean=cfg.normalize_mean,
            std=cfg.normalize_std,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


# ════════════════════════════════════════════════════════════════════════
# DATASET CLASS
# ════════════════════════════════════════════════════════════════════════

class GestureDataset(Dataset):
    """
    Loads hand-gesture images from a preprocessed split folder.

    Folder layout expected (produced by preprocess.py):
        <split_dir>/
            0/  1.jpg  2.jpg  …      ← class folder named with integer ID
            1/  …
            …
            9/  …

    Parameters
    ----------
    split_dir : Path
        Path to one of preprocessed_dataset/{train,val,test}/
    augment : bool
        True  → apply training augmentations (use for train split only)
        False → resize + normalise only       (use for val / test)
    image_size : tuple (H, W)
        Overrides config.image_size if provided.

    Attributes
    ----------
    class_to_idx : dict[str, int]
        Maps folder name (e.g. "0") → integer label (0-9).
    idx_to_class : dict[int, str]
        Reverse mapping — useful for human-readable predictions.
    class_names  : list[str]
        Human-readable names from config.class_names (if defined).
    """

    def __init__(
        self,
        split_dir: Path,
        augment: bool = False,
        image_size: Tuple[int, int] = None,
    ):
        split_dir = Path(split_dir)
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Run preprocess.py first."
            )

        size = image_size or cfg.image_size
        self.transform = _train_transforms(size) if augment else _val_transforms(size)

        # ── Discover class folders ───────────────────────────────────── #
        class_dirs = sorted(
            [d for d in split_dir.iterdir() if d.is_dir()],
            key=lambda d: int(d.name) if d.name.isdigit() else d.name,
        )
        if not class_dirs:
            raise ValueError(
                f"No class sub-folders found in {split_dir}.\n"
                f"Expected: {split_dir}/0/, {split_dir}/1/, …"
            )

        self.class_to_idx: Dict[str, int] = {
            d.name: i for i, d in enumerate(class_dirs)
        }
        self.idx_to_class: Dict[int, str] = {
            i: d.name for i, d in enumerate(class_dirs)
        }

        # Human-readable names (optional — falls back to folder name)
        _name_map = cfg.class_names or {}
        self.class_names = [
            _name_map.get(i, f"class_{d.name}")
            for i, d in enumerate(class_dirs)
        ]

        # ── Collect all (image_path, label) pairs ───────────────────── #
        self.samples: list = []
        for cls_dir in class_dirs:
            label = self.class_to_idx[cls_dir.name]
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in SUPPORTED_EXTS:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise ValueError(f"No images found under {split_dir}.")

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ #
    def get_labels(self) -> list:
        """Return all integer labels — used by WeightedRandomSampler."""
        return [s[1] for s in self.samples]

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # ── Load image with OpenCV (fast), fall back to PIL ─────────── #
        img = cv2.imread(str(img_path))
        if img is None:
            # cv2 failed (e.g. unicode path on Windows) — try PIL
            try:
                pil_img = Image.open(img_path).convert("RGB")
                img = np.array(pil_img)
            except Exception:
                # Last resort: blank image so the batch doesn't crash
                h, w = cfg.image_size
                img = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR → RGB

        # ── Apply transforms ─────────────────────────────────────────── #
        img_tensor = self.transform(image=img)["image"]  # Tensor[3, H, W]
        return img_tensor, label


# ════════════════════════════════════════════════════════════════════════
# WEIGHTED SAMPLER  (fixes class imbalance)
# ════════════════════════════════════════════════════════════════════════

def _make_weighted_sampler(labels: list, num_classes: int) -> WeightedRandomSampler:
    """
    Gives every class equal probability of appearing in each batch.

    Why this matters: if class 0 has 500 images and class 9 has 50,
    the model will see class 9 ~10× more often per epoch, preventing
    it from ignoring rare classes.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)   # avoid division by zero
    class_weights   = 1.0 / counts
    sample_weights  = [float(class_weights[l]) for l in labels]
    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )


# ════════════════════════════════════════════════════════════════════════
# PUBLIC API — create_dataloaders()
# ════════════════════════════════════════════════════════════════════════

def create_dataloaders(
    preprocessed_root: str  = None,
    batch_size:        int  = None,
    num_workers:       int  = 4,
    pin_memory:        bool = True,
    image_size:        Tuple[int, int] = None,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from the preprocessed dataset.

    Parameters
    ----------
    preprocessed_root : str or None
        Root folder containing train/ val/ test/ sub-folders.
        Defaults to cfg.preprocessed_dir from config.py.
    batch_size : int or None
        Overrides cfg.batch_size if given.
    num_workers : int
        Number of parallel data-loading workers. Set to 0 on Windows
        if you get 'BrokenPipeError'.
    pin_memory : bool
        True is faster when training on GPU (CUDA).
        Set False if training on CPU only.
    image_size : (H, W) tuple or None
        Overrides cfg.image_size if given.
    use_weighted_sampler : bool
        Balance class frequencies in each training batch.

    Returns
    -------
    dict with keys "train", "val", "test"
    Each value is a torch.utils.data.DataLoader.
    """
    root = Path(preprocessed_root or cfg.preprocessed_dir)
    bs   = batch_size or cfg.batch_size

    # ── Check root exists ────────────────────────────────────────────── #
    if not root.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found: {root}\n"
            f"Run: python preprocess.py"
        )

    # ── Build datasets ───────────────────────────────────────────────── #
    train_ds = GestureDataset(root / "train", augment=True,  image_size=image_size)
    val_ds   = GestureDataset(root / "val",   augment=False, image_size=image_size)
    test_ds  = GestureDataset(root / "test",  augment=False, image_size=image_size)

    # ── Sanity check: same classes across all splits ─────────────────── #
    if train_ds.class_to_idx != val_ds.class_to_idx:
        print("⚠️  WARNING: train and val have different class folders!")
    if train_ds.class_to_idx != test_ds.class_to_idx:
        print("⚠️  WARNING: train and test have different class folders!")

    # ── Train: weighted sampler so rare classes aren't ignored ───────── #
    if use_weighted_sampler:
        sampler = _make_weighted_sampler(
            train_ds.get_labels(), cfg.num_classes
        )
        train_loader = DataLoader(
            train_ds,
            batch_size  = bs,
            sampler     = sampler,       # sampler replaces shuffle=True
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = True,          # keeps batch size consistent
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size  = bs,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size  = bs,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size  = bs,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    # ── Print summary ────────────────────────────────────────────────── #
    print("\n" + "=" * 50)
    print("  DATALOADERS READY")
    print("=" * 50)
    print(f"  Root          : {root}")
    print(f"  Classes ({cfg.num_classes})  : {train_ds.class_names}")
    print(f"  Train images  : {len(train_ds)}")
    print(f"  Val   images  : {len(val_ds)}")
    print(f"  Test  images  : {len(test_ds)}")
    print(f"  Batch size    : {bs}")
    print(f"  Image size    : {image_size or cfg.image_size}")
    print(f"  Weighted samp : {use_weighted_sampler}")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print("=" * 50 + "\n")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ════════════════════════════════════════════════════════════════════════
# SANITY CHECK  — run as: python dataset.py
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running dataset sanity check …\n")

    loaders = create_dataloaders(num_workers=0)   # num_workers=0 for quick test

    for split_name, loader in loaders.items():
        images, labels = next(iter(loader))
        print(f"[{split_name}] batch shape : {images.shape}   "
              f"dtype: {images.dtype}   "
              f"label range: {labels.min().item()}–{labels.max().item()}")
        print(f"         pixel range : [{images.min():.3f}, {images.max():.3f}]")

    print("\n✅  Dataset is correctly wired to preprocess.py output.")
    print("    Next step → python train.py")