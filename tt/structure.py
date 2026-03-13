"""
structure.py — Dataset Structure Inspector & Fixer
===================================================
Run this FIRST after downloading the Kaggle dataset.

What it does
------------
1. Inspects the downloaded folder and prints a report.
2. Tells you whether the layout is already correct.
3. If the layout is the nested Kaggle pattern
   (train/train/0/, test/test/0/) it FIXES it automatically
   by flattening + adding a validation split.

Expected OUTPUT after this script:
    structured_dataset/
        train/
            0/  *.jpg
            1/  *.jpg
            …
        val/
            0/  *.jpg
            …
        test/
            0/  *.jpg
            …

Usage
-----
    python structure.py                         # uses paths from config.py
    python structure.py --raw ./my_raw_folder   # override raw path
    python structure.py --dry-run               # just inspect, don't copy
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

# ── bring in config ──────────────────────────────────────────────────── #
sys.path.insert(0, str(Path(__file__).parent))
from config import GestureConfig

cfg = GestureConfig(mode="train")


# ════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ════════════════════════════════════════════════════════════════════════

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _count_images(folder: Path) -> int:
    return sum(
        1 for f in folder.rglob("*") if f.suffix.lower() in SUPPORTED_EXTS
    )


def _get_class_folders(split_dir: Path):
    """Return sorted list of subdirectories (class folders) inside split_dir."""
    return sorted([d for d in split_dir.iterdir() if d.is_dir()])


# ════════════════════════════════════════════════════════════════════════
# STEP 1 — INSPECT
# ════════════════════════════════════════════════════════════════════════

def inspect_dataset(raw_root: Path) -> dict:
    """
    Walk raw_root and return a summary dict describing what was found.
    Detects these patterns:

    Pattern A (CORRECT FLAT):
        raw_root/train/0/img.jpg
        raw_root/test/0/img.jpg

    Pattern B (KAGGLE NESTED — needs fixing):
        raw_root/train/train/0/img.jpg
        raw_root/test/test/0/img.jpg

    Pattern C (UNKNOWN):
        anything else
    """
    print("\n" + "=" * 60)
    print("  DATASET INSPECTOR")
    print("=" * 60)
    print(f"Root   : {raw_root}")

    if not raw_root.exists():
        print(f"\n❌  Folder not found: {raw_root}")
        print("    → Check 'raw_dataset_root' in config.py")
        return {"pattern": "missing"}

    top_level = sorted([d.name for d in raw_root.iterdir() if d.is_dir()])
    print(f"Top-level dirs: {top_level}")

    # ── Pattern B detection ─────────────────────────────────────────── #
    nested_train = raw_root / "train" / "train"
    nested_test  = raw_root / "test"  / "test"

    if nested_train.exists():
        class_dirs = _get_class_folders(nested_train)
        class_names = [d.name for d in class_dirs]
        total_train = _count_images(nested_train)
        total_test  = _count_images(nested_test) if nested_test.exists() else 0

        print("\n⚠️   DETECTED: Kaggle nested pattern  (train/train/…  test/test/…)")
        print(f"    Classes found : {class_names}")
        print(f"    Train images  : {total_train}")
        print(f"    Test images   : {total_test}")

        per_class = {}
        for cd in class_dirs:
            imgs = list(cd.rglob("*"))
            imgs = [i for i in imgs if i.suffix.lower() in SUPPORTED_EXTS]
            per_class[cd.name] = len(imgs)

        print("\n    Images per class (train):")
        for cls, cnt in sorted(per_class.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            bar = "█" * (cnt // 10)
            print(f"      class {cls:>3}: {cnt:>5}  {bar}")

        return {
            "pattern": "B_nested_kaggle",
            "nested_train": nested_train,
            "nested_test" : nested_test if nested_test.exists() else None,
            "classes"     : class_names,
            "per_class"   : per_class,
            "total_train" : total_train,
            "total_test"  : total_test,
        }

    # ── Pattern A detection ─────────────────────────────────────────── #
    flat_train = raw_root / "train"
    if flat_train.exists() and any(d.is_dir() for d in flat_train.iterdir()):
        # check the subdirs are class folders, not another "train"
        sub = [d for d in flat_train.iterdir() if d.is_dir()]
        if all(d.name != "train" for d in sub):
            total = _count_images(flat_train)
            print("\n✅  DETECTED: Flat / already-structured pattern")
            print(f"    Classes: {[d.name for d in sub]}")
            print(f"    Total images in train: {total}")
            return {"pattern": "A_flat_correct", "train_dir": flat_train}

    print("\n❓  DETECTED: Unknown / unsupported pattern")
    print("    Please restructure manually or update this script.")
    return {"pattern": "C_unknown"}


# ════════════════════════════════════════════════════════════════════════
# STEP 2 — FIX  (Pattern B → structured_dataset/)
# ════════════════════════════════════════════════════════════════════════

def fix_dataset(info: dict, structured_root: Path, val_split: float, seed: int, dry_run: bool):
    """
    Copy (not move) files from the nested Kaggle layout into a clean
    structured_dataset/ tree with train / val / test splits.
    """
    if info["pattern"] == "A_flat_correct":
        print("\n✅  Dataset is already correctly structured — nothing to fix.")
        print(f"    Make sure config.py train_dir points to: {info['train_dir']}")
        return

    if info["pattern"] != "B_nested_kaggle":
        print("\n⛔  Cannot auto-fix unknown pattern. Exiting.")
        return

    print("\n" + "=" * 60)
    print("  FIXING DATASET STRUCTURE")
    print("=" * 60)
    print(f"Destination : {structured_root}")
    print(f"Val split   : {val_split * 100:.0f}% of train")
    print(f"Seed        : {seed}")
    if dry_run:
        print("🔍  DRY-RUN — no files will be copied")
    print()

    random.seed(seed)
    nested_train: Path = info["nested_train"]
    nested_test : Path = info["nested_test"]

    stats = defaultdict(int)

    # ── Train & Val ─────────────────────────────────────────────────── #
    for class_dir in sorted(nested_train.iterdir()):
        if not class_dir.is_dir():
            continue
        cls = class_dir.name
        images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        ]
        random.shuffle(images)

        n_val   = max(1, int(len(images) * val_split))
        val_imgs   = images[:n_val]
        train_imgs = images[n_val:]

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest_dir = structured_root / split_name / cls
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                dest = dest_dir / img.name
                if not dry_run:
                    shutil.copy2(img, dest)
                stats[split_name] += 1

        print(f"  class {cls:>3}: {len(train_imgs):>5} train  |  {n_val:>4} val")

    # ── Test ────────────────────────────────────────────────────────── #
    if nested_test:
        for class_dir in sorted(nested_test.iterdir()):
            if not class_dir.is_dir():
                continue
            cls = class_dir.name
            images = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
            ]
            dest_dir = structured_root / "test" / cls
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                dest = dest_dir / img.name
                if not dry_run:
                    shutil.copy2(img, dest)
                stats["test"] += 1

    print()
    print("Summary:")
    for split, count in stats.items():
        print(f"  {split:>6}: {count} images")

    if not dry_run:
        print(f"\n✅  Structured dataset saved to: {structured_root}")
        print("    Update config.py if your paths differ.")
    else:
        print("\n🔍  Dry-run complete — no files were written.")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Dataset Structuring Tool")
    parser.add_argument("--raw",     default=cfg.raw_dataset_root,
                        help="Path to the raw/downloaded dataset folder")
    parser.add_argument("--out",     default=cfg.structured_dir,
                        help="Where to save the structured dataset")
    parser.add_argument("--val",     type=float, default=cfg.val_split,
                        help="Validation split fraction (default: 0.15)")
    parser.add_argument("--seed",    type=int,   default=cfg.random_seed)
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect only — do not copy any files")
    args = parser.parse_args()

    raw_root       = Path(args.raw)
    structured_root = Path(args.out)

    info = inspect_dataset(raw_root)
    fix_dataset(info, structured_root, args.val, args.seed, args.dry_run)


if __name__ == "__main__":
    main()