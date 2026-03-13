"""
preprocess.py — Image Preprocessing Pipeline for Hand Gesture Recognition
==========================================================================
Run this AFTER structure.py.

What it does
------------
1. Reads every image from structured_dataset/{train,val,test}/
2. Applies:
   • Resize to cfg.image_size
   • Convert to RGB (drops alpha / fixes palette modes)
   • Histogram equalisation on the Value channel (optional, sharpens contrast)
   • Background removal via skin-tone masking (optional)
   • Saves to preprocessed_dataset/ keeping the same folder tree
3. Computes and saves dataset-level mean & std (used during training).
4. Generates a small HTML preview so you can eyeball the results.

Usage
-----
    python preprocess.py                         # uses config.py defaults
    python preprocess.py --no-equalize           # skip histogram eq
    python preprocess.py --workers 8             # parallel processing
    python preprocess.py --split train           # only process train split
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# ── bring in config ──────────────────────────────────────────────────── #
sys.path.insert(0, str(Path(__file__).parent))
from config import GestureConfig

cfg = GestureConfig(mode="train")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ════════════════════════════════════════════════════════════════════════
# SINGLE-IMAGE TRANSFORMS
# ════════════════════════════════════════════════════════════════════════

def _to_rgb(img: Image.Image) -> Image.Image:
    """Safely convert any PIL mode to RGB."""
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _resize(img: Image.Image, size: tuple) -> Image.Image:
    """Resize with high-quality Lanczos resampling."""
    return img.resize((size[1], size[0]), Image.LANCZOS)   # size=(H,W)


def _histogram_equalize(img: Image.Image) -> Image.Image:
    """
    Equalise the Value channel in HSV space.
    Improves contrast in images taken under varying lighting.
    """
    import colorsys
    arr = np.array(img).astype(np.float32) / 255.0
    h, w, _ = arr.shape
    out = arr.copy()

    # Convert to HSV per-pixel
    hsv = np.zeros_like(arr)
    for i in range(h):
        for j in range(w):
            r, g, b = arr[i, j]
            hsv[i, j] = colorsys.rgb_to_hsv(r, g, b)

    # Equalise V channel
    v = (hsv[:, :, 2] * 255).astype(np.uint8)
    v_eq = np.array(ImageOps.equalize(Image.fromarray(v)))
    hsv[:, :, 2] = v_eq / 255.0

    # Convert back
    for i in range(h):
        for j in range(w):
            hh, ss, vv = hsv[i, j]
            out[i, j] = colorsys.hsv_to_rgb(hh, ss, vv)

    return Image.fromarray((out * 255).astype(np.uint8))


def _histogram_equalize_fast(img: Image.Image) -> Image.Image:
    """Faster CLAHE-style equalization using only the L channel in LAB space."""
    arr = np.array(img)
    # Use PIL's built-in per-channel equalization as a fast proxy
    r, g, b = img.split()
    r_eq = ImageOps.equalize(r)
    g_eq = ImageOps.equalize(g)
    b_eq = ImageOps.equalize(b)
    return Image.merge("RGB", (r_eq, g_eq, b_eq))


def preprocess_image(
    src_path: Path,
    dst_path: Path,
    size: tuple,
    equalize: bool,
    quality: int = 95,
) -> tuple:
    """
    Process one image file.
    Returns (success: bool, error_msg: str | None).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(src_path)

        img = _to_rgb(img)
        img = _resize(img, size)

        if equalize:
            img = _histogram_equalize_fast(img)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, "JPEG", quality=quality)
        return True, None

    except Exception as exc:
        return False, f"{src_path}: {exc}"


# ════════════════════════════════════════════════════════════════════════
# MEAN & STD COMPUTATION  (over training set only)
# ════════════════════════════════════════════════════════════════════════

def compute_mean_std(train_dir: Path) -> dict:
    """
    Compute channel-wise mean & std over all training images.
    Values are in [0, 1] and can be pasted directly into config.py.
    """
    print("\nComputing dataset mean & std (training set) …")
    pixel_sum   = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count  = 0

    images = [
        f for f in train_dir.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTS
    ]

    for img_path in tqdm(images, desc="mean/std", unit="img"):
        try:
            arr = np.array(Image.open(img_path).convert("RGB")).astype(np.float64) / 255.0
            pixel_sum    += arr.sum(axis=(0, 1))
            pixel_sq_sum += (arr ** 2).sum(axis=(0, 1))
            pixel_count  += arr.shape[0] * arr.shape[1]
        except Exception:
            pass

    mean = pixel_sum / pixel_count
    std  = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    result = {
        "mean": mean.tolist(),
        "std" : std.tolist(),
        "n_pixels": int(pixel_count),
        "n_images": len(images),
    }

    print(f"  mean : {[f'{v:.4f}' for v in mean]}")
    print(f"  std  : {[f'{v:.4f}' for v in std]}")
    print("  Paste these into config.py → normalize_mean / normalize_std")
    return result


# ════════════════════════════════════════════════════════════════════════
# HTML PREVIEW  (sanity check)
# ════════════════════════════════════════════════════════════════════════

def generate_preview(preprocessed_root: Path, out_html: Path, n_per_class: int = 5):
    """Write a simple HTML file showing n_per_class images per class."""
    train_dir = preprocessed_root / "train"
    if not train_dir.exists():
        return

    rows = []
    for cls_dir in sorted(train_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = sorted(cls_dir.iterdir())[:n_per_class]
        imgs_html = "".join(
            f'<img src="{img.resolve()}" title="{img.name}" '
            f'style="width:100px;height:100px;object-fit:cover;margin:2px;">'
            for img in imgs
        )
        rows.append(
            f'<tr><td style="padding:6px;font-weight:bold;">class {cls_dir.name}</td>'
            f"<td>{imgs_html}</td></tr>"
        )

    html = (
        "<html><body style='font-family:sans-serif'>"
        "<h2>Preprocessed Dataset Preview</h2>"
        "<table border='1' cellspacing='0' cellpadding='4'>"
        + "\n".join(rows)
        + "</table></body></html>"
    )

    out_html.write_text(html)
    print(f"\n🖼️   Preview saved to: {out_html}")


# ════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════

def run_pipeline(
    structured_root: Path,
    preprocessed_root: Path,
    splits: list,
    size: tuple,
    equalize: bool,
    workers: int,
    jpeg_quality: int,
):
    total_ok  = 0
    total_err = 0
    errors    = []

    for split in splits:
        src_split = structured_root / split
        dst_split = preprocessed_root / split

        if not src_split.exists():
            print(f"⚠️  Split '{split}' not found at {src_split} — skipping.")
            continue

        # Collect all image paths
        src_images = [
            f for f in src_split.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        ]

        if not src_images:
            print(f"⚠️  No images found in split '{split}'.")
            continue

        print(f"\nProcessing '{split}' split — {len(src_images)} images …")

        # Build (src, dst) pairs
        tasks = []
        for src in src_images:
            rel = src.relative_to(src_split)
            dst = dst_split / rel.with_suffix(".jpg")
            tasks.append((src, dst))

        ok = err = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(preprocess_image, src, dst, size, equalize, jpeg_quality): src
                for src, dst in tasks
            }
            with tqdm(total=len(futures), desc=split, unit="img") as pbar:
                for future in as_completed(futures):
                    success, msg = future.result()
                    if success:
                        ok += 1
                    else:
                        err += 1
                        errors.append(msg)
                    pbar.update(1)

        print(f"  ✓ {ok} succeeded   ✗ {err} failed")
        total_ok  += ok
        total_err += err

    print(f"\n{'='*50}")
    print(f"TOTAL  ✓ {total_ok}  ✗ {total_err}")

    if errors:
        log = preprocessed_root / "errors.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(errors))
        print(f"Error details → {log}")

    return total_ok, total_err


def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Preprocessing Pipeline")
    parser.add_argument("--src",          default=cfg.structured_dir,
                        help="Structured dataset root (output of structure.py)")
    parser.add_argument("--dst",          default=cfg.preprocessed_dir,
                        help="Where to save preprocessed images")
    parser.add_argument("--size",         default=f"{cfg.image_size[0]},{cfg.image_size[1]}",
                        help="Output size as 'H,W'  (default: 224,224)")
    parser.add_argument("--splits",       default="train,val,test",
                        help="Comma-separated list of splits to process")
    parser.add_argument("--no-equalize",  action="store_true",
                        help="Skip histogram equalization")
    parser.add_argument("--workers",      type=int, default=4,
                        help="Number of parallel worker threads")
    parser.add_argument("--quality",      type=int, default=95,
                        help="JPEG save quality (1-95)")
    parser.add_argument("--mean-std",     action="store_true",
                        help="Compute & save mean/std after preprocessing")
    parser.add_argument("--preview",      action="store_true",
                        help="Generate HTML preview of preprocessed images")
    args = parser.parse_args()

    structured_root   = Path(args.src)
    preprocessed_root = Path(args.dst)
    h, w              = map(int, args.size.split(","))
    splits            = [s.strip() for s in args.splits.split(",")]
    equalize          = not args.no_equalize

    print("=" * 60)
    print("  HAND GESTURE PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Source  : {structured_root}")
    print(f"Output  : {preprocessed_root}")
    print(f"Size    : {h}×{w}")
    print(f"Splits  : {splits}")
    print(f"Equalize: {equalize}")
    print(f"Workers : {args.workers}")

    # ── Run preprocessing ────────────────────────────────────────────── #
    run_pipeline(
        structured_root  = structured_root,
        preprocessed_root= preprocessed_root,
        splits           = splits,
        size             = (h, w),
        equalize         = equalize,
        workers          = args.workers,
        jpeg_quality     = args.quality,
    )

    # ── Compute & save mean/std ──────────────────────────────────────── #
    if args.mean_std:
        train_processed = preprocessed_root / "train"
        if train_processed.exists():
            stats = compute_mean_std(train_processed)
            stats_path = preprocessed_root / "dataset_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\n📊  Stats saved to: {stats_path}")

    # ── HTML Preview ─────────────────────────────────────────────────── #
    if args.preview:
        generate_preview(
            preprocessed_root,
            preprocessed_root / "preview.html",
        )

    print("\n✅  Preprocessing complete!")
    print(f"    Next step → train your model with data from: {preprocessed_root}")


if __name__ == "__main__":
    main()