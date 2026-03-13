"""
code/data/preprocess.py
=======================
STEP 2 — Feature Extraction + Train/Val/Test Splits

Walks the raw dataset, extracts 63-dim feature vectors from MediaPipe JSON
files, and creates participant-level train / val / test splits.

Supports two dataset sources
-----------------------------
1. Custom webcam dataset (collector.py):
      <root>/raw/<participant>/<session>/<gesture_id>/
          frame_000001.jpg
          landmarks_000001.json

2. Downloaded image dataset (from internet/Kaggle):
      <root>/<gesture_id>/*.{jpg,jpeg,png}           numeric class folders
      <root>/<gesture_name>/*.{jpg,jpeg,png}          named class folders

   For downloaded datasets, MediaPipe is run on each image to extract
   hand landmarks, and landmark JSON files are saved alongside the images.
   Custom webcam collection remains fully optional.

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
    # Custom webcam dataset (collect with collect.sh first):
    python code/data/preprocess.py --task hgrd
    python code/data/preprocess.py --task custom

    # Downloaded image dataset (images organised by class folder):
    python code/data/preprocess.py --task hgrd --from-images
    python code/data/preprocess.py --task both
"""

import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Any, List, Dict, Optional, Tuple

import cv2
from sklearn.model_selection import StratifiedShuffleSplit

from utils.config import (
    get_dataset_root, DATA_SPLITS, NUM_CLASSES, GESTURE_NAMES,
)

# Supported image extensions for downloaded datasets
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =============================================================================
# MEDIAPIPE COMPATIBILITY LAYER
# =============================================================================

def _create_hands_detector() -> Any:
    """
    Create a MediaPipe hand-landmark detector that works across API versions.

    - mediapipe <0.10  : uses the legacy ``mp.solutions.hands.Hands`` API.
    - mediapipe >=0.10 : uses the new ``HandLandmarker`` tasks API.
      The hand_landmarker.task model is downloaded automatically on first use.

    Returns an opaque handle consumed only by ``_extract_landmarks_from_image``.
    """
    import mediapipe as mp

    # ── Legacy API (mediapipe < 0.10) ────────────────────────────────────────
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        return mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )

    # ── New Tasks API (mediapipe >= 0.10) ────────────────────────────────────
    import urllib.request
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = Path(__file__).parent / "hand_landmarker.task"
    if not model_path.exists():
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        print(f"  Downloading MediaPipe hand landmarker model → {model_path}")
        try:
            urllib.request.urlretrieve(url, str(model_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download MediaPipe hand landmarker model: {exc}\n"
                f"Please download it manually from:\n  {url}\n"
                f"and place it at: {model_path}"
            ) from exc

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def _close_hands_detector(hands: Any) -> None:
    """Close / release the detector created by ``_create_hands_detector``."""
    try:
        hands.close()
    except Exception:
        pass


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
# IMAGE DATASET INGESTION  (for internet-downloaded datasets)
# =============================================================================

def _name_to_gesture_id(folder_name: str) -> Optional[int]:
    """
    Map a folder name to a gesture class ID.

    Supports:
      - Numeric folder names: "0", "00", "1", etc. → gesture ID directly
      - Gesture name folders: "thumb_up", "Thumb Up", "thumbup" → matched
        case-insensitively after stripping underscores, hyphens, and spaces

    Returns None if no match is found.
    """
    # Numeric folder → direct gesture ID
    try:
        gid = int(folder_name)
        if gid in GESTURE_NAMES:
            return gid
    except ValueError:
        pass

    # Name-based matching (case-insensitive, normalise separators)
    normalized = folder_name.lower().replace("_", " ").replace("-", " ").strip()
    for gid, name in GESTURE_NAMES.items():
        if name.lower() == normalized:
            return gid

    return None


def _extract_landmarks_from_image(
    img_path: Path,
    hands: Any,
) -> Optional[List[dict]]:
    """
    Run MediaPipe Hands on a single image file.

    Parameters
    ----------
    img_path : path to the image file
    hands    : detector handle returned by ``_create_hands_detector()``

    Returns a list of 21 landmark dicts (id, x, y, z, visibility), or None if
    the image cannot be read or no hand is detected.
    Works with both the legacy (mediapipe < 0.10) and new tasks API (≥ 0.10).
    """
    import mediapipe as mp

    img = cv2.imread(str(img_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Legacy API ────────────────────────────────────────────────────────────
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        results = hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None
        hand_lms = results.multi_hand_landmarks[0]
        return [
            {
                "id":         i,
                "x":          float(lm.x),
                "y":          float(lm.y),
                "z":          float(lm.z),
                "visibility": float(getattr(lm, "visibility", 1.0)),
            }
            for i, lm in enumerate(hand_lms.landmark)
        ]

    # ── New Tasks API ─────────────────────────────────────────────────────────
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )
    result = hands.detect(mp_image)
    if not result.hand_landmarks:
        return None
    hand_lms = result.hand_landmarks[0]
    return [
        {
            "id":         i,
            "x":          float(lm.x),
            "y":          float(lm.y),
            "z":          float(lm.z),
            "visibility": float(getattr(lm, "visibility", 1.0)),
        }
        for i, lm in enumerate(hand_lms)
    ]


def ingest_image_dataset(root: Path) -> List[Dict]:
    """
    Process a downloaded image dataset: run MediaPipe on each image, save
    landmark JSON files alongside the images, and return sample dicts
    compatible with the rest of the preprocessing pipeline.

    Supported directory structures
    --------------------------------
    Flat (numeric or named class folders):
        <root>/<gesture_id>/*.{jpg,jpeg,png}
        <root>/<gesture_name>/*.{jpg,jpeg,png}

    Collector-style with images but no JSON files yet:
        <root>/raw/<participant>/<session>/<gesture_id>/*.{jpg,jpeg,png}

    Output
    ------
    For each image where a hand is detected:
      - Saves ``landmarks_<stem>.json`` alongside the image.
      - If the image is not already named ``frame_<stem>.jpg``, also creates
        a ``frame_<stem>.jpg`` copy so CNN training can locate the image.
    Returns list of sample dicts (same schema as ``_scan_dataset``).

    Skips images where no hand is detected (prints a count at the end).
    Does NOT re-process images whose landmark JSON already exists.
    The MediaPipe detector is created lazily — only if there are new images
    to process (not needed when all landmarks are already present).
    """
    print(f"\n[Ingest] Scanning images in {root}")

    # Lazy detector: created only when we actually need to process a new image
    _detector_cache: list = []   # use a list so the nested closure can rebind it

    def _get_hands() -> Any:
        if not _detector_cache:
            _detector_cache.append(_create_hands_detector())
        return _detector_cache[0]

    samples: List[Dict] = []
    processed = 0
    already_done = 0
    skipped_no_hand = 0

    def _process_gesture_dir(
        gesture_dir: Path, gid: int, participant: str, session: str
    ) -> None:
        nonlocal processed, already_done, skipped_no_hand

        for img_path in sorted(gesture_dir.iterdir()):
            if img_path.suffix.lower() not in _IMAGE_EXTS:
                continue

            stem = img_path.stem
            lm_path = gesture_dir / f"landmarks_{stem}.json"

            # Derive canonical frame path (used by CNNDataset)
            if stem.startswith("frame_"):
                frame_path = img_path  # already frame_<stem>.jpg
            else:
                frame_path = gesture_dir / f"frame_{stem}{img_path.suffix}"

            # Skip if landmark already extracted
            if lm_path.exists():
                already_done += 1
                samples.append({
                    "landmark_path": lm_path,
                    "image_path":    frame_path if frame_path.exists() else img_path,
                    "gesture_id":    gid,
                    "participant":   participant,
                    "session":       session,
                })
                continue

            # Run MediaPipe (create detector lazily on first new image)
            lm_data = _extract_landmarks_from_image(img_path, _get_hands())
            if lm_data is None:
                skipped_no_hand += 1
                continue

            # Save landmark JSON
            payload = {
                "landmarks":    lm_data,
                "gesture_id":   gid,
                "gesture_name": GESTURE_NAMES[gid],
                "participant":  participant,
                "session":      session,
                "source_image": str(img_path),
            }
            with open(lm_path, "w") as f:
                json.dump(payload, f)

            # Ensure frame_<stem>.<ext> exists for CNNDataset.
            # Symlinks are preferred (no disk duplication); fall back to a copy
            # on Windows or filesystems that don't support symlinks.
            if not frame_path.exists() and frame_path != img_path:
                try:
                    frame_path.symlink_to(img_path.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(str(img_path), str(frame_path))

            samples.append({
                "landmark_path": lm_path,
                "image_path":    frame_path if frame_path.exists() else img_path,
                "gesture_id":    gid,
                "participant":   participant,
                "session":       session,
            })
            processed += 1

    def _finish() -> List[Dict]:
        if _detector_cache:
            _close_hands_detector(_detector_cache[0])
        if unmatched_dirs:
            print(f"  [WARN] Could not map to gesture IDs: {unmatched_dirs}")
            print("  Rename folders to numeric IDs (0–24) or exact gesture names.")
            print("  Gesture names:", list(GESTURE_NAMES.values()))
        print(f"  Ingested {processed} new images  |  "
              f"Already done: {already_done}  |  "
              f"No hand detected: {skipped_no_hand}  |  "
              f"Total samples: {len(samples)}")
        return samples

    unmatched_dirs: List[str] = []

    # ── collector.py raw structure (images, no JSONs yet) ─────────────────────
    raw_root = root / "raw"
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
                    gid = _name_to_gesture_id(gesture_dir.name)
                    if gid is None:
                        continue
                    _process_gesture_dir(gesture_dir, gid,
                                         participant.name, session.name)
        if samples:
            return _finish()

    # ── Flat structure: <root>/<gesture_id_or_name>/*.jpg ─────────────────────
    for gesture_dir in sorted(root.iterdir()):
        if not gesture_dir.is_dir() or gesture_dir.name.startswith("."):
            continue
        # Skip the splits directory generated by this script
        if gesture_dir.name == "splits":
            continue
        gid = _name_to_gesture_id(gesture_dir.name)
        if gid is None:
            unmatched_dirs.append(gesture_dir.name)
            continue
        _process_gesture_dir(gesture_dir, gid, "p001", "default")

    return _finish()


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

def preprocess(task: str, from_images: bool = False) -> None:
    root = get_dataset_root(task)
    print(f"\n[Preprocess] Dataset: {task}  root={root}")

    if not root.exists():
        print(f"  [ERROR] Dataset root not found: {root}")
        print("  Options:")
        print("    1. Collect a custom webcam dataset:  bash collect.sh -p p001")
        print(f"   2. Download a dataset and place images under {root}/")
        print("       organised as:  <gesture_id>/<image>.jpg  or  <gesture_name>/<image>.jpg")
        return

    # ── Step 1: gather samples ────────────────────────────────────────────────
    if from_images:
        # Explicit image ingestion requested
        samples = ingest_image_dataset(root)
    else:
        # Try to find existing landmark JSON files first
        samples = _scan_dataset(root)
        if not samples:
            # No landmark files found — check if there are images to process
            has_images = any(
                f.suffix.lower() in _IMAGE_EXTS
                for f in root.rglob("*")
                if f.is_file() and "splits" not in f.parts
            )
            if has_images:
                print("  No landmark JSON files found. "
                      "Detected images — running MediaPipe ingestion...")
                samples = ingest_image_dataset(root)
            else:
                print(f"  [ERROR] No samples found in {root}")
                print("  Options:")
                print("    1. Collect webcam data:  bash collect.sh -p p001")
                print(f"   2. Download a dataset to {root}/  organised as:")
                print("       <gesture_id>/<image>.jpg  or  <gesture_name>/<image>.jpg")
                print("    3. Run with --from-images to re-process existing images")
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
    ap = argparse.ArgumentParser(
        description="Extract landmarks and build train/val/test splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Custom webcam dataset (collect with collect.sh first):
  python code/data/preprocess.py --task hgrd
  python code/data/preprocess.py --task custom

  # Downloaded image dataset (images in <gesture_id>/ or <gesture_name>/ folders):
  python code/data/preprocess.py --task hgrd --from-images
  python code/data/preprocess.py --task both --from-images

  # Preprocess both datasets:
  python code/data/preprocess.py --task both
""",
    )
    ap.add_argument("-t", "--task", required=True,
                    choices=["hgrd", "custom", "both"])
    ap.add_argument("--from-images", action="store_true",
                    help="Run MediaPipe on raw images to generate landmark JSON files "
                         "before building splits. Use this for internet-downloaded "
                         "datasets that contain images but no landmark files.")
    args = ap.parse_args()
    tasks = ["hgrd", "custom"] if args.task == "both" else [args.task]
    for t in tasks:
        preprocess(t, from_images=args.from_images)
