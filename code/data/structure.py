"""
code/data/structure.py
======================
AMOS-like Dataset Structurer for the Gesture Recognition System.

What it does
------------
Walks the canonical raw dataset tree::

    <raw_root>/<participant>/<session>/<gesture_id>/
        frame_000001.jpg
        landmarks_000001.json
        ...

and reorganises it into a clean, split-based output tree::

    <output_root>/
    ├── imagesTr/       training frame images
    ├── landmarksTr/    training landmark JSON files
    ├── imagesVa/       validation frame images
    ├── landmarksVa/    validation landmark JSON files
    ├── imagesTs/       test frame images (images only — labels withheld)
    ├── landmarksTs/    test landmark JSON files
    └── dataset.json    metadata (gesture vocabulary, split statistics, file lists)

The split is performed at the **participant level** so no participant's frames
appear in more than one split (no data leakage).  Split ratios come from
``utils.config.DATA_SPLITS`` (default 70 / 15 / 15).

File naming convention in output folders
-----------------------------------------
Each copied file is renamed to::

    <participant>_<session>_<gesture_id>_<original_filename>

e.g. ``p001_morning_0_frame_000001.jpg``

This makes every filename globally unique and self-describing.

Run
---
    # Use default paths from config.py:
    python code/data/structure.py

    # Custom paths:
    python code/data/structure.py \\
        --raw-root  /path/to/dataset/raw \\
        --output-root /path/to/dataset/structured

    # Copy files instead of moving them:
    python code/data/structure.py --copy

    # Reproducible shuffle with a fixed seed:
    python code/data/structure.py --seed 42
"""

import json
import os
import random
import shutil
import argparse
import textwrap
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from utils.config import (
    RAW_DATASET_PATH,
    PREPROCESSED_DATASET_PATH,
    DATA_SPLITS,
    GESTURE_NAMES,
    GESTURE_ACTIONS,
    NUM_CLASSES,
)

# ---------------------------------------------------------------------------
# Supported image extensions (matches structure_dataset.py)
# ---------------------------------------------------------------------------
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =============================================================================
# HELPERS
# =============================================================================

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS


def _is_landmark_json(path: Path) -> bool:
    return path.suffix.lower() == ".json" and path.stem.startswith("landmarks_")


def _name_to_gesture_id(folder_name: str) -> int | None:
    """
    Map a folder name to a gesture class ID.

    Supports numeric names (``"0"``, ``"00"``, ``"1"``) and canonical
    gesture-name strings (``"thumb_up"``, ``"Thumb Up"``).

    Returns ``None`` when no mapping is found.
    """
    try:
        gid = int(folder_name)
        if gid in GESTURE_NAMES:
            return gid
    except ValueError:
        pass

    normalized = folder_name.lower().replace("_", " ").replace("-", " ").strip()
    for gid, name in GESTURE_NAMES.items():
        if normalized == name.lower():
            return gid
    return None


# =============================================================================
# 1. COLLECT SAMPLES
# =============================================================================

def collect_samples(raw_root: Path) -> Dict[str, List[Tuple[int, Path, Path | None]]]:
    """
    Walk *raw_root* and collect every (gesture_id, frame_path, landmark_path)
    tuple, grouped by participant.

    Parameters
    ----------
    raw_root : path to the raw dataset root (``dataset/raw``)

    Returns
    -------
    A dict mapping ``participant_id`` → list of
    ``(gesture_id, frame_path, landmark_path_or_None)`` tuples.
    """
    samples: Dict[str, List[Tuple[int, Path, Path | None]]] = defaultdict(list)

    if not raw_root.exists():
        return samples

    for participant_dir in sorted(raw_root.iterdir()):
        if not participant_dir.is_dir():
            continue
        participant = participant_dir.name

        for session_dir in sorted(participant_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session = session_dir.name

            for gesture_dir in sorted(session_dir.iterdir()):
                if not gesture_dir.is_dir():
                    continue
                gid = _name_to_gesture_id(gesture_dir.name)
                if gid is None:
                    continue

                def _index(stem: str) -> str:
                    parts = stem.split("_", 1)
                    return parts[1] if len(parts) == 2 else stem

                frames = {
                    _index(p.stem): p
                    for p in gesture_dir.iterdir()
                    if _is_image(p) and p.stem.startswith("frame_")
                }
                landmarks = {
                    _index(p.stem): p
                    for p in gesture_dir.iterdir()
                    if _is_landmark_json(p)
                }

                for idx, frame_path in sorted(frames.items()):
                    lm_path = landmarks.get(idx)
                    samples[participant].append((gid, frame_path, lm_path))

    return samples


# =============================================================================
# 2. SPLIT PARTICIPANTS
# =============================================================================

def split_participants(
    samples: Dict[str, List],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Divide participants into train / val / test lists.

    The split is deterministic given the same *seed* and participant list.
    All ratios are respected as closely as possible; any rounding remainder
    goes to train.

    Parameters
    ----------
    samples     : dict returned by :func:`collect_samples`
    train_ratio : fraction of participants for training
    val_ratio   : fraction of participants for validation
    seed        : random seed for reproducibility

    Returns
    -------
    (train_participants, val_participants, test_participants) — three lists of
    participant ID strings.
    """
    participants = sorted(samples.keys())
    rng = random.Random(seed)
    rng.shuffle(participants)

    n = len(participants)
    n_val   = max(1, round(n * val_ratio)) if n >= 3 else 0
    n_test  = max(1, round(n * (1.0 - train_ratio - val_ratio))) if n >= 3 else 0
    n_train = n - n_val - n_test

    train_ps = participants[:n_train]
    val_ps   = participants[n_train:n_train + n_val]
    test_ps  = participants[n_train + n_val:]

    return train_ps, val_ps, test_ps


# =============================================================================
# 3. COPY / MOVE FILES
# =============================================================================

def _transfer(
    src: Path,
    dest: Path,
    copy: bool,
) -> None:
    """Copy or move *src* → *dest*, creating parent directories as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(str(src), str(dest))
    else:
        shutil.move(str(src), str(dest))


def populate_split(
    participant_list: List[str],
    samples: Dict[str, List[Tuple[int, Path, Path | None]]],
    img_dest: Path,
    lm_dest: Path,
    copy: bool,
    include_landmarks: bool = True,
) -> List[dict]:
    """
    Transfer frame images (and optionally landmark JSON files) for every
    participant in *participant_list* to the destination folders.

    Returns
    -------
    A list of entry dicts suitable for embedding in ``dataset.json``.
    Each entry has keys:
      - ``"image"``    : relative path from output_root
      - ``"landmark"`` : relative path from output_root  (if include_landmarks)
      - ``"gesture_id"``, ``"gesture_name"``
    """
    entries = []

    for participant in sorted(participant_list):
        for gid, frame_path, lm_path in sorted(
            samples[participant], key=lambda t: (t[0], str(t[1]))
        ):
            # Build unique destination filename:
            # <participant>_<session>_<gesture_id>_<original_name>
            session = frame_path.parent.parent.name
            dest_name = f"{participant}_{session}_{gid}_{frame_path.name}"
            dest_frame = img_dest / dest_name
            _transfer(frame_path, dest_frame, copy)

            entry: dict = {
                "image":        f"./{img_dest.name}/{dest_name}",
                "gesture_id":   gid,
                "gesture_name": GESTURE_NAMES[gid],
            }

            if include_landmarks and lm_path is not None:
                lm_dest_name = f"{participant}_{session}_{gid}_{lm_path.name}"
                dest_lm = lm_dest / lm_dest_name
                _transfer(lm_path, dest_lm, copy)
                entry["landmark"] = f"./{lm_dest.name}/{lm_dest_name}"

            entries.append(entry)

    return entries


# =============================================================================
# 4. DATASET JSON
# =============================================================================

def build_dataset_json(
    output_root: Path,
    train_entries: List[dict],
    val_entries: List[dict],
    test_entries: List[dict],
    train_participants: List[str],
    val_participants: List[str],
    test_participants: List[str],
) -> dict:
    """
    Build and return the ``dataset.json`` metadata dictionary.
    """
    return {
        "name": "GestureRecognitionSystem",
        "description": (
            "Hand gesture dataset for bedridden patient assistance, "
            "structured in AMOS-like format."
        ),
        "reference": "https://github.com/Sudo0xSajal/gesture-recognition-system",
        "licence": "Dataset license information here",
        "release": "1.0",
        "modality": {
            "0": "RGB"
        },
        "labels": {
            str(gid): name for gid, name in GESTURE_NAMES.items()
        },
        "gesture_actions": {
            str(gid): action for gid, action in GESTURE_ACTIONS.items()
        },
        "numClasses": NUM_CLASSES,
        "numTraining": len(list((output_root / "imagesTr").glob("*"))),
        "numValidation": len(list((output_root / "imagesVa").glob("*"))),
        "numTest": len(list((output_root / "imagesTs").glob("*"))),
        "participantSplit": {
            "train": sorted(train_participants),
            "val":   sorted(val_participants),
            "test":  sorted(test_participants),
        },
        "training": [
            {
                "image":    e["image"],
                "landmark": e.get("landmark", ""),
                "gesture_id":   e["gesture_id"],
                "gesture_name": e["gesture_name"],
            }
            for e in train_entries
        ],
        "validation": [
            {
                "image":    e["image"],
                "landmark": e.get("landmark", ""),
                "gesture_id":   e["gesture_id"],
                "gesture_name": e["gesture_name"],
            }
            for e in val_entries
        ],
        "test": [
            e["image"] for e in test_entries
        ],
    }


# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================

def structure_dataset(
    raw_root: Path,
    output_root: Path,
    copy: bool = False,
    seed: int = 42,
) -> None:
    """
    Full structuring pipeline.

    1. Collect all (gesture_id, frame, landmark) samples from *raw_root*.
    2. Split participants into train / val / test.
    3. Create the AMOS-like output directory tree under *output_root*.
    4. Transfer files into the correct split folder.
    5. Write ``dataset.json``.

    Parameters
    ----------
    raw_root    : path to ``dataset/raw``
    output_root : destination root (e.g. ``dataset/structured``)
    copy        : copy files instead of moving them
    seed        : random seed for reproducible participant shuffle
    """
    # ── 1. Collect ────────────────────────────────────────────────────────────
    print(f"[structure] Scanning raw dataset at: {raw_root}")
    samples = collect_samples(raw_root)

    if not samples:
        print("[structure] No samples found.  Check that --raw-root points to a "
              "directory matching the expected layout:\n"
              "    <raw_root>/<participant>/<session>/<gesture_id>/<files>")
        return

    total_frames = sum(len(v) for v in samples.values())
    print(f"[structure] Found {total_frames} frame(s) across "
          f"{len(samples)} participant(s).")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    train_ratio = DATA_SPLITS["train_ratio"]
    val_ratio   = DATA_SPLITS["val_ratio"]

    train_ps, val_ps, test_ps = split_participants(
        samples, train_ratio, val_ratio, seed
    )

    print(f"[structure] Split → train: {len(train_ps)}, "
          f"val: {len(val_ps)}, test: {len(test_ps)} participant(s).")

    # ── 3. Create output directory tree ───────────────────────────────────────
    for sub in ["imagesTr", "landmarksTr",
                "imagesVa", "landmarksVa",
                "imagesTs", "landmarksTs"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    op_label = "Copying" if copy else "Moving"
    print(f"[structure] {op_label} files to: {output_root}")

    # ── 4. Transfer files ─────────────────────────────────────────────────────
    train_entries = populate_split(
        train_ps, samples,
        output_root / "imagesTr",
        output_root / "landmarksTr",
        copy=copy,
        include_landmarks=True,
    )

    val_entries = populate_split(
        val_ps, samples,
        output_root / "imagesVa",
        output_root / "landmarksVa",
        copy=copy,
        include_landmarks=True,
    )

    # Test split: images only (landmarks withheld, mirroring MnMs convention)
    test_entries = populate_split(
        test_ps, samples,
        output_root / "imagesTs",
        output_root / "landmarksTs",
        copy=copy,
        include_landmarks=False,
    )

    # ── 5. Write dataset.json ─────────────────────────────────────────────────
    dataset_info = build_dataset_json(
        output_root,
        train_entries, val_entries, test_entries,
        train_ps, val_ps, test_ps,
    )

    json_path = output_root / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_info, f, indent=4)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n✅  Gesture dataset structuring completed successfully.")
    print(f"   Structured dataset saved at : {output_root}")
    print(f"   Training frames             : {dataset_info['numTraining']}")
    print(f"   Validation frames           : {dataset_info['numValidation']}")
    print(f"   Test frames                 : {dataset_info['numTest']}")
    print(f"   dataset.json written to     : {json_path}")


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            AMOS-like Dataset Structurer for the Gesture Recognition System.

            Splits participants into train / val / test and copies (or moves)
            their frames + landmark JSON files into:

                <output_root>/
                ├── imagesTr/      landmarksTr/
                ├── imagesVa/      landmarksVa/
                ├── imagesTs/      landmarksTs/
                └── dataset.json
        """),
    )
    p.add_argument(
        "--raw-root",
        default=None,
        metavar="DIR",
        help=f"Raw dataset root directory (default: {RAW_DATASET_PATH})",
    )
    p.add_argument(
        "--output-root",
        default=None,
        metavar="DIR",
        help=(
            "Destination root for the structured dataset "
            f"(default: {PREPROCESSED_DATASET_PATH / 'structured'})"
        ),
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them (keeps the raw dataset intact)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for participant shuffle (default: 42)",
    )
    return p


def main() -> None:
    args  = _build_parser().parse_args()
    raw   = Path(args.raw_root)   if args.raw_root    else Path(RAW_DATASET_PATH)
    out   = Path(args.output_root) if args.output_root else (
        Path(PREPROCESSED_DATASET_PATH) / "structured"
    )
    structure_dataset(raw, out, copy=args.copy, seed=args.seed)


if __name__ == "__main__":
    main()
