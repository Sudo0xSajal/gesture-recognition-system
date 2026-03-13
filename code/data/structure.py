"""
code/data/structure.py
======================
AMOS-like Dataset Structurer for the Gesture Recognition System.

Mirrors the scanning, participant-level splitting, and sample-dict conventions
of ``preprocess.py``, while producing an AMOS-like output directory tree
instead of split .txt files and .npz caches.

What it does
------------
Scans the same dataset roots that ``preprocess.py`` uses:

1. **collector.py / webcam dataset** (standard structure)::

       <root>/raw/<participant>/<session>/<gesture_id>/
           frame_000001.jpg
           landmarks_000001.json

2. **Downloaded / flat image dataset**::

       <root>/<gesture_id>/*.{jpg,jpeg,png}
       <root>/<gesture_name>/*.{jpg,jpeg,png}

Then produces an AMOS-like output tree::

    <output_root>/
    ├── imagesTr/       training frame images
    ├── landmarksTr/    training landmark JSON files
    ├── imagesVa/       validation frame images
    ├── landmarksVa/    validation landmark JSON files
    ├── imagesTs/       test frame images  (labels withheld)
    ├── landmarksTs/    test landmark JSON files
    └── dataset.json    metadata (gesture vocabulary, split stats, file lists)

The participant-level split uses ``StratifiedShuffleSplit`` (same as
``preprocess.py``) so every participant's samples stay in exactly one split
and the split is stratified by each participant's dominant gesture class.

File naming convention in output folders
-----------------------------------------
Each file is renamed to::

    <participant>_<session>_<gesture_id>_<original_filename>

e.g. ``p001_morning_0_frame_000001.jpg``

This makes filenames globally unique and self-describing.

Run
---
    # Use default paths from config.py:
    python code/data/structure.py

    # Custom paths:
    python code/data/structure.py \\
        --root        /path/to/dataset/raw \\
        --output-root /path/to/dataset/structured

    # Copy files instead of moving them (keeps raw dataset intact):
    python code/data/structure.py --copy
"""

import json
import shutil
import argparse
import textwrap
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import StratifiedShuffleSplit

from utils.config import (
    RAW_DATASET_PATH,
    PREPROCESSED_DATASET_PATH,
    DATA_SPLITS,
    GESTURE_NAMES,
    GESTURE_ACTIONS,
    NUM_CLASSES,
)

# Supported image extensions — matches preprocess.py
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =============================================================================
# HELPERS  (aligned with preprocess.py conventions)
# =============================================================================

def _name_to_gesture_id(folder_name: str) -> Optional[int]:
    """
    Map a folder name to a gesture class ID.

    Supports:
      - Numeric folder names: ``"0"``, ``"00"``, ``"1"`` → gesture ID directly
      - Gesture name folders: ``"thumb_up"``, ``"Thumb Up"`` → matched
        case-insensitively after stripping underscores, hyphens, and spaces

    Returns ``None`` if no match is found.
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


# =============================================================================
# 1. DATASET SCANNER  (mirrors _scan_dataset in preprocess.py)
# =============================================================================

def _scan_dataset(root: Path) -> List[Dict]:
    """
    Scan dataset directory for all ``(landmark_path, image_path, gesture_id,
    participant, session)`` sample dicts.

    Expected directory structures (same as ``preprocess.py``):

    **Standard** (collector.py output)::

        <root>/raw/<participant>/<session>/<gesture_id>/
            frame_000001.jpg
            landmarks_000001.json

    **Legacy flat**::

        <root>/<gesture_id>/*.json

    Returns a list of sample dicts with keys:
      ``landmark_path``, ``image_path``, ``gesture_id``,
      ``participant``, ``session``.
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


def _scan_image_dataset(root: Path) -> List[Dict]:
    """
    Scan a downloaded image dataset that is organised by class folder but may
    not have landmark JSON files yet.

    Supported layouts (same as ``ingest_image_dataset`` in preprocess.py):

    * ``<root>/raw/<participant>/<session>/<gesture_id>/*.{jpg,png,...}``
    * ``<root>/<gesture_id>/*.{jpg,png,...}``
    * ``<root>/<gesture_name>/*.{jpg,png,...}``

    Only images whose companion ``landmarks_<stem>.json`` already exists are
    returned.  Run ``preprocess.py --from-images`` first to generate the JSON
    files, then call this script.

    Returns the same sample-dict format as :func:`_scan_dataset`.
    """
    samples = []
    raw_root = root / "raw"

    def _collect_from_gesture_dir(
        gesture_dir: Path, gid: int, participant: str, session: str
    ) -> None:
        for img_path in sorted(gesture_dir.iterdir()):
            if img_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            stem = img_path.stem
            lm_path = gesture_dir / f"landmarks_{stem}.json"
            if not lm_path.exists():
                continue
            frame_path = (
                img_path if stem.startswith("frame_")
                else gesture_dir / f"frame_{stem}{img_path.suffix}"
            )
            samples.append({
                "landmark_path": lm_path,
                "image_path":    frame_path if frame_path.exists() else img_path,
                "gesture_id":    gid,
                "participant":   participant,
                "session":       session,
            })

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
                    _collect_from_gesture_dir(
                        gesture_dir, gid, participant.name, session.name)
        if samples:
            return samples

    for gesture_dir in sorted(root.iterdir()):
        if not gesture_dir.is_dir() or gesture_dir.name.startswith("."):
            continue
        if gesture_dir.name == "splits":
            continue
        gid = _name_to_gesture_id(gesture_dir.name)
        if gid is None:
            continue
        _collect_from_gesture_dir(gesture_dir, gid, "p001", "default")

    return samples


# =============================================================================
# 2. PARTICIPANT-LEVEL SPLIT  (mirrors _participant_split in preprocess.py)
# =============================================================================

def _participant_split(
    samples: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split at participant level so no participant appears in more than one split.

    This prevents data leakage: the model cannot memorise a specific person's
    hand shape and must generalise to unseen participants.

    Strategy (identical to ``preprocess.py``):
      1. Compute each participant's dominant gesture class.
      2. Use ``StratifiedShuffleSplit`` to divide participants into
         train+val vs test (stratified by dominant gesture).
      3. Repeat for train vs val within the train+val pool.
      4. Assign all samples from each participant to the corresponding split.

    Falls back to sample-level split when fewer than 3 participants are present.

    Returns ``(train_samples, val_samples, test_samples)``.
    """
    by_participant: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_participant[s["participant"]].append(s)

    participants = list(by_participant.keys())

    # Fewer than 3 participants → sample-level fallback
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
        counts: Dict[int, int] = defaultdict(int)
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
# 3. FILE TRANSFER
# =============================================================================

def _transfer(src: Path, dest: Path, copy: bool) -> None:
    """Copy or move *src* → *dest*, creating parent directories as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(str(src), str(dest))
    else:
        shutil.move(str(src), str(dest))


def _populate_split(
    split_samples: List[Dict],
    img_dest: Path,
    lm_dest: Path,
    copy: bool,
    include_landmarks: bool = True,
) -> List[dict]:
    """
    Transfer frame images (and optionally landmark JSON files) for every
    sample in *split_samples* to the destination folders.

    Uses the same sample-dict format as ``preprocess.py``
    (``{landmark_path, image_path, gesture_id, participant, session}``).

    Returns a list of entry dicts for embedding in ``dataset.json``.
    Each entry has keys:
      ``"image"``, ``"gesture_id"``, ``"gesture_name"``,
      and ``"landmark"`` when *include_landmarks* is ``True``.
    """
    entries = []

    for s in sorted(split_samples, key=lambda x: (x["gesture_id"],
                                                   str(x["landmark_path"]))):
        participant = s["participant"]
        session     = s["session"]
        gid         = s["gesture_id"]
        lm_path     = s["landmark_path"]
        img_path    = s["image_path"]

        entry: dict = {"gesture_id": gid, "gesture_name": GESTURE_NAMES[gid]}

        # ── Frame image ───────────────────────────────────────────────────────
        if img_path is not None and img_path.exists():
            dest_name  = f"{participant}_{session}_{gid}_{img_path.name}"
            dest_frame = img_dest / dest_name
            _transfer(img_path, dest_frame, copy)
            entry["image"] = f"./{img_dest.name}/{dest_name}"
        else:
            entry["image"] = ""

        # ── Landmark JSON ─────────────────────────────────────────────────────
        if include_landmarks:
            lm_dest_name = f"{participant}_{session}_{gid}_{lm_path.name}"
            dest_lm      = lm_dest / lm_dest_name
            _transfer(lm_path, dest_lm, copy)
            entry["landmark"] = f"./{lm_dest.name}/{lm_dest_name}"

        entries.append(entry)

    return entries


# =============================================================================
# 4. DATASET JSON
# =============================================================================

def _build_dataset_json(
    train_entries: List[dict],
    val_entries: List[dict],
    test_entries: List[dict],
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
) -> dict:
    """
    Build and return the ``dataset.json`` metadata dictionary.
    """
    train_participants = sorted({s["participant"] for s in train_samples})
    val_participants   = sorted({s["participant"] for s in val_samples})
    test_participants  = sorted({s["participant"] for s in test_samples})

    return {
        "name": "GestureRecognitionSystem",
        "description": (
            "Hand gesture dataset for bedridden patient assistance, "
            "structured in AMOS-like format."
        ),
        "reference": "https://github.com/Sudo0xSajal/gesture-recognition-system",
        "license": "Dataset license information here",
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
        "numTraining":   len(train_entries),
        "numValidation": len(val_entries),
        "numTest":       len(test_entries),
        "participantSplit": {
            "train": train_participants,
            "val":   val_participants,
            "test":  test_participants,
        },
        "training": [
            {
                "image":        e["image"],
                "landmark":     e.get("landmark", ""),
                "gesture_id":   e["gesture_id"],
                "gesture_name": e["gesture_name"],
            }
            for e in train_entries
        ],
        "validation": [
            {
                "image":        e["image"],
                "landmark":     e.get("landmark", ""),
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
    root: Path,
    output_root: Path,
    copy: bool = False,
) -> None:
    """
    Full AMOS-like structuring pipeline.

    Mirrors ``preprocess.preprocess()`` in scanning and splitting strategy,
    then copies (or moves) files into an AMOS-like output tree and writes
    ``dataset.json``.

    Parameters
    ----------
    root        : dataset root — same path passed to ``preprocess.py``
                  (usually ``RAW_DATASET_PATH = "dataset/raw"``).
                  The script looks for ``root/raw/`` (collector.py output)
                  and falls back to a flat ``<root>/<gesture_id>/`` layout.
    output_root : destination root for the structured dataset.
    copy        : if ``True``, copy files instead of moving them.
    """
    print(f"\n[structure] root={root}  output={output_root}")

    if not root.exists():
        print(f"  [ERROR] Dataset root not found: {root}")
        print("  Options:")
        print("    1. Collect a custom webcam dataset:  bash collect.sh -p p001")
        print(f"   2. Download a dataset and place images under {root}/")
        print("       organised as:  <gesture_id>/<image>.jpg  "
              "or  <gesture_name>/<image>.jpg")
        return

    # ── 1. Scan ───────────────────────────────────────────────────────────────
    samples = _scan_dataset(root)
    if not samples:
        # Try image-only layout (landmark JSONs alongside images)
        samples = _scan_image_dataset(root)
    if not samples:
        print(f"  [ERROR] No samples found in {root}")
        print("  Options:")
        print("    1. Collect webcam data:  bash collect.sh -p p001")
        print(f"   2. Download a dataset to {root}/  organised as:")
        print("       <gesture_id>/<image>.jpg  or  <gesture_name>/<image>.jpg")
        print("    3. Run preprocess.py --from-images first to generate landmark "
              "JSON files, then run this script.")
        return

    n_participants = len({s["participant"] for s in samples})
    print(f"  Found {len(samples)} samples  |  {n_participants} participant(s)")

    # Class distribution (matches preprocess.py output style)
    by_class: Dict[int, int] = defaultdict(int)
    for s in samples:
        by_class[s["gesture_id"]] += 1
    print("  Class distribution:")
    for gid in range(NUM_CLASSES):
        cnt = by_class.get(gid, 0)
        bar = "█" * (cnt // 10)
        print(f"    [{gid:02d}] {GESTURE_NAMES[gid]:22s}: {cnt:4d}  {bar}")

    # ── 2. Participant-level split ─────────────────────────────────────────────
    train_s, val_s, test_s = _participant_split(samples)
    print(f"\n  Split → train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")

    # ── 3. Create output directory tree ───────────────────────────────────────
    for sub in ["imagesTr", "landmarksTr",
                "imagesVa", "landmarksVa",
                "imagesTs", "landmarksTs"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    op_label = "Copying" if copy else "Moving"
    print(f"\n  {op_label} files to: {output_root}")

    # ── 4. Transfer files ─────────────────────────────────────────────────────
    train_entries = _populate_split(
        train_s,
        output_root / "imagesTr",
        output_root / "landmarksTr",
        copy=copy,
        include_landmarks=True,
    )

    val_entries = _populate_split(
        val_s,
        output_root / "imagesVa",
        output_root / "landmarksVa",
        copy=copy,
        include_landmarks=True,
    )

    # Test split: images only (landmarks withheld, mirroring MnMs convention)
    test_entries = _populate_split(
        test_s,
        output_root / "imagesTs",
        output_root / "landmarksTs",
        copy=copy,
        include_landmarks=False,
    )

    # ── 5. Write dataset.json ─────────────────────────────────────────────────
    dataset_info = _build_dataset_json(
        train_entries, val_entries, test_entries,
        train_s, val_s, test_s,
    )

    json_path = output_root / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_info, f, indent=4)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[structure] Done — structured dataset saved to {output_root}\n")
    print("✅  Gesture dataset structuring completed successfully.")
    print(f"   Training samples   : {dataset_info['numTraining']}")
    print(f"   Validation samples : {dataset_info['numValidation']}")
    print(f"   Test samples       : {dataset_info['numTest']}")
    print(f"   dataset.json       : {json_path}")


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            AMOS-like Dataset Structurer for the Gesture Recognition System.

            Applies the same scanning and participant-level splitting logic as
            preprocess.py, then copies (or moves) files into:

                <output_root>/
                ├── imagesTr/      landmarksTr/
                ├── imagesVa/      landmarksVa/
                ├── imagesTs/      landmarksTs/
                └── dataset.json
        """),
        epilog=textwrap.dedent("""\
            Examples
            --------
              # Default paths from config.py:
              python code/data/structure.py

              # Custom paths, keep raw data intact:
              python code/data/structure.py \\
                  --root /path/to/dataset/raw \\
                  --output-root /path/to/dataset/structured \\
                  --copy
        """),
    )
    p.add_argument(
        "--root",
        default=None,
        metavar="DIR",
        help=f"Dataset root directory (default: {RAW_DATASET_PATH})",
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
    return p


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(args.root) if args.root else Path(RAW_DATASET_PATH)
    out  = (
        Path(args.output_root) if args.output_root
        else Path(PREPROCESSED_DATASET_PATH) / "structured"
    )
    structure_dataset(root, out, copy=args.copy)


if __name__ == "__main__":
    main()
