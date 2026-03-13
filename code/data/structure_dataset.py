"""
code/data/structure_dataset.py
===============================
Raw Dataset Organiser — prepares any raw gesture image collection for the
gesture-recognition pipeline.

What it does
------------
1. **Validate** — walks the expected raw dataset tree and reports missing
   gesture folders, empty folders, corrupt JSON files, and mismatched image/
   landmark pairs.

2. **Scaffold** — creates the full directory tree
   ``<raw_root>/<participant>/<session>/<gesture_id>/``
   for every participant/session combination that already exists but is
   missing some gesture sub-folders.

3. **Reorganise** — ingests an *unstructured* flat image dump (images named
   ``<gesture_id>_*.jpg`` or arranged in ``<gesture_id>/`` sub-folders) and
   moves or copies them into the canonical layout that ``preprocess.py`` and
   ``collector.py`` expect.

4. **Report** — prints a per-participant, per-gesture frame count table and
   flags any class that has fewer frames than the configurable minimum.

Expected canonical layout (output / validator target)
------------------------------------------------------
    dataset/raw/
    └── <participant>/          e.g. p001
        └── <session>/          e.g. morning  or  20240313_143022
            ├── 0/
            │   ├── frame_000001.jpg
            │   ├── landmarks_000001.json
            │   └── ...
            ├── 1/
            │   └── ...
            └── [2-24]/
                └── ...

Run
---
    # Validate existing dataset and print report:
    python code/data/structure_dataset.py

    # Scaffold missing gesture folders for every existing participant/session:
    python code/data/structure_dataset.py --scaffold

    # Reorganise an unstructured image dump into participant p001 / session raw:
    python code/data/structure_dataset.py --reorganise /path/to/flat/dump \\
        --participant p001 --session raw

    # Validate and report, specify a custom dataset root:
    python code/data/structure_dataset.py --root /custom/dataset/raw
"""

import json
import shutil
import argparse
import textwrap
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from utils.config import (
    RAW_DATASET_PATH,
    NUM_CLASSES,
    GESTURE_NAMES,
)

# Supported image extensions
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Minimum frames per gesture class considered "well-populated"
_MIN_FRAMES_DEFAULT = 50


# =============================================================================
# HELPERS
# =============================================================================

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS


def _is_landmark_json(path: Path) -> bool:
    return path.suffix.lower() == ".json" and path.stem.startswith("landmarks_")


def _is_frame_jpg(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg"} and path.stem.startswith("frame_")


def _validate_landmark_json(path: Path) -> Tuple[bool, str]:
    """
    Check that a landmarks JSON file is well-formed.

    Returns (True, "") if valid, or (False, reason) if not.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        lms = data.get("landmarks")
        if not isinstance(lms, list):
            return False, "missing 'landmarks' list"
        if len(lms) != 21:
            return False, f"expected 21 landmarks, got {len(lms)}"
        for lm in lms:
            for key in ("x", "y", "z"):
                if key not in lm:
                    return False, f"landmark missing key '{key}'"
        if "gesture_id" not in data:
            return False, "missing 'gesture_id'"
        return True, ""
    except json.JSONDecodeError as exc:
        return False, f"JSON parse error: {exc}"
    except Exception as exc:
        return False, f"unexpected error: {exc}"


def _name_to_gesture_id(folder_name: str) -> Optional[int]:
    """
    Map a folder name to a gesture class ID.

    Supports:
      - Numeric folder names: "0", "00", "1" → gesture ID directly
      - Gesture name folders: "thumb_up", "Thumb Up" → matched
        case-insensitively after stripping underscores, hyphens, spaces

    Returns None if no match is found.
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
# 1. VALIDATOR
# =============================================================================

class DatasetReport:
    """
    Holds the results of a dataset validation pass.

    Attributes
    ----------
    frame_counts       : {participant → {gesture_id → frame_count}}
    missing_gestures   : {(participant, session) → [missing gesture IDs]}
    empty_folders      : list of Path objects that contain no images
    corrupt_json       : list of (Path, reason) tuples
    unpaired_frames    : list of frame Paths that have no matching landmark JSON
    unpaired_landmarks : list of landmark Paths that have no matching frame
    """

    def __init__(self):
        self.frame_counts: Dict[str, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.missing_gestures: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.empty_folders: List[Path] = []
        self.corrupt_json: List[Tuple[Path, str]] = []
        self.unpaired_frames: List[Path] = []
        self.unpaired_landmarks: List[Path] = []


def validate_dataset(root: Path, min_frames: int = _MIN_FRAMES_DEFAULT) -> DatasetReport:
    """
    Walk *root* and validate its structure, returning a :class:`DatasetReport`.

    Parameters
    ----------
    root       : path to the raw dataset root (e.g. ``dataset/raw``)
    min_frames : warn when a gesture class has fewer than this many frames
    """
    report = DatasetReport()

    if not root.exists():
        print(f"[structure_dataset] Raw dataset root does not exist: {root}")
        return report

    for participant_dir in sorted(root.iterdir()):
        if not participant_dir.is_dir():
            continue
        participant = participant_dir.name

        for session_dir in sorted(participant_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session = session_dir.name

            # Check which gesture sub-folders exist
            present_gids = set()
            for gesture_dir in sorted(session_dir.iterdir()):
                if not gesture_dir.is_dir():
                    continue
                gid = _name_to_gesture_id(gesture_dir.name)
                if gid is None:
                    continue
                present_gids.add(gid)

                # Collect frames and landmarks in this gesture folder
                frames    = {p.stem.split("_", 1)[1]: p
                             for p in gesture_dir.iterdir()
                             if _is_frame_jpg(p)}
                landmarks = {p.stem.split("_", 1)[1]: p
                             for p in gesture_dir.iterdir()
                             if _is_landmark_json(p)}

                if not frames and not landmarks:
                    report.empty_folders.append(gesture_dir)
                    continue

                frame_count = len(frames)
                report.frame_counts[participant][gid] += frame_count

                # Paired-file checks
                for idx, frame_path in frames.items():
                    if idx not in landmarks:
                        report.unpaired_frames.append(frame_path)

                for idx, lm_path in landmarks.items():
                    if idx not in frames:
                        report.unpaired_landmarks.append(lm_path)
                    else:
                        ok, reason = _validate_landmark_json(lm_path)
                        if not ok:
                            report.corrupt_json.append((lm_path, reason))

            # Missing gesture sub-folders
            missing = [gid for gid in range(NUM_CLASSES) if gid not in present_gids]
            if missing:
                report.missing_gestures[(participant, session)] = missing

    return report


# =============================================================================
# 2. SCAFFOLD
# =============================================================================

def scaffold_dataset(root: Path) -> None:
    """
    Create all missing ``<gesture_id>/`` sub-directories for every
    participant/session pair already present in *root*.

    Does nothing if ``root`` does not exist yet (use ``--root`` to point to
    the correct location before collecting data).
    """
    if not root.exists():
        print(f"[structure_dataset] Root does not exist, nothing to scaffold: {root}")
        return

    created = 0
    for participant_dir in sorted(root.iterdir()):
        if not participant_dir.is_dir():
            continue
        for session_dir in sorted(participant_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            for gid in range(NUM_CLASSES):
                gdir = session_dir / str(gid)
                if not gdir.exists():
                    gdir.mkdir(parents=True, exist_ok=True)
                    created += 1

    if created:
        print(f"[structure_dataset] Scaffolded {created} missing gesture folder(s).")
    else:
        print("[structure_dataset] All gesture folders already present — nothing to scaffold.")


# =============================================================================
# 3. REORGANISER
# =============================================================================

def reorganise_flat_dump(
    src: Path,
    root: Path,
    participant: str,
    session: str,
    copy: bool = False,
) -> None:
    """
    Move (or copy) images from an unstructured *src* directory into the
    canonical ``<root>/<participant>/<session>/<gesture_id>/`` layout.

    Supported source layouts
    ------------------------
    a) Numeric sub-folders::

        src/0/*.jpg   src/1/*.jpg   ...   src/24/*.jpg

    b) Named sub-folders matching GESTURE_NAMES::

        src/thumb_up/*.jpg   src/open_palm/*.jpg   ...

    c) Flat dump with ``<gesture_id>_<anything>.<ext>`` filename prefix::

        src/0_frame001.jpg   src/0_frame002.jpg   src/1_img.png   ...

    d) Flat dump with ``<gesture_name>_<anything>.<ext>`` filename prefix
       (case-insensitive, underscores accepted instead of spaces)::

        src/thumb_up_001.jpg   src/open_palm_002.jpg   ...

    Parameters
    ----------
    src         : path to the flat/unstructured source directory
    root        : raw dataset root (usually ``dataset/raw``)
    participant : participant ID string, e.g. ``"p001"``
    session     : session name, e.g. ``"downloaded"``
    copy        : if True, *copy* files instead of moving them
    """
    if not src.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src}")

    op       = shutil.copy2 if copy else shutil.move
    op_label = "Copied" if copy else "Moved"
    counters: Dict[int, int] = defaultdict(int)

    # ── Layout (a) and (b): sub-folder per gesture ───────────────────────────
    has_subdirs = any(p.is_dir() for p in src.iterdir())
    if has_subdirs:
        for sub in sorted(src.iterdir()):
            if not sub.is_dir():
                continue
            gid = _name_to_gesture_id(sub.name)
            if gid is None:
                print(f"  [SKIP] Cannot map folder '{sub.name}' to a gesture ID.")
                continue
            dest_dir = root / participant / session / str(gid)
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in sorted(sub.iterdir()):
                if _is_image(img):
                    idx       = counters[gid] + 1
                    dest_name = f"frame_{idx:06d}{img.suffix.lower()}"
                    op(str(img), str(dest_dir / dest_name))
                    counters[gid] += 1
        _print_reorganise_summary(counters, op_label, root, participant, session)
        return

    # ── Layout (c) and (d): flat dump ────────────────────────────────────────
    for img in sorted(src.iterdir()):
        if not _is_image(img):
            continue
        gid = _guess_gesture_id_from_filename(img.stem)
        if gid is None:
            print(f"  [SKIP] Cannot determine gesture ID from filename: {img.name}")
            continue
        dest_dir = root / participant / session / str(gid)
        dest_dir.mkdir(parents=True, exist_ok=True)
        idx       = counters[gid] + 1
        dest_name = f"frame_{idx:06d}{img.suffix.lower()}"
        op(str(img), str(dest_dir / dest_name))
        counters[gid] += 1

    _print_reorganise_summary(counters, op_label, root, participant, session)


def _guess_gesture_id_from_filename(stem: str) -> Optional[int]:
    """
    Try to extract a gesture ID from a bare filename stem using two strategies:

    1. Numeric prefix: ``"0_frame001"`` → 0, ``"24_img"`` → 24
    2. Gesture-name prefix: ``"thumb_up_001"`` → 0
    """
    # Strategy 1: numeric prefix
    parts = stem.split("_", 1)
    try:
        gid = int(parts[0])
        if gid in GESTURE_NAMES:
            return gid
    except ValueError:
        pass

    # Strategy 2: gesture-name prefix (try successively longer token groups)
    tokens = stem.lower().replace("-", "_").split("_")
    for length in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:length])
        for gid, name in GESTURE_NAMES.items():
            if candidate == name.lower():
                return gid
    return None


def _print_reorganise_summary(
    counters: Dict[int, int],
    op_label: str,
    root: Path,
    participant: str,
    session: str,
) -> None:
    total = sum(counters.values())
    if total == 0:
        print("[structure_dataset] No images were reorganised.")
        return
    print(f"\n[structure_dataset] {op_label} {total} image(s) → "
          f"{root / participant / session}")
    for gid in sorted(counters):
        print(f"  [{gid:02d}] {GESTURE_NAMES[gid]:20s}: {counters[gid]:5d} frame(s)")


# =============================================================================
# 4. REPORTER
# =============================================================================

def print_report(report: DatasetReport, min_frames: int = _MIN_FRAMES_DEFAULT) -> None:
    """
    Pretty-print the contents of *report* to stdout.
    """
    sep = "─" * 72

    # ── Frame count table ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  DATASET FRAME COUNT REPORT")
    print(sep)

    all_participants = sorted(report.frame_counts.keys())
    if not all_participants:
        print("  No data found under the raw dataset root.")
    else:
        header = f"  {'Gesture':24s}" + "".join(f"{p:>10s}" for p in all_participants)
        print(header)
        print("  " + "-" * (24 + 10 * len(all_participants)))
        for gid in range(NUM_CLASSES):
            name   = GESTURE_NAMES[gid]
            counts = [report.frame_counts[p].get(gid, 0) for p in all_participants]
            flag   = "  ⚠ LOW" if any(0 < c < min_frames for c in counts) else ""
            row    = f"  [{gid:02d}] {name:20s}" + "".join(f"{c:>10d}" for c in counts)
            print(row + flag)
        totals = [sum(report.frame_counts[p].values()) for p in all_participants]
        print("  " + "-" * (24 + 10 * len(all_participants)))
        print(f"  {'TOTAL':24s}" + "".join(f"{t:>10d}" for t in totals))

    # ── Issues ────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  ISSUES")
    print(sep)

    issue_count = 0

    if report.missing_gestures:
        print("\n  Missing gesture folders:")
        for (participant, session), gids in sorted(report.missing_gestures.items()):
            names = ", ".join(f"{g} ({GESTURE_NAMES[g]})" for g in gids[:5])
            more  = f"  … and {len(gids) - 5} more" if len(gids) > 5 else ""
            print(f"    {participant}/{session}: [{names}]{more}")
        issue_count += len(report.missing_gestures)

    if report.empty_folders:
        print(f"\n  Empty gesture folders ({len(report.empty_folders)}):")
        for p in report.empty_folders[:10]:
            print(f"    {p}")
        if len(report.empty_folders) > 10:
            print(f"    … and {len(report.empty_folders) - 10} more")
        issue_count += len(report.empty_folders)

    if report.corrupt_json:
        print(f"\n  Corrupt / malformed landmark JSON files ({len(report.corrupt_json)}):")
        for path, reason in report.corrupt_json[:10]:
            print(f"    {path}  →  {reason}")
        if len(report.corrupt_json) > 10:
            print(f"    … and {len(report.corrupt_json) - 10} more")
        issue_count += len(report.corrupt_json)

    if report.unpaired_frames:
        print(f"\n  Frame images with no matching landmark JSON ({len(report.unpaired_frames)}):")
        for p in report.unpaired_frames[:5]:
            print(f"    {p}")
        if len(report.unpaired_frames) > 5:
            print(f"    … and {len(report.unpaired_frames) - 5} more")
        issue_count += len(report.unpaired_frames)

    if report.unpaired_landmarks:
        print(f"\n  Landmark JSON files with no matching frame ({len(report.unpaired_landmarks)}):")
        for p in report.unpaired_landmarks[:5]:
            print(f"    {p}")
        if len(report.unpaired_landmarks) > 5:
            print(f"    … and {len(report.unpaired_landmarks) - 5} more")
        issue_count += len(report.unpaired_landmarks)

    if not issue_count:
        print("\n  ✔  No issues found.")

    print(f"\n{sep}\n")


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="structure_dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Raw Dataset Organiser for the Gesture Recognition System.

            Modes
            -----
            (default)     Validate and report on the existing raw dataset.
            --scaffold    Also create any missing gesture sub-folders.
            --reorganise  Ingest an unstructured image dump into the canonical
                          directory layout.
        """),
    )
    p.add_argument(
        "--root",
        default=None,
        help=f"Raw dataset root directory (default: {RAW_DATASET_PATH})",
    )
    p.add_argument(
        "--min-frames",
        type=int,
        default=_MIN_FRAMES_DEFAULT,
        metavar="N",
        help="Minimum frames per gesture to avoid a LOW warning (default: 50)",
    )
    p.add_argument(
        "--scaffold",
        action="store_true",
        help="Create missing gesture sub-folders for all existing participant/session pairs",
    )
    p.add_argument(
        "--reorganise",
        metavar="SRC",
        default=None,
        help="Path to an unstructured image dump to reorganise into the canonical layout",
    )
    p.add_argument(
        "-p", "--participant",
        default="p001",
        help="Participant ID to use when reorganising (default: p001)",
    )
    p.add_argument(
        "-s", "--session",
        default="downloaded",
        help="Session name to use when reorganising (default: downloaded)",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them when using --reorganise",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(args.root) if args.root else Path(RAW_DATASET_PATH)

    if args.reorganise:
        reorganise_flat_dump(
            src=Path(args.reorganise),
            root=root,
            participant=args.participant,
            session=args.session,
            copy=args.copy,
        )

    if args.scaffold:
        scaffold_dataset(root)

    # Always validate and report
    print(f"[structure_dataset] Validating dataset at: {root}")
    report = validate_dataset(root, min_frames=args.min_frames)
    print_report(report, min_frames=args.min_frames)


if __name__ == "__main__":
    main()
