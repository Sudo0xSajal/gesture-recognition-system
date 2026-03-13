#!/usr/bin/env bash
# Extract landmarks and build train/val/test splits.
#
# Usage:
#   bash preprocess.sh -t hgrd|custom|both
#   bash preprocess.sh -t hgrd --from-images   # for internet-downloaded datasets
#
# Use --from-images when your dataset contains raw images (downloaded from the
# internet) instead of MediaPipe landmark JSON files.  The script will run
# MediaPipe on every image to generate landmark files, then build the splits.
source "$(dirname "$0")/env.sh"
TASK="hgrd"
FROM_IMAGES=""
while getopts "t:-:" opt; do
  case $opt in
    t) TASK=$OPTARG ;;
    -)
      case $OPTARG in
        from-images) FROM_IMAGES="--from-images" ;;
      esac
      ;;
  esac
done
python code/data/preprocess.py --task "$TASK" $FROM_IMAGES
