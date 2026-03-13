#!/usr/bin/env bash
# Extract landmarks and build train/val/test splits.
# Usage: bash preprocess.sh -t hgrd|custom|both
source "$(dirname "$0")/env.sh"
TASK="hgrd"
while getopts "t:" opt; do case $opt in t) TASK=$OPTARG ;; esac; done
python code/data/preprocess.py --task "$TASK"
