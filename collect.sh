#!/usr/bin/env bash
# Collect gesture data using webcam + MediaPipe.
# Usage: bash collect.sh -p p001 [-t custom] [-s session_name]
source "$(dirname "$0")/env.sh"
TASK="custom"; PARTICIPANT=""; SESSION=""
while getopts "t:p:s:" opt; do
  case $opt in t) TASK=$OPTARG ;; p) PARTICIPANT=$OPTARG ;; s) SESSION=$OPTARG ;; esac
done
[ -z "$PARTICIPANT" ] && { echo "Usage: bash collect.sh -p <participant_id>"; exit 1; }
ARGS="--task $TASK --participant $PARTICIPANT"
[ -n "$SESSION" ] && ARGS="$ARGS --session $SESSION"
python code/data/collector.py $ARGS
