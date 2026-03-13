#!/usr/bin/env bash
# Collect gesture data using webcam + MediaPipe.
# Usage: bash collect.sh -p p001 [-s session_name]
source "$(dirname "$0")/env.sh"
PARTICIPANT=""; SESSION=""
while getopts "p:s:" opt; do
  case $opt in p) PARTICIPANT=$OPTARG ;; s) SESSION=$OPTARG ;; esac
done
[ -z "$PARTICIPANT" ] && { echo "Usage: bash collect.sh -p <participant_id>"; exit 1; }
ARGS="--participant $PARTICIPANT"
[ -n "$SESSION" ] && ARGS="$ARGS --session $SESSION"
python code/data/collector.py $ARGS
