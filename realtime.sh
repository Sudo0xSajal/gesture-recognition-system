#!/usr/bin/env bash
# Start live webcam inference.
# Usage: bash realtime.sh -c log/<exp>/best_model.pt    -m cnn
#        bash realtime.sh -c log/<exp>/best_model.pt    -m lstm
#        bash realtime.sh -c log/<exp>/svm_model.joblib -m svm
#        bash realtime.sh -c ... -m svm --save-log events.json --window 7
source "$(dirname "$0")/env.sh"
CKPT=""; MODEL="cnn"; ARGS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -c) CKPT="$2"; shift 2 ;;
    -m) MODEL="$2"; shift 2 ;;
    *) ARGS="$ARGS $1"; shift ;;
  esac
done
[ -z "$CKPT" ] && { echo "Usage: bash realtime.sh -c <checkpoint> -m <model>"; exit 1; }
python integrations/realtime/run_webcam.py \
  --checkpoint "$CKPT" --model "$MODEL" $ARGS
