#!/usr/bin/env bash
# Start REST API server.
# Usage: bash serve.sh -c log/<exp>/best_model.pt      -m cnn
#        bash serve.sh -c log/<exp>/best_model.pt      -m lstm
#        bash serve.sh -c log/<exp>/svm_model.joblib   -m svm
source "$(dirname "$0")/env.sh"
CKPT=""; MODEL="cnn"; PORT=5000; DEVICE="cpu"; DEBUG=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -c) CKPT="$2"; shift 2 ;;
    -m) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --debug) DEBUG="--debug"; shift ;;
    *) shift ;;
  esac
done
[ -z "$CKPT" ] && { echo "Usage: bash serve.sh -c <checkpoint> -m <model>"; exit 1; }
python integrations/api/server.py \
  --checkpoint "$CKPT" --model "$MODEL" \
  --port "$PORT" --device "$DEVICE" $DEBUG
