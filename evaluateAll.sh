#!/usr/bin/env bash
# Evaluate all trained models in log/.
# Usage: bash evaluateAll.sh -t hgrd
source "$(dirname "$0")/env.sh"
TASK="hgrd"; DEVICE="cpu"
while getopts "t:d:" opt; do
  case $opt in t) TASK=$OPTARG ;; d) DEVICE=$OPTARG ;; esac
done
COUNT=0

# PyTorch models (CNN and LSTM)
for CKPT in log/*/best_model.pt; do
  [ -f "$CKPT" ] || continue
  DIR=$(basename "$(dirname "$CKPT")")
  if echo "$DIR" | grep -q "_cnn_"; then      MODEL="cnn"
  elif echo "$DIR" | grep -q "_lstm_"; then   MODEL="lstm"
  else continue; fi
  echo ""
  echo "[evaluateAll] $CKPT  model=$MODEL"
  python code/evaluate.py --task "$TASK" --model "$MODEL" \
         --checkpoint "$CKPT" --device "$DEVICE"
  COUNT=$((COUNT+1))
done

# SVM models
for CKPT in log/*/svm_model.joblib; do
  [ -f "$CKPT" ] || continue
  echo ""
  echo "[evaluateAll] $CKPT  model=svm"
  python code/evaluate.py --task "$TASK" --model svm --checkpoint "$CKPT"
  COUNT=$((COUNT+1))
done

echo ""
echo "Evaluated $COUNT models total."
