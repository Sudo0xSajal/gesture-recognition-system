#!/usr/bin/env bash
# Train LSTM/RNN gesture model — runs 3 seeds for robustness.
# Usage: bash train_lstm.sh -c 0 -t hgrd -e exp1 -l 3e-4
source "$(dirname "$0")/env.sh"
GPU=0; TASK="hgrd"; EXP="exp"; LR=3e-4
while getopts "c:t:e:l:" opt; do
  case $opt in c) GPU=$OPTARG ;; t) TASK=$OPTARG ;; e) EXP=$OPTARG ;; l) LR=$OPTARG ;; esac
done
for SEED in 42 123 999; do
  echo ""
  echo "========================================================"
  echo "  LSTM/RNN  task=$TASK  seed=$SEED  lr=$LR"
  echo "========================================================"
  python code/train_lstm.py --task "$TASK" --exp-name "$EXP" \
    --seed "$SEED" --gpu "$GPU" --lr "$LR"
done
echo ""
echo "LSTM training done. Check log/ folder."
