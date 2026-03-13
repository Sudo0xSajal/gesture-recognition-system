#!/usr/bin/env bash
# Train CNN gesture model — runs 3 seeds for robustness.
# Usage: bash train_cnn.sh -c 0 -t hgrd -e exp1 -l 3e-4
# Flags: -c GPU, -t task, -e experiment name, -l learning rate
source "$(dirname "$0")/env.sh"
GPU=0; TASK="hgrd"; EXP="exp"; LR=3e-4
while getopts "c:t:e:l:" opt; do
  case $opt in c) GPU=$OPTARG ;; t) TASK=$OPTARG ;; e) EXP=$OPTARG ;; l) LR=$OPTARG ;; esac
done
for SEED in 42 123 999; do
  echo ""
  echo "========================================================"
  echo "  CNN  task=$TASK  seed=$SEED  lr=$LR"
  echo "========================================================"
  python code/train_cnn.py --task "$TASK" --exp-name "$EXP" \
    --seed "$SEED" --gpu "$GPU" --lr "$LR"
done
echo ""
echo "CNN training done. Check log/ folder."
