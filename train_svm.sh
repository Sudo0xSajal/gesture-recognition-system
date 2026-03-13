#!/usr/bin/env bash
# Train SVM gesture model.
# Usage: bash train_svm.sh -t hgrd -e exp1 [-k rbf] [-C 10]
source "$(dirname "$0")/env.sh"
TASK="hgrd"; EXP="exp"; KERNEL="rbf"; C_VAL=10
while getopts "t:e:k:C:" opt; do
  case $opt in t) TASK=$OPTARG ;; e) EXP=$OPTARG ;; k) KERNEL=$OPTARG ;; C) C_VAL=$OPTARG ;; esac
done
echo ""
echo "========================================================"
echo "  SVM  task=$TASK  kernel=$KERNEL  C=$C_VAL"
echo "========================================================"
python code/train_svm.py --task "$TASK" --exp-name "$EXP" \
  --kernel "$KERNEL" --C "$C_VAL"
echo ""
echo "SVM training done. Check log/ folder."
