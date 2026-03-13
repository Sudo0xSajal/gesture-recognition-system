#!/usr/bin/env bash
# Train ALL THREE models: CNN, LSTM (RNN), and SVM.
# Usage: bash train_all.sh -c 0 -t hgrd -e exp1 -l 3e-4
source "$(dirname "$0")/env.sh"
GPU=0; TASK="hgrd"; EXP="exp"; LR=3e-4
while getopts "c:t:e:l:" opt; do
  case $opt in c) GPU=$OPTARG ;; t) TASK=$OPTARG ;; e) EXP=$OPTARG ;; l) LR=$OPTARG ;; esac
done
echo ""
echo "###############################################"
echo "#  Training ALL 3 Models (Ma'am's Step 3)    #"
echo "#  1. CNN   2. LSTM/RNN   3. SVM             #"
echo "###############################################"
echo ""
echo "--- STEP 1/3: CNN ---"
bash "$(dirname "$0")/train_cnn.sh"  -c "$GPU" -t "$TASK" -e "$EXP" -l "$LR"
echo ""
echo "--- STEP 2/3: LSTM/RNN ---"
bash "$(dirname "$0")/train_lstm.sh" -c "$GPU" -t "$TASK" -e "$EXP" -l "$LR"
echo ""
echo "--- STEP 3/3: SVM ---"
bash "$(dirname "$0")/train_svm.sh"  -t "$TASK" -e "$EXP"
echo ""
echo "###############################################"
echo "#  All 3 models trained. Check log/ folder.  #"
echo "###############################################"
