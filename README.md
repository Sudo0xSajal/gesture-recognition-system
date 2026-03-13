# AI-Driven Hand Gesture Recognition System
### For Bedridden Patient Assistance

---

## System Architecture (Ma'am's Document — 7 Steps)

```
Bedridden Patient (Hand Gesture / Body Movement)
        ↓
Step 1: Camera / Vision Sensor  (RGB Camera — OpenCV)
        ↓
Step 2: Hand & Body Landmark Detection
        MediaPipe — 21 Hand Landmarks per frame
        OpenCV    — Image processing
        ↓
Step 3: Gesture Recognition AI Model
        CNN   — Convolutional Neural Network
        RNN   — Recurrent Neural Network (LSTM)
        SVM   — Support Vector Machine
        ↓
Step 4: Feature Extraction + Body Language Understanding
        ↓
Step 5: Decision System
        Gesture → Command → Robot Action
        ↓
        |                    |                    |
        v                    v                    v
  Robot Response       IoT Control          Alert System
  Voice Output         Light/Fan/Bed        Caregiver App
```

---

## Quick Start

```bash
# 1. Unzip
unzip gesture_recognition_system_FINAL.zip && cd grs_final

# 2. Create Python environment
conda create -n gesture_env python=3.10 && conda activate gesture_env
pip install -r requirements.txt

# 3. MUST run in every terminal before any command
source env.sh

# 4. Collect gesture data
bash collect.sh -p p001

# 5. Preprocess (extract landmarks + build splits)
bash preprocess.sh -t hgrd

# 6. Train ALL THREE models
bash train_all.sh -c 0 -t hgrd -e exp1 -l 3e-4

# 7. Evaluate
bash evaluateAll.sh -t hgrd

# 8. Run live webcam inference
bash realtime.sh -c log/exp1_s42_cnn_hgrd/best_model.pt -m cnn
```

---

## Step 3 — Three AI Models

| Model | Algorithm | Input | Accuracy | Latency |
|-------|-----------|-------|----------|---------|
| `cnn` | Convolutional Neural Network | 224x224 image | 97-98% | 50-70ms |
| `lstm` | Recurrent Neural Network (LSTM) | 15-frame sequence | 95-97% | 30-50ms |
| `svm` | Support Vector Machine RBF | 63-dim landmarks | 94-96% | ~1ms |

---

## 25 Gesture Dataset (Ma'am's Table)

| ID | Gesture | Meaning | IoT Command |
|----|---------|---------|-------------|
| 0 | Thumb Up | Yes/OK | — |
| 1 | Thumb Down | No | — |
| 2 | Open Palm | Stop | — |
| 3 | Hand Wave | Call someone | call_button |
| 4 | Two Fingers | Attention | — |
| 5 | Hand to Mouth | Need water | water_dispenser |
| 6 | Eating Gesture | Need food | — |
| 7 | Chest Hold | Heart pain | emergency_alarm |
| 8 | Head Touch | Headache | medicine_reminder |
| 9 | Stomach Hold | Stomach pain | — |
| 10 | Arm Point | Arm pain | — |
| 11 | Leg Point | Leg pain | — |
| 12 | Point Up | Light ON | smart_light_on |
| 13 | Point Down | Light OFF | smart_light_off |
| 14 | Circle Motion | Fan ON | fan_on |
| 15 | Palm Rotate | Fan speed | fan_speed |
| 16 | Two Finger Swipe | Change channel | tv_channel |
| 17 | Hand Raise | Need help | call_button |
| 18 | Both Hands Raise | EMERGENCY | emergency_alarm |
| 19 | Hand Shake | Distress | emergency_alarm |
| 20 | Medicine Point | Need medicine | medicine_dispenser |
| 21 | Throat Touch | Breathing problem | emergency_alarm |
| 22 | Pillow Touch | Need rest | smart_bed |
| 23 | Forehead Wipe | Feeling hot | — |
| 24 | Arm Rub | Feeling cold | — |

---

## Shell Scripts

| Script | Usage | What it does |
|--------|-------|-------------|
| `collect.sh` | `-p p001` | Record webcam gesture data |
| `preprocess.sh` | `-t hgrd` | Build train/val/test splits |
| `train_cnn.sh` | `-c 0 -t hgrd -e exp1 -l 3e-4` | Train CNN (3 seeds) |
| `train_lstm.sh` | `-c 0 -t hgrd -e exp1 -l 3e-4` | Train LSTM/RNN (3 seeds) |
| `train_svm.sh` | `-t hgrd -e exp1 -k rbf -C 10` | Train SVM |
| `train_all.sh` | `-c 0 -t hgrd -e exp1 -l 3e-4` | Train CNN + LSTM + SVM |
| `evaluateAll.sh` | `-t hgrd` | Evaluate all trained models |
| `serve.sh` | `-c <ckpt> -m cnn` | REST API server |
| `realtime.sh` | `-c <ckpt> -m cnn` | Live webcam inference |
| `deploy_edge.sh` | `-c <ckpt> -m cnn --host <pi>` | Deploy to Raspberry Pi |

---

## Project Files

```
grs_final/
├── code/
│   ├── utils/config.py         all constants, gesture names, IoT actions
│   ├── utils/loss.py           FocalLoss, LabelSmoothing (supervised only)
│   ├── data/collector.py       Step 1+2: camera + MediaPipe collection
│   ├── data/preprocess.py      landmark extraction + splits
│   ├── data/dataset.py         PyTorch Dataset for CNN/LSTM + numpy for SVM
│   ├── models/cnn_model.py     Step 3: CNN
│   ├── models/lstm_model.py    Step 3: RNN/LSTM
│   ├── models/svm_classifier.py Step 3: SVM
│   ├── train.py                CNN + LSTM training entry
│   ├── train_svm.py            SVM training entry
│   ├── trainer.py              supervised training loop
│   ├── evaluate.py             confusion matrix + reports
│   └── test.py                 single-image inference
├── integrations/
│   ├── api/server.py           Flask REST API (CNN/LSTM/SVM)
│   ├── api/client.py           API test client
│   ├── edge/convert_tflite.py  PyTorch to TFLite
│   ├── edge/deploy_pi.py       SSH deploy to Raspberry Pi
│   └── realtime/run_webcam.py  live webcam inference
└── *.sh                        shell scripts
```
