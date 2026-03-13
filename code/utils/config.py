"""
code/utils/config.py
====================
SINGLE SOURCE OF TRUTH for the entire project.

Research Alignment (Ma'am's Document)
--------------------------------------
Step 1  — CAMERA_CONFIG  : RGB camera sensing
Step 2  — MediaPipe      : 21 hand landmarks extracted per frame
Step 3  — MODEL_CONFIG   : CNN, RNN (LSTM), SVM — the three algorithms specified
Step 4  — GESTURE_ACTIONS: Body language → clinical interpretation table
Step 5  — Decision engine: gesture → robot/IoT action
Step 6  — Robot response : alert / IoT / caregiver notification

Training is FULLY SUPERVISED.
Every sample has a class label. CNN and LSTM are trained with PyTorch.
SVM is trained with scikit-learn on 63-dim landmark feature vectors.
No semi-supervised, no contrastive loss, no DHC.
"""

import os
import yaml
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. PROJECT PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LOG_ROOT     = Path(os.environ.get("LOG_ROOT",     PROJECT_ROOT / "log"))


def _resolve_path(path) -> Path:
    """Return *path* as an absolute Path.

    Relative paths are resolved against PROJECT_ROOT so the config works
    regardless of the working directory from which scripts are executed.
    Absolute paths are returned unchanged.
    """
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PATHS  — edit ONLY these two variables to match your environment.
#
# Both accept absolute or relative paths.
# Relative paths are resolved from the project root automatically.
# Directories are NOT created here; the preprocessing pipeline creates them.
# ─────────────────────────────────────────────────────────────────────────────
RAW_DATASET_PATH          = _resolve_path("dataset/raw")
PREPROCESSED_DATASET_PATH = _resolve_path("dataset/preprocessed")

# Derived paths — computed automatically from PREPROCESSED_DATASET_PATH.
# Do NOT edit these directly; change PREPROCESSED_DATASET_PATH above instead.
SPLITS_DIR      = PREPROCESSED_DATASET_PATH / "splits"
TRAIN_SPLIT_DIR = SPLITS_DIR / "train"
VAL_SPLIT_DIR   = SPLITS_DIR / "val"
TEST_SPLIT_DIR  = SPLITS_DIR / "test"

DATASETS = {
    "hgrd":   Path(os.environ.get("HGRD_ROOT",  PROJECT_ROOT / "hand-gesture-recognition-dataset")),
    "custom": Path(os.environ.get("CUSTOM_ROOT", PROJECT_ROOT / "custom_dataset")),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. GESTURE VOCABULARY  (exact 25 gestures from Ma'am's Table)
# ─────────────────────────────────────────────────────────────────────────────
GESTURE_NAMES = {
    0:  "Thumb Up",           # Yes / OK
    1:  "Thumb Down",         # No
    2:  "Open Palm",          # Stop
    3:  "Hand Wave",          # Call someone
    4:  "Two Fingers",        # Attention
    5:  "Hand to Mouth",      # Need water
    6:  "Eating Gesture",     # Need food
    7:  "Chest Hold",         # Heart pain
    8:  "Head Touch",         # Headache
    9:  "Stomach Hold",       # Stomach pain
    10: "Arm Point",          # Arm pain
    11: "Leg Point",          # Leg pain
    12: "Point Up",           # Light ON
    13: "Point Down",         # Light OFF
    14: "Circle Motion",      # Fan ON
    15: "Palm Rotate",        # Fan speed
    16: "Two Finger Swipe",   # Change channel
    17: "Hand Raise",         # Need help
    18: "Both Hands Raise",   # Emergency
    19: "Hand Shake",         # Distress
    20: "Medicine Point",     # Need medicine
    21: "Throat Touch",       # Breathing problem
    22: "Pillow Touch",       # Need rest
    23: "Forehead Wipe",      # Feeling hot
    24: "Arm Rub",            # Feeling cold
}
NUM_CLASSES = len(GESTURE_NAMES)   # 25

GESTURE_IDS = {v: k for k, v in GESTURE_NAMES.items()}

# ─────────────────────────────────────────────────────────────────────────────
# 3. CLINICAL ACTION MAPPING  (Step 4 & 5 — Decision System)
# ─────────────────────────────────────────────────────────────────────────────
GESTURE_ACTIONS = {
    0:  {"type": "confirm",   "message": "Patient confirmed: Yes / OK",                  "alert": False, "iot": None},
    1:  {"type": "cancel",    "message": "Patient said: No / Cancel",                    "alert": False, "iot": None},
    2:  {"type": "stop",      "message": "Patient says: Stop / Wait",                    "alert": False, "iot": None},
    3:  {"type": "call",      "message": "Patient is calling for attention",              "alert": True,  "iot": "call_button"},
    4:  {"type": "attention", "message": "Patient needs attention",                      "alert": True,  "iot": None},
    5:  {"type": "water",     "message": "Patient needs WATER — notify caregiver",       "alert": True,  "iot": "water_dispenser"},
    6:  {"type": "food",      "message": "Patient needs FOOD — notify caregiver",        "alert": True,  "iot": None},
    7:  {"type": "emergency", "message": "CHEST PAIN / HEART ATTACK — EMERGENCY",       "alert": True,  "iot": "emergency_alarm", "priority": "HIGH"},
    8:  {"type": "health",    "message": "Patient has HEADACHE — notify caregiver",      "alert": True,  "iot": "medicine_reminder"},
    9:  {"type": "health",    "message": "Patient has STOMACH PAIN — notify caregiver",  "alert": True,  "iot": None},
    10: {"type": "health",    "message": "Patient reports ARM PAIN — check IV",          "alert": True,  "iot": None},
    11: {"type": "health",    "message": "Patient reports LEG PAIN — check circulation", "alert": True,  "iot": None},
    12: {"type": "iot",       "message": "Light ON command",                              "alert": False, "iot": "smart_light_on"},
    13: {"type": "iot",       "message": "Light OFF command",                             "alert": False, "iot": "smart_light_off"},
    14: {"type": "iot",       "message": "Fan ON command",                                "alert": False, "iot": "fan_on"},
    15: {"type": "iot",       "message": "Fan speed adjust",                              "alert": False, "iot": "fan_speed"},
    16: {"type": "iot",       "message": "Change TV channel",                             "alert": False, "iot": "tv_channel"},
    17: {"type": "call",      "message": "Patient needs HELP — notify caregiver",        "alert": True,  "iot": "call_button"},
    18: {"type": "emergency", "message": "EMERGENCY — CALL DOCTOR IMMEDIATELY",          "alert": True,  "iot": "emergency_alarm", "priority": "HIGH"},
    19: {"type": "distress",  "message": "Patient in DISTRESS — immediate assistance",   "alert": True,  "iot": "emergency_alarm", "priority": "HIGH"},
    20: {"type": "medicine",  "message": "Patient needs MEDICINE — notify nurse",        "alert": True,  "iot": "medicine_dispenser"},
    21: {"type": "emergency", "message": "BREATHING PROBLEM — urgent check needed",      "alert": True,  "iot": "emergency_alarm", "priority": "HIGH"},
    22: {"type": "comfort",   "message": "Patient needs REST — adjust pillow/bed",       "alert": True,  "iot": "smart_bed"},
    23: {"type": "health",    "message": "Patient is HOT / FEVER — check temperature",  "alert": True,  "iot": None},
    24: {"type": "comfort",   "message": "Patient is COLD — adjust temperature",         "alert": True,  "iot": None},
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. CAMERA CONFIG  (Step 1 — Sensing the Human Gesture)
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_CONFIG = {
    "device_id":          0,
    "width":              640,
    "height":             480,
    "fps":                30,
    "autofocus":          True,
    "quality_threshold":  0.60,
    "min_detection_conf": 0.70,
    "min_tracking_conf":  0.50,
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA SPLITS  (participant-level — no data leakage)
# ─────────────────────────────────────────────────────────────────────────────
DATA_SPLITS = {
    "train_ratio": 0.70,
    "val_ratio":   0.15,
    "test_ratio":  0.15,
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING CONFIG
#    CNN  → trained with PyTorch, FocalLoss, AdamW
#    LSTM → trained with PyTorch, FocalLoss, AdamW
#    SVM  → trained with scikit-learn RBF kernel on 63-dim landmark vectors
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_CONFIG = {
    # PyTorch (CNN + LSTM)
    "batch_size":          32,
    "num_workers":         4,
    "pin_memory":          True,
    "max_epochs":          150,
    "early_stop_patience": 20,
    "clip_grad_norm":      1.0,

    # Optimizer
    "optimizer":    "adamw",   # adamw | adam | sgd
    "learning_rate": 3e-4,
    "weight_decay":  1e-4,

    # LR Scheduler
    "scheduler":  "reduce",   # reduce | cosine
    "lr_factor":   0.5,
    "lr_patience":  5,
    "lr_min":      1e-6,

    # Loss (for CNN and LSTM)
    "loss":            "focal",  # focal | label_smooth | ce
    "focal_alpha":      0.25,
    "focal_gamma":      2.0,
    "label_smoothing":  0.1,

    # AMP (CUDA only)
    "use_amp":    True,

    # Checkpoint every N epochs
    "ckpt_every": 10,

    # SVM hyperparameters (used in train_svm.py)
    "svm_kernel":      "rbf",    # rbf | linear | poly
    "svm_C":           10.0,
    "svm_gamma":       "scale",
    "svm_probability":  True,    # enables confidence scores via predict_proba
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. MODEL ARCHITECTURE CONFIG  (Step 3)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    # CNN
    "cnn_base_filters": 32,
    "cnn_input_size":   224,
    "cnn_dropout":      0.4,

    # RNN / LSTM
    "lstm_hidden":  128,
    "lstm_layers":    2,
    "lstm_dropout": 0.3,
    "lstm_seq_len":  15,   # consecutive frames per sample

    # SVM uses 63-dim landmark vector — no model architecture config needed
}

# ─────────────────────────────────────────────────────────────────────────────
# 8. INFERENCE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
INFERENCE_CONFIG = {
    "smoothing_window":  7,
    "smoothing_method":  "majority",  # majority | ewa
    "confidence_thresh": 0.75,
    "device":            "cpu",
}

# ─────────────────────────────────────────────────────────────────────────────
# 9. HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_root(task: str) -> Path:
    if task not in DATASETS:
        raise ValueError(f"Unknown task '{task}'. Choose: {list(DATASETS)}")
    return DATASETS[task]


def get_split_file(task: str, split: str) -> Path:
    """Return path to <dataset>/splits/<split>.txt"""
    return get_dataset_root(task) / "splits" / f"{split}.txt"


def get_log_dir(exp_name: str, model: str, task: str) -> Path:
    d = LOG_ROOT / f"{exp_name}_{model}_{task}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_action(gesture_id: int) -> dict:
    return GESTURE_ACTIONS.get(gesture_id, {
        "type": "unknown", "message": f"Unknown gesture {gesture_id}",
        "alert": False, "iot": None,
    })


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(cfg: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
