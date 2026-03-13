"""
config.py — Central configuration for Hand Gesture Recognition System
=====================================================================
Edit the values in GestureConfig to match your setup.
Everything else in the project reads from this one file.
"""

import os


class GestureConfig:
    def __init__(self, mode: str = "train"):
        """
        Args:
            mode: "train" | "eval" | "infer"
                  Controls which paths / behaviours are active.
        """

        # ──────────────────────────────────────────────
        # 1.  DATASET  (raw download from Kaggle)
        # ──────────────────────────────────────────────
        # Root folder that Kaggle unzips to.
        # Expected layout coming FROM Kaggle:
        #   hand-gesture-recognition-dataset/
        #       train/
        #           train/
        #               0/  1.jpg  10.jpg  100.jpg …
        #               1/  …
        #               …
        #       test/
        #           test/
        #               0/  …
        self.raw_dataset_root = "/content/Dataset"

        # ──────────────────────────────────────────────
        # 2.  STRUCTURED DATASET  (output of structure.py)
        # ──────────────────────────────────────────────
        # After structure.py runs, it produces a clean flat layout:
        #   structured_dataset/
        #       train/
        #           0/  …
        #           1/  …
        #       val/
        #           0/  …
        #       test/
        #           0/  …
        self.structured_dir = "/content/structured_dataset"
        self.train_dir      = os.path.join(self.structured_dir, "train")
        self.val_dir        = os.path.join(self.structured_dir, "val")
        self.test_dir       = os.path.join(self.structured_dir, "test")

        # ──────────────────────────────────────────────
        # 3.  PREPROCESSED DATASET  (output of preprocess.py)
        # ──────────────────────────────────────────────
        self.preprocessed_dir = "/content/preprocessed_dataset"

        # ──────────────────────────────────────────────
        # 4.  CLASS / LABEL SETTINGS
        # ──────────────────────────────────────────────
        # How many gesture classes exist in the dataset.
        # The Kaggle hand-gesture dataset has 10 classes (0-9).
        self.num_classes = 20

        # Optional: map numeric folder names → human-readable labels.
        # Leave as None to auto-generate ("class_0", "class_1", …).
        self.class_names = {
            0: "palm",
            1: "l",
            2: "fist",
            3: "fist_moved",
            4: "thumb",
            5: "index",
            6: "ok",
            7: "palm_moved",
            8: "c",
            9: "down",
        }   # set to None if you don't want to rename

        # ──────────────────────────────────────────────
        # 5.  IMAGE PREPROCESSING
        # ──────────────────────────────────────────────
        self.image_size      = (224, 224)   # (height, width) fed to the model
        self.num_channels    = 3            # 3 = RGB, 1 = Grayscale
        self.normalize_mean  = [0.485, 0.456, 0.406]   # ImageNet means (RGB)
        self.normalize_std   = [0.229, 0.224, 0.225]   # ImageNet stds  (RGB)

        # ──────────────────────────────────────────────
        # 6.  DATA SPLIT
        # ──────────────────────────────────────────────
        self.val_split       = 0.15   # fraction of train set used for validation
        self.random_seed     = 42

        # ──────────────────────────────────────────────
        # 7.  MODEL ARCHITECTURE
        # ──────────────────────────────────────────────
        # Backbone: "mobilenetv2" | "resnet18" | "efficientnet_b0" | "custom_cnn"
        self.backbone        = "mobilenetv2"
        self.pretrained      = True          # use ImageNet pretrained weights
        self.dropout_rate    = 0.3

        # ──────────────────────────────────────────────
        # 8.  TRAINING HYPERPARAMETERS
        # ──────────────────────────────────────────────
        self.batch_size          = 32
        self.num_epochs          = 50
        self.learning_rate       = 1e-3
        self.weight_decay        = 1e-4
        self.scheduler           = "cosine"   # "cosine" | "step" | "none"
        self.early_stop_patience = 10         # stop if val-loss doesn't improve

        # ──────────────────────────────────────────────
        # 9.  DATA AUGMENTATION  (training only)
        # ──────────────────────────────────────────────
        self.augment_flip        = True
        self.augment_rotate      = True
        self.augment_rotate_deg  = 15         # ± degrees
        self.augment_brightness  = True
        self.augment_brightness_factor = 0.2

        # ──────────────────────────────────────────────
        # 10. CHECKPOINTS & LOGGING
        # ──────────────────────────────────────────────
        self.checkpoint_dir  = "./checkpoints"
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        self.log_dir         = "./logs"
        self.save_every_n_epochs = 5

        # ──────────────────────────────────────────────
        # 11. INFERENCE / PRODUCTION
        # ──────────────────────────────────────────────
        self.inference_model_path = self.best_model_path
        self.inference_device     = "cpu"     # "cpu" | "cuda" | "mps"
        self.confidence_threshold = 0.6       # predictions below this are "unknown"

        # ──────────────────────────────────────────────
        # 12. DEVICE  (auto-detect if not set)
        # ──────────────────────────────────────────────
        self.device = self._resolve_device(mode)

    # ------------------------------------------------------------------ #
    def _resolve_device(self, mode: str) -> str:
        import torch
        if mode == "infer":
            return self.inference_device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        lines = ["GestureConfig:"]
        for k, v in self.__dict__.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ── quick sanity-check ──────────────────────────────────────────────── #
if __name__ == "__main__":
    cfg = GestureConfig(mode="train")
    print(cfg)