from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# 1. Paths you need to edit
# ============================================================

# Put your new trial image folder here.
# It should contain frame_01.png ... frame_40.png.
NEW_TRIAL_DIR = Path(
    "/kaggle/input/your-new-trial-dataset/trial_001_repeat_01_center_x_pos_slide_sequence"
)

# These two model files should exist under /kaggle/input or /kaggle/working.
CNN_MODEL_NAME = "best_resnet18_flat_slip_phase.pth"
LSTM_MODEL_NAME = "best_lstm_from_cnn_features.pth"

OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_CSV = OUTPUT_DIR / "new_trial_cnn_lstm_predictions.csv"


PHASE_NAMES = {
    0: "stable",
    1: "incipient_slip",
    2: "translational_slip",
}

NUM_CLASSES = 3
IMAGE_SIZE = (384, 512)
FEATURE_DIM = 512


# ============================================================
# 2. Device
# ============================================================

def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    major, minor = torch.cuda.get_device_capability(0)
    print("CUDA GPU:", torch.cuda.get_device_name(0))
    print(f"CUDA capability: sm_{major}{minor}")

    if major < 7:
        print("This PyTorch build may not support this GPU. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cuda")


device = get_device()
print("Device:", device)


# ============================================================
# 3. Utilities
# ============================================================

def find_file(filename: str) -> Path:
    search_roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    matches = []
    for root in search_roots:
        if root.exists():
            matches.extend(root.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under /kaggle/input or /kaggle/working. "
            "Please upload the model file or run the training cell first."
        )

    print(f"Found {filename}:", matches[0])
    return matches[0]


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        print("Default torch.load failed:", repr(exc))
        print("Retrying with weights_only=False because this is your own trusted checkpoint.")
        return torch.load(path, map_location="cpu", weights_only=False)


# ============================================================
# 4. CNN feature extractor
# ============================================================

def build_resnet18_classifier() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.features = nn.Sequential(*list(classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def load_cnn_feature_extractor() -> nn.Module:
    cnn_path = find_file(CNN_MODEL_NAME)
    classifier = build_resnet18_classifier()
    checkpoint = safe_torch_load(cnn_path)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.replace("module.", "")
        cleaned_state_dict[cleaned_key] = value

    missing, unexpected = classifier.load_state_dict(cleaned_state_dict, strict=False)
    print("CNN missing keys:", missing)
    print("CNN unexpected keys:", unexpected)

    feature_extractor = ResNet18FeatureExtractor(classifier).to(device)
    feature_extractor.eval()
    return feature_extractor


# ============================================================
# 5. LSTM model
# ============================================================

class LSTMFrameClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out)
        return logits


def load_lstm_model() -> tuple[nn.Module, np.ndarray, np.ndarray]:
    lstm_path = find_file(LSTM_MODEL_NAME)
    checkpoint = safe_torch_load(lstm_path)

    hidden_size = int(checkpoint.get("hidden_size", 128))
    num_layers = int(checkpoint.get("num_lstm_layers", 1))

    model = LSTMFrameClassifier(
        feature_dim=FEATURE_DIM,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=NUM_CLASSES,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    if "feature_mean" not in checkpoint or "feature_std" not in checkpoint:
        raise KeyError(
            "The LSTM checkpoint does not contain feature_mean/feature_std. "
            "Please use the corrected LSTM training script that saves normalization parameters."
        )

    feature_mean = checkpoint["feature_mean"].detach().cpu().numpy().astype(np.float32)
    feature_std = checkpoint["feature_std"].detach().cpu().numpy().astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-6)

    return model, feature_mean, feature_std


# ============================================================
# 6. Load new trial frames
# ============================================================

image_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_new_trial_images(trial_dir: Path) -> tuple[torch.Tensor, list[str]]:
    if not trial_dir.exists():
        raise FileNotFoundError(f"NEW_TRIAL_DIR does not exist: {trial_dir}")

    frame_paths = sorted(trial_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.png files found in: {trial_dir}")

    # Match the training setup: exclude frame_01.
    frame_paths = [p for p in frame_paths if p.stem != "frame_01"]

    if len(frame_paths) != 39:
        print(f"Warning: expected 39 frames after excluding frame_01, got {len(frame_paths)}")

    images = []
    frame_ids = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        images.append(image_transform(image))
        frame_ids.append(path.stem)

    image_tensor = torch.stack(images, dim=0)
    return image_tensor, frame_ids


# ============================================================
# 7. Predict
# ============================================================

cnn_feature_extractor = load_cnn_feature_extractor()
lstm_model, feature_mean, feature_std = load_lstm_model()

images, frame_ids = load_new_trial_images(NEW_TRIAL_DIR)
print("New trial image tensor shape:", images.shape)

with torch.no_grad():
    images = images.to(device)
    cnn_features = cnn_feature_extractor(images)
    cnn_features_np = cnn_features.detach().cpu().numpy().astype(np.float32)

print("CNN feature matrix shape before LSTM:", cnn_features_np.shape)

# Apply the same normalization used during LSTM training.
cnn_features_np = (cnn_features_np - feature_mean.reshape(1, FEATURE_DIM)) / feature_std.reshape(1, FEATURE_DIM)

# LSTM expects [B, T, 512]. Here B=1.
lstm_input = torch.tensor(cnn_features_np, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    logits = lstm_model(lstm_input)
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    pred_phase = probs.argmax(axis=1)

rows = []
for i, frame_id in enumerate(frame_ids):
    rows.append(
        {
            "trial_folder": NEW_TRIAL_DIR.name,
            "frame_id": frame_id,
            "pred_phase": int(pred_phase[i]),
            "pred_phase_name": PHASE_NAMES[int(pred_phase[i])],
            "prob_stable": float(probs[i, 0]),
            "prob_incipient_slip": float(probs[i, 1]),
            "prob_translational_slip": float(probs[i, 2]),
        }
    )

prediction_df = pd.DataFrame(rows)
prediction_df.to_csv(OUTPUT_CSV, index=False)

print("Saved predictions to:", OUTPUT_CSV)
print(prediction_df)

print("\nPhase counts:")
print(prediction_df["pred_phase_name"].value_counts())

