from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# 1. Kaggle paths
# ============================================================

DATA_DIR = Path(
    "/kaggle/input/datasets/shishensaiweng/kaggle-flat-90trials-images-and-labels"
)
GENERATED_LABEL_CSV = Path(
    "/kaggle/working/generated_rigid_residual_labels_from_images_only/"
    "generated_training_labels_excluding_frame01.csv"
)
ANNOTATION_CSV = GENERATED_LABEL_CSV if GENERATED_LABEL_CSV.exists() else DATA_DIR / "flat_training_labels.csv"
OUTPUT_DIR = Path("/kaggle/working")

# Upload your previous CNN model to Kaggle as a dataset.
# The script will search for this file under /kaggle/input.
CNN_MODEL_NAME = "best_resnet18_flat_slip_phase.pth"

NUM_CLASSES = 3
IMAGE_SIZE = (384, 512)
BATCH_SIZE_FRAMES = 32


PHASE_NAMES = {
    0: "stable",
    1: "incipient_slip",
    2: "translational_slip",
}


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
# 3. Dataset for individual frames
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


class FrameDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, data_dir: Path, transform=None):
        self.df = data_frame.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.data_dir / row["image_file"]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, idx


# ============================================================
# 4. Load previous CNN and remove classification layer
# ============================================================

def find_cnn_model_path() -> Path | None:
    exact_matches = list(Path("/kaggle/input").rglob(CNN_MODEL_NAME))
    if exact_matches:
        return exact_matches[0]

    pth_files = list(Path("/kaggle/input").rglob("*.pth"))
    if pth_files:
        return pth_files[0]

    return None


def build_resnet18_classifier() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_cnn_classifier() -> nn.Module:
    model_path = find_cnn_model_path()

    if model_path is None:
        raise FileNotFoundError(
            "No .pth CNN model was found under /kaggle/input. "
            "Please upload best_resnet18_flat_slip_phase.pth as a Kaggle dataset."
        )

    print("Loading CNN model from:", model_path)
    model = build_resnet18_classifier()
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as exc:
        print("Default torch.load failed:", repr(exc))
        print("Retrying with weights_only=False because this is your own trusted CNN checkpoint.")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

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

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    return model


class ResNet18FeatureExtractor(nn.Module):
    """
    ResNet18 without the final fully connected classification layer.

    Input:
        [N, 3, H, W]

    Output:
        [N, 512]
    """

    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.features = nn.Sequential(*list(classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


# ============================================================
# 5. Extract features
# ============================================================

print("DATA_DIR exists:", DATA_DIR.exists())
print("ANNOTATION_CSV exists:", ANNOTATION_CSV.exists())

if not DATA_DIR.exists():
    raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")
if not ANNOTATION_CSV.exists():
    raise FileNotFoundError(f"ANNOTATION_CSV does not exist: {ANNOTATION_CSV}")

df = pd.read_csv(ANNOTATION_CSV)
required_columns = {"image_file", "trial_folder", "frame_id", "phase", "split"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns in annotation CSV: {sorted(missing_columns)}")

df["frame_number"] = df["frame_id"].str.extract(r"(\d+)").astype(int)
df = df.sort_values(["trial_folder", "frame_number"]).reset_index(drop=True)

print(df.head())
print("Rows:", len(df))
print("Trials:", df["trial_folder"].nunique())
print("Split counts:")
print(df["split"].value_counts())
print("Phase counts:")
print(df["phase"].value_counts().sort_index())

frame_dataset = FrameDataset(df, DATA_DIR, image_transform)
frame_loader = DataLoader(
    frame_dataset,
    batch_size=BATCH_SIZE_FRAMES,
    shuffle=False,
    num_workers=2,
)

cnn_classifier = load_cnn_classifier()
feature_extractor = ResNet18FeatureExtractor(cnn_classifier).to(device)
feature_extractor.eval()

all_features = np.zeros((len(df), 512), dtype=np.float32)

with torch.no_grad():
    for images, indices in frame_loader:
        images = images.to(device)
        features = feature_extractor(images)
        all_features[indices.numpy()] = features.detach().cpu().numpy()

print("All frame feature shape:", all_features.shape)


# ============================================================
# 6. Build trial-level feature matrix
# ============================================================

trial_names = sorted(df["trial_folder"].unique().tolist())
sequence_lengths = []

for trial_name in trial_names:
    sequence_lengths.append(len(df[df["trial_folder"] == trial_name]))

print("Sequence lengths:", sorted(set(sequence_lengths)))

if len(set(sequence_lengths)) != 1:
    raise ValueError(
        "Not all trials have the same sequence length. "
        "This script expects frame_02 ... frame_40 for every trial."
    )

num_trials = len(trial_names)
seq_len = sequence_lengths[0]
feature_dim = all_features.shape[1]

feature_matrix = np.zeros((num_trials, seq_len, feature_dim), dtype=np.float32)
label_matrix = np.zeros((num_trials, seq_len), dtype=np.int64)
frame_id_matrix = []
split_list = []

for trial_idx, trial_name in enumerate(trial_names):
    trial_rows = df[df["trial_folder"] == trial_name].sort_values("frame_number")
    row_indices = trial_rows.index.to_numpy()

    feature_matrix[trial_idx] = all_features[row_indices]
    label_matrix[trial_idx] = trial_rows["phase"].to_numpy(dtype=np.int64)
    frame_id_matrix.append(trial_rows["frame_id"].tolist())
    split_list.append(trial_rows["split"].iloc[0])

print("CNN feature matrix shape:", feature_matrix.shape)
print("Label matrix shape:", label_matrix.shape)

expected_shape = (90, 39, 512)
if feature_matrix.shape != expected_shape:
    print(f"Warning: expected feature shape {expected_shape}, got {feature_matrix.shape}")


# ============================================================
# 7. Save outputs
# ============================================================

np.save(OUTPUT_DIR / "cnn_feature_matrix.npy", feature_matrix)
np.save(OUTPUT_DIR / "cnn_label_matrix.npy", label_matrix)

trial_info = pd.DataFrame(
    {
        "trial_index": np.arange(num_trials),
        "trial_folder": trial_names,
        "split": split_list,
        "sequence_length": seq_len,
        "frame_ids": [";".join(ids) for ids in frame_id_matrix],
    }
)
trial_info.to_csv(OUTPUT_DIR / "cnn_feature_trial_info.csv", index=False)

metadata = {
    "feature_matrix": "cnn_feature_matrix.npy",
    "label_matrix": "cnn_label_matrix.npy",
    "trial_info": "cnn_feature_trial_info.csv",
    "shape": list(feature_matrix.shape),
    "meaning": "[num_trials, sequence_length, cnn_feature_dim]",
    "phase_names": PHASE_NAMES,
    "cnn_model_name": CNN_MODEL_NAME,
    "image_size": list(IMAGE_SIZE),
}

with (OUTPUT_DIR / "cnn_feature_metadata.json").open("w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("\nSaved:")
print(OUTPUT_DIR / "cnn_feature_matrix.npy")
print(OUTPUT_DIR / "cnn_label_matrix.npy")
print(OUTPUT_DIR / "cnn_feature_trial_info.csv")
print(OUTPUT_DIR / "cnn_feature_metadata.json")

print("\nUse these files as input for the LSTM training script.")
