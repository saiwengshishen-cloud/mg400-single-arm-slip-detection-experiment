from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
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

# If you upload the CNN model to Kaggle as a dataset, this function will try
# to find it automatically under /kaggle/input.
CNN_MODEL_NAME = "best_resnet18_flat_slip_phase.pth"


PHASE_NAMES = {
    0: "stable",
    1: "incipient_slip",
    2: "translational_slip",
}

NUM_CLASSES = 3
IMAGE_SIZE = (384, 512)
BATCH_SIZE_TRIALS = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LSTM_LAYERS = 1
FREEZE_CNN = True
RANDOM_SEED = 42


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 2. Device
# ============================================================

def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    major, minor = torch.cuda.get_device_capability(0)
    print("CUDA GPU:", torch.cuda.get_device_name(0))
    print(f"CUDA capability: sm_{major}{minor}")

    # Kaggle sometimes gives P100 with a PyTorch build that no longer supports sm_60.
    if major < 7:
        print("This PyTorch build may not support this GPU. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cuda")


device = get_device()
print("Device:", device)


# ============================================================
# 3. Load labels
# ============================================================

print("DATA_DIR exists:", DATA_DIR.exists())
print("ANNOTATION_CSV exists:", ANNOTATION_CSV.exists())

df = pd.read_csv(ANNOTATION_CSV)
df["frame_number"] = df["frame_id"].str.extract(r"(\d+)").astype(int)

print(df.head())
print("Rows:", len(df))
print("Split counts:")
print(df["split"].value_counts())
print("Phase counts:")
print(df["phase"].value_counts().sort_index())
print("Train trials:", df[df["split"] == "train"]["trial_folder"].nunique())
print("Val trials:", df[df["split"] == "val"]["trial_folder"].nunique())


# ============================================================
# 4. Image transform
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


# ============================================================
# 5. Trial sequence dataset
# ============================================================

class TrialSequenceDataset(Dataset):
    """
    One sample = one trial sequence.

    Input:
        images: [T, 3, H, W]

    Target:
        labels: [T]

    Since frame_01 was excluded from flat_training_labels.csv, T is normally 39:
        frame_02 ... frame_40
    """

    def __init__(self, data_frame: pd.DataFrame, data_dir: Path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.trials = []

        for trial_folder, trial_df in data_frame.groupby("trial_folder", sort=True):
            trial_df = trial_df.sort_values("frame_number").reset_index(drop=True)
            image_files = trial_df["image_file"].tolist()
            labels = trial_df["phase"].astype(int).tolist()
            frame_ids = trial_df["frame_id"].tolist()

            self.trials.append(
                {
                    "trial_folder": trial_folder,
                    "image_files": image_files,
                    "labels": labels,
                    "frame_ids": frame_ids,
                }
            )

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        item = self.trials[idx]
        images = []

        for image_file in item["image_files"]:
            image_path = self.data_dir / image_file
            img = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images, dim=0)
        labels = torch.tensor(item["labels"], dtype=torch.long)

        return {
            "images": images,
            "labels": labels,
            "trial_folder": item["trial_folder"],
            "frame_ids": item["frame_ids"],
        }


def collate_trials(batch):
    """
    All trials should have the same length here. This collate function still
    keeps the code explicit and easy to extend if sequence length changes.
    """
    images = torch.stack([item["images"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    trial_folders = [item["trial_folder"] for item in batch]
    frame_ids = [item["frame_ids"] for item in batch]

    return {
        "images": images,
        "labels": labels,
        "trial_folders": trial_folders,
        "frame_ids": frame_ids,
    }


train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()

train_dataset = TrialSequenceDataset(train_df, DATA_DIR, image_transform)
val_dataset = TrialSequenceDataset(val_df, DATA_DIR, image_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_TRIALS,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_trials,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE_TRIALS,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_trials,
)

sample_batch = next(iter(train_loader))
print("Trial batch image shape:", sample_batch["images"].shape)
print("Trial batch label shape:", sample_batch["labels"].shape)
print("Example trial:", sample_batch["trial_folders"][0])
print("Example labels:", sample_batch["labels"][0])


# ============================================================
# 6. Load previous CNN and convert it to feature extractor
# ============================================================

def find_cnn_model_path() -> Path | None:
    candidates = list(Path("/kaggle/input").rglob(CNN_MODEL_NAME))
    if candidates:
        return candidates[0]
    candidates = list(Path("/kaggle/input").rglob("*.pth"))
    if candidates:
        return candidates[0]
    return None


def build_resnet18_classifier() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_cnn_classifier() -> nn.Module:
    model = build_resnet18_classifier()
    model_path = find_cnn_model_path()

    if model_path is None:
        print("No CNN .pth file found under /kaggle/input.")
        print("Using ImageNet-initialized ResNet18 feature extractor instead.")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model

    print("Loading CNN model from:", model_path)
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove possible DataParallel prefix.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.replace("module.", "")
        cleaned_state_dict[cleaned_key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model


class CNNFeatureExtractor(nn.Module):
    """
    ResNet18 without the final classification layer.

    Output:
        feature vector with shape [N, 512]
    """

    def __init__(self, cnn_classifier: nn.Module):
        super().__init__()
        self.features = nn.Sequential(*list(cnn_classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


# ============================================================
# 7. CNN + LSTM model
# ============================================================

class CNNLSTMFrameClassifier(nn.Module):
    """
    Many-to-many sequence classifier.

    Input:
        images: [B, T, 3, H, W]

    Output:
        logits: [B, T, 3]

    This means the model predicts stable / incipient / translational for
    every frame in the trial sequence.
    """

    def __init__(
        self,
        cnn_feature_extractor: nn.Module,
        feature_dim: int = 512,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 3,
        freeze_cnn: bool = True,
    ):
        super().__init__()
        self.cnn = cnn_feature_extractor

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

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

    def forward(self, images):
        batch_size, seq_len, channels, height, width = images.shape

        flat_images = images.view(batch_size * seq_len, channels, height, width)
        features = self.cnn(flat_images)
        features = features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)
        logits = self.classifier(lstm_out)
        return logits


cnn_classifier = load_cnn_classifier()
feature_extractor = CNNFeatureExtractor(cnn_classifier)

model = CNNLSTMFrameClassifier(
    cnn_feature_extractor=feature_extractor,
    feature_dim=512,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LSTM_LAYERS,
    num_classes=NUM_CLASSES,
    freeze_cnn=FREEZE_CNN,
).to(device)


# ============================================================
# 8. Loss, optimizer, metrics
# ============================================================

train_counts = train_df["phase"].value_counts().sort_index()
class_counts = np.asarray([train_counts.get(i, 0) for i in range(NUM_CLASSES)], dtype=np.float32)
class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
class_weights = class_weights / class_weights.mean()
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Class counts:", class_counts)
print("Class weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)


def run_one_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_true = []
    all_pred = []

    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits.view(-1, NUM_CLASSES), labels.view(-1))

            if is_train:
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=-1)

        total_loss += float(loss.item()) * labels.numel()
        total_correct += int((preds == labels).sum().item())
        total_count += int(labels.numel())

        all_true.extend(labels.detach().cpu().view(-1).tolist())
        all_pred.extend(preds.detach().cpu().view(-1).tolist())

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2])
    return avg_loss, acc, cm


# ============================================================
# 9. Train LSTM
# ============================================================

best_val_acc = -1.0
best_model_path = OUTPUT_DIR / "best_cnn_lstm_slip_phase.pth"

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc, train_cm = run_one_epoch(model, train_loader, optimizer)
    val_loss, val_acc, val_cm = run_one_epoch(model, val_loader, optimizer=None)

    print(
        f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
    )
    print("Val confusion matrix rows=true, cols=pred:")
    print(val_cm)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cnn_model_name": CNN_MODEL_NAME,
                "phase_names": PHASE_NAMES,
                "image_size": IMAGE_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_lstm_layers": NUM_LSTM_LAYERS,
                "freeze_cnn": FREEZE_CNN,
                "best_val_acc": best_val_acc,
            },
            best_model_path,
        )
        print("Saved best model:", best_model_path)

print("Best Val Acc:", best_val_acc)


# ============================================================
# 10. Save frame-level validation predictions
# ============================================================

model.eval()
prediction_rows = []

with torch.no_grad():
    for batch in val_loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        for b, trial_folder in enumerate(batch["trial_folders"]):
            for t, frame_id in enumerate(batch["frame_ids"][b]):
                true_phase = int(labels[b, t].item())
                pred_phase = int(preds[b, t].item())
                prob_values = probs[b, t].detach().cpu().numpy()

                prediction_rows.append(
                    {
                        "trial_folder": trial_folder,
                        "frame_id": frame_id,
                        "true_phase": true_phase,
                        "true_phase_name": PHASE_NAMES[true_phase],
                        "pred_phase": pred_phase,
                        "pred_phase_name": PHASE_NAMES[pred_phase],
                        "prob_stable": float(prob_values[0]),
                        "prob_incipient_slip": float(prob_values[1]),
                        "prob_translational_slip": float(prob_values[2]),
                    }
                )

prediction_df = pd.DataFrame(prediction_rows)
prediction_path = OUTPUT_DIR / "cnn_lstm_val_predictions.csv"
prediction_df.to_csv(prediction_path, index=False)
print("Saved validation predictions to:", prediction_path)

print("\nExample sequence prediction:")
example_trial = prediction_df["trial_folder"].iloc[0]
display_cols = [
    "trial_folder",
    "frame_id",
    "true_phase_name",
    "pred_phase_name",
    "prob_stable",
    "prob_incipient_slip",
    "prob_translational_slip",
]
print(prediction_df[prediction_df["trial_folder"] == example_trial][display_cols].head(15))
