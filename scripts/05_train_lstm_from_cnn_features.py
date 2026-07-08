from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. Paths
# ============================================================

WORKING_DIR = Path("/kaggle/working")

FEATURE_MATRIX_PATH = WORKING_DIR / "cnn_feature_matrix.npy"
LABEL_MATRIX_PATH = WORKING_DIR / "cnn_label_matrix.npy"
TRIAL_INFO_PATH = WORKING_DIR / "cnn_feature_trial_info.csv"
OUTPUT_DIR = WORKING_DIR


PHASE_NAMES = {
    0: "stable",
    1: "incipient_slip",
    2: "translational_slip",
}

NUM_CLASSES = 3
FEATURE_DIM = 512
HIDDEN_SIZE = 128
NUM_LSTM_LAYERS = 1
BATCH_SIZE_TRIALS = 8
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
RANDOM_SEED = 42


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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
# 2. Load CNN feature matrix
# ============================================================

for path in [FEATURE_MATRIX_PATH, LABEL_MATRIX_PATH, TRIAL_INFO_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}. "
            "Run kaggle_step1_extract_cnn_feature_matrix.py first in the same Kaggle session."
        )

features = np.load(FEATURE_MATRIX_PATH).astype(np.float32)
labels = np.load(LABEL_MATRIX_PATH).astype(np.int64)
trial_info = pd.read_csv(TRIAL_INFO_PATH)

print("features shape:", features.shape)
print("labels shape:", labels.shape)
print(trial_info.head())
print("Split counts:")
print(trial_info["split"].value_counts())

assert features.ndim == 3, "features should be [num_trials, sequence_length, 512]"
assert labels.ndim == 2, "labels should be [num_trials, sequence_length]"
assert features.shape[:2] == labels.shape
assert features.shape[2] == FEATURE_DIM, f"Expected feature dim {FEATURE_DIM}, got {features.shape[2]}"

required_columns = {"trial_index", "trial_folder", "split", "sequence_length", "frame_ids"}
missing_columns = required_columns - set(trial_info.columns)
if missing_columns:
    raise ValueError(f"Missing required columns in trial info CSV: {sorted(missing_columns)}")

train_indices_for_scaling = trial_info.index[trial_info["split"].eq("train")].to_numpy()
if len(train_indices_for_scaling) == 0:
    raise ValueError("No train trials found in trial_info['split'].")

# Normalize CNN features using train-set statistics only.
# Shape remains [num_trials, sequence_length, 512].
train_feature_pool = features[train_indices_for_scaling].reshape(-1, FEATURE_DIM)
feature_mean = train_feature_pool.mean(axis=0, keepdims=True).astype(np.float32)
feature_std = train_feature_pool.std(axis=0, keepdims=True).astype(np.float32)
feature_std = np.maximum(feature_std, 1e-6)
features = ((features - feature_mean.reshape(1, 1, FEATURE_DIM)) / feature_std.reshape(1, 1, FEATURE_DIM)).astype(
    np.float32
)

print("Features normalized using train-set mean/std.")


# ============================================================
# 3. Dataset
# ============================================================

class FeatureSequenceDataset(Dataset):
    def __init__(
        self,
        feature_matrix: np.ndarray,
        label_matrix: np.ndarray,
        info_df: pd.DataFrame,
        split: str,
    ):
        self.indices = info_df.index[info_df["split"].eq(split)].to_numpy()
        self.feature_matrix = feature_matrix
        self.label_matrix = label_matrix
        self.info_df = info_df.reset_index(drop=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        trial_idx = int(self.indices[idx])
        x = torch.tensor(self.feature_matrix[trial_idx], dtype=torch.float32)
        y = torch.tensor(self.label_matrix[trial_idx], dtype=torch.long)
        trial_folder = self.info_df.loc[trial_idx, "trial_folder"]
        frame_ids = self.info_df.loc[trial_idx, "frame_ids"].split(";")
        return x, y, trial_folder, frame_ids


def collate_feature_sequences(batch):
    x = torch.stack([item[0] for item in batch], dim=0)
    y = torch.stack([item[1] for item in batch], dim=0)
    trial_folders = [item[2] for item in batch]
    frame_ids = [item[3] for item in batch]
    return x, y, trial_folders, frame_ids


train_dataset = FeatureSequenceDataset(features, labels, trial_info, split="train")
val_dataset = FeatureSequenceDataset(features, labels, trial_info, split="val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_TRIALS,
    shuffle=True,
    collate_fn=collate_feature_sequences,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE_TRIALS,
    shuffle=False,
    collate_fn=collate_feature_sequences,
)

sample_x, sample_y, sample_trials, _ = next(iter(train_loader))
print("Batch feature shape:", sample_x.shape)
print("Batch label shape:", sample_y.shape)
print("Example trial:", sample_trials[0])


# ============================================================
# 4. LSTM model
# ============================================================

class LSTMFrameClassifier(nn.Module):
    """
    Input:
        [B, T, 512]

    Output:
        [B, T, 3]
    """

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


model = LSTMFrameClassifier(
    feature_dim=FEATURE_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LSTM_LAYERS,
    num_classes=NUM_CLASSES,
).to(device)


# ============================================================
# 5. Loss and optimizer
# ============================================================

train_indices = trial_info.index[trial_info["split"].eq("train")].to_numpy()
train_labels_flat = labels[train_indices].reshape(-1)
class_counts = np.asarray([(train_labels_flat == i).sum() for i in range(NUM_CLASSES)], dtype=np.float32)
class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
class_weights = class_weights / class_weights.mean()
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Class counts:", class_counts)
print("Class weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)


def run_one_epoch(loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_true = []
    all_pred = []

    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))

            if is_train:
                loss.backward()
                optimizer.step()

        pred = logits.argmax(dim=-1)
        total_loss += float(loss.item()) * y.numel()
        total_correct += int((pred == y).sum().item())
        total_count += int(y.numel())

        all_true.extend(y.detach().cpu().reshape(-1).tolist())
        all_pred.extend(pred.detach().cpu().reshape(-1).tolist())

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2])
    return avg_loss, acc, cm


# ============================================================
# 6. Train
# ============================================================

best_val_acc = -1.0
best_model_path = OUTPUT_DIR / "best_lstm_from_cnn_features.pth"

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc, _ = run_one_epoch(train_loader, optimizer)
    val_loss, val_acc, val_cm = run_one_epoch(val_loader, optimizer=None)

    print(
        f"Epoch [{epoch:03d}/{NUM_EPOCHS}] "
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
                "feature_dim": FEATURE_DIM,
                "hidden_size": HIDDEN_SIZE,
                "num_lstm_layers": NUM_LSTM_LAYERS,
                "phase_names": PHASE_NAMES,
                "best_val_acc": best_val_acc,
                "feature_mean": torch.tensor(feature_mean.squeeze(0), dtype=torch.float32),
                "feature_std": torch.tensor(feature_std.squeeze(0), dtype=torch.float32),
            },
            best_model_path,
        )
        print("Saved best model:", best_model_path)

print("Best Val Acc:", best_val_acc)


# ============================================================
# 7. Save validation predictions
# ============================================================

model.eval()
prediction_rows = []

with torch.no_grad():
    for x, y, trial_folders, frame_ids_batch in val_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)

        for b, trial_folder in enumerate(trial_folders):
            for t, frame_id in enumerate(frame_ids_batch[b]):
                true_phase = int(y[b, t].item())
                pred_phase = int(pred[b, t].item())
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
prediction_path = OUTPUT_DIR / "lstm_from_cnn_features_val_predictions.csv"
prediction_df.to_csv(prediction_path, index=False)

print("Saved validation predictions to:", prediction_path)
print("\nExample prediction table:")
print(prediction_df.head(20))
