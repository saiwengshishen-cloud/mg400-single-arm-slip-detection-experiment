"""
Kaggle CNN training code for flat 90-trial tactile image dataset.

Dataset format:
    /kaggle/input/datasets/shishensaiweng/kaggle-flat-90trials-images-and-labels/
        flat_training_labels.csv
        trial_001_repeat_01_center_x_pos_slide_sequence__frame_02.png
        trial_001_repeat_01_center_x_pos_slide_sequence__frame_03.png
        ...

CSV columns:
    image_file,trial_folder,frame_id,phase,phase_name,split

Labels:
    0 = stable
    1 = incipient_slip
    2 = translational_slip
"""

from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# -----------------------------
# 1. Paths
# -----------------------------

DATA_DIR = Path(
    "/kaggle/input/datasets/shishensaiweng/"
    "kaggle-flat-90trials-images-and-labels"
)

GENERATED_LABEL_CSV = Path(
    "/kaggle/working/generated_rigid_residual_labels_from_images_only/"
    "generated_training_labels_excluding_frame01.csv"
)
ANNOTATION_CSV = GENERATED_LABEL_CSV if GENERATED_LABEL_CSV.exists() else DATA_DIR / "flat_training_labels.csv"

print("DATA_DIR exists:", DATA_DIR.exists())
print("ANNOTATION_CSV exists:", ANNOTATION_CSV.exists())


# -----------------------------
# 2. Load labels
# -----------------------------

df = pd.read_csv(ANNOTATION_CSV)

print(df.head())
print("Rows:", len(df))
print("Split counts:")
print(df["split"].value_counts())
print("Phase counts:")
print(df["phase"].value_counts().sort_index())

label_names = {
    0: "stable",
    1: "incipient_slip",
    2: "translational_slip",
}

train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df = df[df["split"] == "val"].reset_index(drop=True)

print("Train rows:", len(train_df))
print("Val rows:", len(val_df))
print("Train trials:", train_df["trial_folder"].nunique())
print("Val trials:", val_df["trial_folder"].nunique())


# -----------------------------
# 3. Dataset
# -----------------------------

class SlipFlatFrameDataset(Dataset):
    def __init__(self, annotation_df, image_dir, transform=None):
        self.df = annotation_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_file = row["image_file"]
        label = int(row["phase"])

        img_path = self.image_dir / image_file
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label


# 640x480 original images. Resize to 512x384 keeps the 4:3 ratio and preserves
# marker detail better than 224x224.
transform_train = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.RandomRotation(degrees=3),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

transform_val = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset = SlipFlatFrameDataset(train_df, DATA_DIR, transform_train)
val_dataset = SlipFlatFrameDataset(val_df, DATA_DIR, transform_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

images, labels = next(iter(train_loader))
print("Batch image shape:", images.shape)
print("Batch labels:", labels)


# -----------------------------
# 4. Device
# -----------------------------

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    print("CUDA GPU:", gpu_name)
    print("CUDA capability:", f"sm_{major}{minor}")

    # Kaggle sometimes gives Tesla P100 (sm_60), which may be incompatible
    # with newer PyTorch CUDA builds. Use CPU in that case.
    if major >= 7:
        device = torch.device("cuda")
    else:
        print("GPU is not compatible with this PyTorch CUDA build. Using CPU.")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

print("Device:", device)


# -----------------------------
# 5. Model: ResNet18
# -----------------------------

# weights=None avoids internet download. If Kaggle internet is enabled, you can use:
# model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)


# -----------------------------
# 6. Loss and optimizer
# -----------------------------

# Handle class imbalance.
class_counts = train_df["phase"].value_counts().sort_index()
weights = []
for class_id in [0, 1, 2]:
    weights.append(1.0 / class_counts.get(class_id, 1))
weights = torch.tensor(weights, dtype=torch.float32)
weights = weights / weights.sum() * 3
weights = weights.to(device)

print("Class weights:", weights.detach().cpu().numpy())

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# -----------------------------
# 7. Train and evaluate
# -----------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(3, 3, dtype=torch.long)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
                confusion[true_label, pred_label] += 1

    return total_loss / total, correct / total, confusion


num_epochs = 20
best_val_acc = 0.0
best_path = "/kaggle/working/best_resnet18_flat_slip_phase.pth"

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    val_loss, val_acc, confusion = evaluate(
        model, val_loader, criterion, device
    )

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
    )
    print("Confusion matrix rows=true, cols=pred:")
    print(confusion)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_names": label_names,
                "image_size": (384, 512),
                "best_val_acc": best_val_acc,
            },
            best_path,
        )
        print("Saved best model:", best_path)


print("Best Val Acc:", best_val_acc)


# -----------------------------
# 8. Predict one image
# -----------------------------

def predict_image(image_file):
    model.eval()
    img_path = DATA_DIR / image_file
    image = Image.open(img_path).convert("RGB")
    image = transform_val(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)[0].cpu()
        pred = int(probs.argmax().item())

    return pred, label_names[pred], probs.numpy()


sample_image = val_df.iloc[0]["image_file"]
pred_id, pred_name, probs = predict_image(sample_image)
print("Sample prediction:", sample_image, pred_id, pred_name, probs)
