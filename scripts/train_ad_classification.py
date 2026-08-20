import argparse
import csv
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from qssm.models.encoder_3d import QSSM3DEncoder


class ADDataset(Dataset):
    """
    CSV format:

    image_path,label
    subject001.nii.gz,0
    subject002.nii.gz,1

    0 = Control
    1 = Alzheimer's disease
    """

    def __init__(self, samples, data_root=None):
        self.samples = samples
        self.data_root = Path(data_root) if data_root else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        if self.data_root is not None:
            image_path = self.data_root / image_path

        image = nib.load(str(image_path)).get_fdata(dtype=np.float32)

        if image.shape != (160, 160, 160):
            raise ValueError(
                f"{image_path}: expected shape (160,160,160), "
                f"got {image.shape}"
            )

        image = (image - image.mean()) / (image.std() + 1e-8)

        image = torch.from_numpy(image).float().unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return image, label


class QSSM3DClassifier(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()

        self.encoder = QSSM3DEncoder(
            embed_dim=embed_dim
        )

        self.head = nn.Linear(
            embed_dim,
            2,
        )

    def forward(self, x):
        features = self.encoder(x)

        features = features.mean(dim=1)

        logits = self.head(features)

        return logits


def read_csv(csv_file):
    samples = []

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            samples.append(
                (
                    row["image_path"],
                    int(row["label"]),
                )
            )

    return samples


def make_folds(samples, n_folds=5, seed=42):
    indices = list(range(len(samples)))

    random.Random(seed).shuffle(indices)

    return np.array_split(
        indices,
        n_folds,
    )


def train_epoch(model, loader, optimizer, device):
    model.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return correct / total


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="checkpoints/ad")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    samples = read_csv(args.csv)

    folds = make_folds(samples)

    results = []

    for fold in range(5):

        val_indices = set(
            folds[fold].tolist()
        )

        train_samples = [
            samples[i]
            for i in range(len(samples))
            if i not in val_indices
        ]

        val_samples = [
            samples[i]
            for i in folds[fold]
        ]

        train_loader = DataLoader(
            ADDataset(
                train_samples,
                args.data_root,
            ),
            batch_size=args.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            ADDataset(
                val_samples,
                args.data_root,
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )

        model = QSSM3DClassifier().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.05,
        )

        best_accuracy = 0.0

        for epoch in range(args.epochs):

            loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
            )

            accuracy = evaluate(
                model,
                val_loader,
                device,
            )

            print(
                f"Fold {fold + 1} | "
                f"Epoch {epoch + 1} | "
                f"Loss {loss:.4f} | "
                f"Accuracy {accuracy:.4f}"
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy

        results.append(best_accuracy)

    print(
        "Mean accuracy:",
        np.mean(results),
    )


if __name__ == "__main__":
    main()
