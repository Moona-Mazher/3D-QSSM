import argparse
import csv
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from qssm.models.brain_age_model import QSSM3DBrainAge


class BrainAgeDataset(Dataset):
    """
    CSV format:

    image_path,age
    subject001.nii.gz,45
    subject002.nii.gz,62
    """

    def __init__(self, samples, data_root=None):
        self.samples = samples
        self.data_root = Path(data_root) if data_root else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, age = self.samples[idx]

        if self.data_root is not None:
            image_path = self.data_root / image_path

        image = nib.load(str(image_path)).get_fdata(dtype=np.float32)

        if image.shape != (160, 160, 160):
            raise ValueError(
                f"{image_path}: expected shape (160,160,160), "
                f"got {image.shape}"
            )

        # Volume-wise normalization
        image = (image - image.mean()) / (image.std() + 1e-8)

        image = torch.from_numpy(image).float().unsqueeze(0)
        age = torch.tensor(age, dtype=torch.float32)

        return image, age


def read_csv(csv_file):
    samples = []

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)

        if "image_path" not in reader.fieldnames or "age" not in reader.fieldnames:
            raise ValueError(
                "CSV must contain columns: image_path, age"
            )

        for row in reader:
            samples.append(
                (row["image_path"], float(row["age"]))
            )

    return samples


def make_folds(samples, n_folds=5, seed=42):
    """
    Deterministic 5-fold split.
    """

    indices = list(range(len(samples)))

    rng = random.Random(seed)
    rng.shuffle(indices)

    folds = np.array_split(indices, n_folds)

    return folds


def train_epoch(model, loader, optimizer, device):
    model.train()

    criterion = nn.L1Loss()
    total_loss = 0.0

    for images, ages in loader:
        images = images.to(device)
        ages = ages.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(predictions, ages)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    absolute_errors = []

    for images, ages in loader:
        images = images.to(device)
        ages = ages.to(device)

        predictions = model(images)

        errors = torch.abs(predictions - ages)

        absolute_errors.extend(
            errors.cpu().numpy().tolist()
        )

    return float(np.mean(absolute_errors))


def run_fold(
    fold,
    samples,
    folds,
    args,
    device,
):
    val_indices = set(folds[fold].tolist())

    train_samples = [
        samples[i]
        for i in range(len(samples))
        if i not in val_indices
    ]

    val_samples = [
        samples[i]
        for i in folds[fold]
    ]

    train_dataset = BrainAgeDataset(
        train_samples,
        data_root=args.data_root,
    )

    val_dataset = BrainAgeDataset(
        val_samples,
        data_root=args.data_root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = QSSM3DBrainAge().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_mae = float("inf")

    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(args.epochs):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_mae = evaluate(
            model,
            val_loader,
            device,
        )

        scheduler.step()

        print(
            f"Fold {fold + 1} | "
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train MAE: {train_loss:.3f} | "
            f"Val MAE: {val_mae:.3f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "fold": fold,
                    "val_mae": best_mae,
                },
                checkpoint_dir / f"brain_age_fold{fold + 1}.pt",
            )

    return best_mae


def main():

    parser = argparse.ArgumentParser(
        description="3D-QSSM brain-age fine-tuning"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="CSV containing image_path and age columns",
    )

    parser.add_argument(
        "--data_root",
        default=None,
        help="Optional root directory containing MRI files",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--output_dir",
        default="checkpoints/brain_age",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    samples = read_csv(args.csv)

    print("Subjects:", len(samples))

    folds = make_folds(
        samples,
        n_folds=5,
        seed=args.seed,
    )

    fold_results = []

    for fold in range(5):

        print(
            f"\n========== Fold {fold + 1}/5 =========="
        )

        mae = run_fold(
            fold,
            samples,
            folds,
            args,
            device,
        )

        fold_results.append(mae)

    print("\n========== Final Results ==========")

    for i, mae in enumerate(fold_results):
        print(
            f"Fold {i + 1}: MAE = {mae:.3f} years"
        )

    print(
        f"Mean MAE: "
        f"{np.mean(fold_results):.3f} ± "
        f"{np.std(fold_results):.3f} years"
    )


if __name__ == "__main__":
    main()
