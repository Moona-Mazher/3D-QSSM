import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from qssm.models.encoder_3d import QSSM3DEncoder


class TumorDataset(Dataset):
    """
    CSV columns:
    t1,t2,flair,t1gd,wt,tc,et
    """

    def __init__(self, rows, data_root=None):
        self.rows = rows
        self.root = Path(data_root) if data_root else None

    def __len__(self):
        return len(self.rows)

    def load(self, path):
        path = self.root / path if self.root else Path(path)
        return nib.load(str(path)).get_fdata(dtype=np.float32)

    def __getitem__(self, idx):
        row = self.rows[idx]

        images = []

        for key in ["t1", "t2", "flair", "t1gd"]:
            x = self.load(row[key])

            if x.shape != (160, 160, 160):
                raise ValueError(f"{key} has shape {x.shape}")

            x = (x - x.mean()) / (x.std() + 1e-8)
            images.append(x)

        image = np.stack(images, axis=0)

        masks = np.stack(
            [
                self.load(row["wt"]),
                self.load(row["tc"]),
                self.load(row["et"]),
            ],
            axis=0,
        )

        image = torch.from_numpy(image).float()
        masks = torch.from_numpy(masks).float()

        return image, masks


class QSSM3DSegmenter(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = QSSM3DEncoder(
            in_channels=4,
            embed_dim=384,
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(384, 192, 2, 2),
            nn.GELU(),
            nn.ConvTranspose3d(192, 96, 2, 2),
            nn.GELU(),
            nn.ConvTranspose3d(96, 48, 2, 2),
            nn.GELU(),
            nn.ConvTranspose3d(48, 24, 2, 2),
            nn.GELU(),
            nn.Conv3d(24, 3, 1),
        )

    def forward(self, x):
        x = self.encoder(x)

        B, N, C = x.shape

        # 1000 tokens -> 10 x 10 x 10 feature volume
        x = x.transpose(1, 2).reshape(
            B, C, 10, 10, 10
        )

        return self.decoder(x)


def dice_loss(logits, target):
    pred = torch.sigmoid(logits)

    intersection = (pred * target).sum(
        dim=(2, 3, 4)
    )

    denominator = (
        pred.sum(dim=(2, 3, 4))
        + target.sum(dim=(2, 3, 4))
    )

    dice = (
        2 * intersection + 1e-6
    ) / (
        denominator + 1e-6
    )

    return 1 - dice.mean()


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    rows = read_csv(args.csv)

    dataset = TumorDataset(
        rows,
        args.data_root,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    model = QSSM3DSegmenter().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
    )

    bce = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0

        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = (
                bce(logits, masks)
                + dice_loss(logits, masks)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Loss: {total_loss / len(loader):.4f}"
        )


if __name__ == "__main__":
    main()
