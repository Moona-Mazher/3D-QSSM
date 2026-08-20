import argparse

import torch
from torch.utils.data import DataLoader

from qssm.data.mri_dataset import MRIDataset, find_nifti_files
from qssm.models.mae_3d import QSSM3DMAE


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0

    for step, images in enumerate(dataloader):
        images = images.to(device)

        optimizer.zero_grad()

        loss, _, _ = model(images)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(
            f"Step {step + 1}/{len(dataloader)} "
            f"| Loss: {loss.item():.6f}"
        )

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain 3D-QSSM using masked autoencoding."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed NIfTI MRI volumes.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    image_paths = find_nifti_files(
        args.data_dir
    )

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No NIfTI files found in {args.data_dir}"
        )

    print(
        f"Found {len(image_paths)} MRI volumes."
    )

    dataset = MRIDataset(
        image_paths=image_paths,
        expected_size=(160, 160, 160),
        normalize=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = QSSM3DMAE().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    for epoch in range(args.epochs):

        print(
            f"\nEpoch {epoch + 1}/{args.epochs}"
        )

        avg_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            device,
        )

        scheduler.step()

        print(
            f"Average loss: {avg_loss:.6f}"
        )


if __name__ == "__main__":
    main()
