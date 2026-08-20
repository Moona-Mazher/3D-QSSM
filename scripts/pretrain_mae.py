import torch
from torch.utils.data import DataLoader, Dataset

from qssm.models.mae_3d import QSSM3DMAE


class DummyMRIDataset(Dataset):
    """
    Temporary dummy dataset for testing the training loop.

    Later, this will be replaced with the real MRI dataset loader.
    """

    def __init__(self, num_samples=8):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy 3D MRI volume
        x = torch.randn(1, 160, 160, 160)
        return x


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

        loss, pred, mask = model(images)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        print(
            f"Step {step + 1}/{len(dataloader)} "
            f"| Loss: {loss.item():.6f}"
        )

    return total_loss / len(dataloader)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    dataset = DummyMRIDataset(
        num_samples=8
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    model = QSSM3DMAE().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.05,
    )

    num_epochs = 2

    for epoch in range(num_epochs):

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}"
        )

        avg_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            device,
        )

        print(
            f"Average loss: {avg_loss:.6f}"
        )


if __name__ == "__main__":
    main()
