import torch
import torch.nn as nn

from .encoder_3d import QSSM3DEncoder
from .brain_age_head import BrainAgeHead


class QSSM3DBrainAge(nn.Module):
    """
    3D-QSSM model for brain-age regression.
    """

    def __init__(
        self,
        img_size=160,
        patch_size=16,
        in_channels=1,
        embed_dim=384,
        depth=12,
        d_state=64,
        d_conv=7,
        expand=2,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = QSSM3DEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.head = BrainAgeHead(
            embed_dim=embed_dim,
            dropout=dropout,
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W, D]

        Returns:
            predicted_age: [B]
        """

        features = self.encoder(x)
        predicted_age = self.head(features)

        return predicted_age


if __name__ == "__main__":
    model = QSSM3DBrainAge()

    x = torch.randn(
        1,
        1,
        160,
        160,
        160,
    )

    age = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", age.shape)

    assert age.shape == (1,)

    print("Brain-age model test passed.")
