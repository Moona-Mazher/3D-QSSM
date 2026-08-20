import torch
import torch.nn as nn


class BrainAgeHead(nn.Module):
    """
    Regression head for brain-age prediction.

    Input:
        Encoded token features [B, N, C]

    Output:
        Predicted age [B]
    """

    def __init__(
        self,
        embed_dim=384,
        dropout=0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, C]

        Returns:
            age: [B]
        """

        # Global mean pooling across tokens
        x = x.mean(dim=1)

        x = self.norm(x)

        age = self.regressor(x)

        return age.squeeze(-1)


if __name__ == "__main__":
    x = torch.randn(2, 1000, 384)

    model = BrainAgeHead(
        embed_dim=384
    )

    age = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", age.shape)

    assert age.shape == (2,)

    print("BrainAgeHead test passed.")
