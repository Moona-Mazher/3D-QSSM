import torch
import torch.nn as nn

try:
    from hydra.modules.hydra import Hydra
except ImportError:
    Hydra = None


class QSSMBlock(nn.Module):
    """
    3D-QSSM encoder block.

    This block applies:
        LayerNorm
        -> bidirectional Hydra/QSSM mixer
        -> residual connection
        -> LayerNorm
        -> MLP
        -> residual connection
    """

    def __init__(
        self,
        dim=384,
        d_state=64,
        d_conv=7,
        expand=2,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        if Hydra is None:
            raise ImportError(
                "Hydra is not installed. Install the Hydra package "
                "before using QSSMBlock."
            )

        self.norm1 = nn.LayerNorm(dim)

        self.mixer = Hydra(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_mem_eff_path=False,
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, C]

        Returns:
            x: [B, N, C]
        """

        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


if __name__ == "__main__":
    x = torch.randn(2, 1000, 384)

    model = QSSMBlock(dim=384)

    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    assert y.shape == x.shape

    print("QSSMBlock test passed.")
