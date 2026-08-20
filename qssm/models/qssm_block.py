import sys
from pathlib import Path

import torch
import torch.nn as nn


HYDRA_ROOT = Path(__file__).resolve().parents[2] / "external" / "hydra"

if str(HYDRA_ROOT) not in sys.path:
    sys.path.insert(0, str(HYDRA_ROOT))

try:
    from hydra.modules.hydra import Hydra
except ImportError as e:
    raise ImportError(
        "Could not import the official Hydra implementation. "
        "Make sure the repository was cloned with submodules:\n"
        "git clone --recurse-submodules https://github.com/Moona-Mazher/3D-QSSM.git"
    ) from e


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
