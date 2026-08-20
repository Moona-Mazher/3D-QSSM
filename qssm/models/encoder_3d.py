import torch
import torch.nn as nn

from .patch_embed_3d import PatchEmbed3D
from .qssm_block import QSSMBlock


class QSSM3DEncoder(nn.Module):
    """
    3D-QSSM encoder.

    Default paper configuration:
        Input size: 160 x 160 x 160
        Patch size: 16 x 16 x 16
        Number of tokens: 1000
        Embedding dimension: 384
        Encoder depth: 12
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
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.num_patches = self.patch_embed.num_patches

        # Learnable 3D positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        self.blocks = nn.ModuleList(
            [
                QSSMBlock(
                    dim=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W, D]

        Returns:
            tokens: [B, N, C]

        Default output:
            [B, 1000, 384]
        """

        x = self.patch_embed(x)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


if __name__ == "__main__":
    model = QSSM3DEncoder()

    x = torch.randn(2, 1, 160, 160, 160)

    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    assert y.shape == (2, 1000, 384)

    print("QSSM3DEncoder test passed.")
    
