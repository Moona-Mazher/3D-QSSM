import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """
    Convert a 3D MRI volume into a sequence of patch embeddings.

    Default configuration from the 3D-QSSM paper:
        Input volume: 160 x 160 x 160
        Patch size:   16 x 16 x 16
        Number of patches: 10 x 10 x 10 = 1000
        Embedding dimension: 384
    """

    def __init__(
        self,
        img_size=160,
        patch_size=16,
        in_channels=1,
        embed_dim=384,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by "
                f"patch_size ({patch_size})."
            )

        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 3

        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, C, H, W, D]

        Returns:
            Tensor of shape [B, N, embed_dim]

            For default settings:
            [B, 1, 160, 160, 160]
                    ->
            [B, 1000, 384]
        """

        if x.ndim != 5:
            raise ValueError(
                f"Expected a 5D tensor [B, C, H, W, D], "
                f"but received shape {tuple(x.shape)}."
            )

        x = self.proj(x)

        # [B, embed_dim, 10, 10, 10]
        # -> [B, embed_dim, 1000]
        x = x.flatten(2)

        # -> [B, 1000, embed_dim]
        x = x.transpose(1, 2)

        return x


if __name__ == "__main__":
    model = PatchEmbed3D()

    x = torch.randn(2, 1, 160, 160, 160)
    tokens = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", tokens.shape)
    print("Number of patches:", model.num_patches)

    assert tokens.shape == (2, 1000, 384)

    print("PatchEmbed3D test passed.")
