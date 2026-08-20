import torch
import torch.nn as nn

from .mae_loss import masked_reconstruction_loss
from .patch_embed_3d import PatchEmbed3D
from .qssm_block import QSSMBlock
from .masking import random_masking


class QSSM3DMAE(nn.Module):
    """
    3D-QSSM Masked Autoencoder.

    Paper defaults:
        Input size: 160 x 160 x 160
        Patch size: 16 x 16 x 16
        Number of patches: 1000
        Encoder dim: 384
        Encoder depth: 12
        Decoder dim: 192
        Decoder depth: 12
        Mask ratio: 75%
    """

    def __init__(
        self,
        img_size=160,
        patch_size=16,
        in_channels=1,
        encoder_dim=384,
        encoder_depth=12,
        decoder_dim=192,
        decoder_depth=12,
        mask_ratio=0.75,
        d_state=64,
        d_conv=7,
        expand=2,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio

        # ---------------------------------------------------------
        # Patch embedding
        # ---------------------------------------------------------
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=encoder_dim,
        )

        self.num_patches = self.patch_embed.num_patches

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, encoder_dim)
        )

        self.encoder_blocks = nn.ModuleList(
            [
                QSSMBlock(
                    dim=encoder_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(encoder_depth)
            ]
        )

        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # ---------------------------------------------------------
        # Encoder -> decoder projection
        # ---------------------------------------------------------
        self.decoder_embed = nn.Linear(
            encoder_dim,
            decoder_dim,
        )

        # Learnable token inserted for masked patches
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, decoder_dim)
        )

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_dim)
        )

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------
        self.decoder_blocks = nn.ModuleList(
            [
                QSSMBlock(
                    dim=decoder_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # Number of voxel values inside each patch
        self.patch_dim = (
            patch_size ** 3
        ) * in_channels

        # Predict original patch voxels
        self.decoder_pred = nn.Linear(
            decoder_dim,
            self.patch_dim,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(
            self.encoder_pos_embed,
            std=0.02,
        )
        nn.init.trunc_normal_(
            self.decoder_pos_embed,
            std=0.02,
        )
        nn.init.normal_(
            self.mask_token,
            std=0.02,
        )

    def forward_encoder(self, x):
        """
        Convert MRI to patches, mask 75%, and encode
        only visible tokens.
        """

        x = self.patch_embed(x)

        # Add positional information before masking
        x = x + self.encoder_pos_embed

        x, mask, ids_restore = random_masking(
            x,
            mask_ratio=self.mask_ratio,
        )

        for block in self.encoder_blocks:
            x = block(x)

        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Restore masked token positions and reconstruct
        every original 3D patch.
        """

        x = self.decoder_embed(x)

        B, N_visible, C = x.shape
        N_total = ids_restore.shape[1]

        num_masked = N_total - N_visible

        mask_tokens = self.mask_token.repeat(
            B,
            num_masked,
            1,
        )

        # visible + mask tokens
        x_full = torch.cat(
            [x, mask_tokens],
            dim=1,
        )

        # Restore original patch ordering
        x_full = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(
                1,
                1,
                C,
            ),
        )

        x_full = x_full + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x_full = block(x_full)

        x_full = self.decoder_norm(x_full)

        # Predict voxel values for each patch
        pred = self.decoder_pred(x_full)

        return pred

    def forward(self, x):
    latent, mask, ids_restore = self.forward_encoder(x)

    pred = self.forward_decoder(
        latent,
        ids_restore,
    )

    loss = masked_reconstruction_loss(
        images=x,
        predictions=pred,
        mask=mask,
        patch_size=self.patch_size,
    )

    return loss, pred, mask


if __name__ == "__main__":
    model = QSSM3DMAE()

    x = torch.randn(
        1,
        1,
        160,
        160,
        160,
    )

    loss, pred, mask = model(x)

    print("Input shape :", x.shape)
    print("Prediction  :", pred.shape)
    print("Mask shape  :", mask.shape)
    print("Loss        :", loss.item())

    # 16^3 = 4096 voxels per patch
    assert pred.shape == (1, 1000, 4096)
    assert mask.shape == (1, 1000)

    print("QSSM3DMAE forward test passed.")
