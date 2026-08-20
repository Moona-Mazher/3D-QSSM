import torch
import torch.nn.functional as F


def patchify_3d(x, patch_size=16):
    """
    Convert a 3D volume into flattened 3D patches.

    Args:
        x:
            Tensor [B, C, H, W, D]

        patch_size:
            Size of each cubic patch.

    Returns:
        patches:
            Tensor [B, N, patch_dim]

    Example:
        [B, 1, 160, 160, 160]
            ->
        [B, 1000, 4096]
    """

    B, C, H, W, D = x.shape
    p = patch_size

    if H % p != 0 or W % p != 0 or D % p != 0:
        raise ValueError(
            "Input dimensions must be divisible by patch_size."
        )

    h = H // p
    w = W // p
    d = D // p

    x = x.reshape(
        B,
        C,
        h,
        p,
        w,
        p,
        d,
        p,
    )

    # Rearrange patches
    x = x.permute(
        0, 2, 4, 6, 1, 3, 5, 7
    )

    patches = x.reshape(
        B,
        h * w * d,
        C * p * p * p,
    )

    return patches


def masked_reconstruction_loss(
    images,
    predictions,
    mask,
    patch_size=16,
):
    """
    MAE reconstruction loss computed only
    over masked patches.

    Args:
        images:
            Original MRI volume [B, C, H, W, D]

        predictions:
            Predicted patches [B, N, patch_dim]

        mask:
            Binary mask [B, N]
            0 = visible
            1 = masked

    Returns:
        Scalar reconstruction loss.
    """

    target = patchify_3d(
        images,
        patch_size=patch_size,
    )

    # Mean squared reconstruction error
    loss = F.mse_loss(
        predictions,
        target,
        reduction="none",
    )

    # Average over voxels within each patch
    loss = loss.mean(dim=-1)

    # Keep only masked patches
    loss = (
        loss * mask
    ).sum() / mask.sum().clamp(min=1.0)

    return loss


if __name__ == "__main__":
    images = torch.randn(
        2,
        1,
        160,
        160,
        160,
    )

    target = patchify_3d(
        images,
        patch_size=16,
    )

    print("Image shape :", images.shape)
    print("Patch shape :", target.shape)

    assert target.shape == (
        2,
        1000,
        4096,
    )

    predictions = torch.randn_like(target)

    mask = torch.zeros(
        2,
        1000,
    )

    # Simulate 75% masked patches
    mask[:, :750] = 1

    loss = masked_reconstruction_loss(
        images,
        predictions,
        mask,
        patch_size=16,
    )

    print("Loss:", loss.item())

    assert torch.isfinite(loss)

    print("MAE loss test passed.")
