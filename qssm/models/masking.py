import torch


def random_masking(x, mask_ratio=0.75):
    """
    Randomly mask tokens for MAE pretraining.

    Args:
        x:
            Tensor of shape [B, N, C]

        mask_ratio:
            Fraction of tokens to mask.
            Default = 0.75

    Returns:
        x_visible:
            Visible tokens after masking.

        mask:
            Binary mask of shape [B, N]
            0 = visible
            1 = masked

        ids_restore:
            Indices used later to restore the
            original token ordering.
    """

    B, N, C = x.shape

    len_keep = int(N * (1 - mask_ratio))

    # Random value for every token
    noise = torch.rand(B, N, device=x.device)

    # Sort tokens according to random noise
    ids_shuffle = torch.argsort(noise, dim=1)

    # Indices required to restore original order
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep only visible tokens
    ids_keep = ids_shuffle[:, :len_keep]

    x_visible = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, C),
    )

    # Create binary mask
    mask = torch.ones(
        [B, N],
        device=x.device,
    )

    mask[:, :len_keep] = 0

    # Restore mask to original token ordering
    mask = torch.gather(
        mask,
        dim=1,
        index=ids_restore,
    )

    return x_visible, mask, ids_restore


if __name__ == "__main__":
    x = torch.randn(2, 1000, 384)

    x_visible, mask, ids_restore = random_masking(
        x,
        mask_ratio=0.75,
    )

    print("Original tokens :", x.shape)
    print("Visible tokens  :", x_visible.shape)
    print("Mask shape      :", mask.shape)

    assert x_visible.shape == (2, 250, 384)
    assert mask.shape == (2, 1000)
    assert ids_restore.shape == (2, 1000)

    print("Random masking test passed.")
