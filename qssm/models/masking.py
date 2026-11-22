
import torch


def random_masking(x, mask_ratio=0.75):
    """
    Randomly mask tokens for MAE pretraining while
    preserving the spatial order of visible tokens.

    Args:
        x:
            Tensor of shape [B, N, C]

        mask_ratio:
            Fraction of tokens to mask.
            Default = 0.75

    Returns:
        x_visible:
            Visible tokens in original spatial order.

        mask:
            Binary mask of shape [B, N]
            0 = visible
            1 = masked

        ids_restore:
            Indices used by the decoder to restore
            the original token ordering.
    """

    B, N, C = x.shape

    len_keep = int(N * (1 - mask_ratio))

    # Generate random values for every token
    noise = torch.rand(B, N, device=x.device)

    # Randomly select visible and masked tokens
    ids_shuffle = torch.argsort(noise, dim=1)

    # Keep visible tokens in original spatial order
    ids_keep = torch.sort(
        ids_shuffle[:, :len_keep], dim=1
    ).values

    # Keep masked token indices in original order
    ids_masked = torch.sort(
        ids_shuffle[:, len_keep:], dim=1
    ).values

    # Construct decoder ordering:
    # visible tokens followed by masked tokens
    ids_order = torch.cat(
        [ids_keep, ids_masked], dim=1
    )

    # Indices to restore original token positions
    ids_restore = torch.argsort(ids_order, dim=1)

    # Extract visible tokens
    x_visible = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, C),
    )

    # Binary mask in original spatial order
    mask = torch.ones(
        (B, N),
        device=x.device,
    )

    mask.scatter_(1, ids_keep, 0)

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
