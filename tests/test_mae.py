import torch

from qssm.models.mae_3d import QSSM3DMAE


def test_mae_forward():
    model = QSSM3DMAE()

    x = torch.randn(
        1,
        1,
        160,
        160,
        160,
    )

    loss, pred, mask = model(x)

    assert pred.shape == (1, 1000, 4096)
    assert mask.shape == (1, 1000)
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_mae_forward()
    print("MAE test passed.")
