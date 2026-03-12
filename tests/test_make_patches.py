import torch

from src.Training.callbacks import make_patches


def test_make_patches_preserves_timestep_feature_order():
    x = torch.tensor(
        [
            [
                [10.0, 100.0],
                [20.0, 200.0],
                [30.0, 300.0],
                [40.0, 400.0],
            ]
        ]
    )

    patches = make_patches(x, patch_len=2, stride=2)

    expected = torch.tensor(
        [
            [
                [10.0, 100.0, 20.0, 200.0],
                [30.0, 300.0, 40.0, 400.0],
            ]
        ]
    )

    assert torch.equal(patches, expected)
