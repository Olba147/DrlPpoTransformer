import torch

from src.models.jepa.jepa import JEPA
from src.models.time_series.patchTransformer import PatchTSTEncoder


class SpyPatchTSTEncoder(PatchTSTEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_patch_indices = None

    def forward(self, *args, patch_indices=None, **kwargs):
        self.last_patch_indices = None if patch_indices is None else patch_indices.detach().clone()
        return super().forward(*args, patch_indices=patch_indices, **kwargs)


def _build_encoder() -> SpyPatchTSTEncoder:
    return SpyPatchTSTEncoder(
        patch_len=2,
        d_model=8,
        n_features=3,
        n_time_features=2,
        nhead=2,
        num_layers=1,
        dim_ff=16,
        dropout=0.0,
        num_assets=None,
    )


def test_multihorizon_forward_shapes_and_block_lengths():
    context_tokens = 4
    horizon_blocks = {
        "near": [1, 1],
        "med": [2, 5],
        "far": [6, 17],
    }
    horizon_tokens = 17

    context_enc = _build_encoder()
    target_enc = _build_encoder()
    model = JEPA(
        context_enc=context_enc,
        target_enc=target_enc,
        d_model=8,
        ema_tau_min=0.99,
        ema_tau_max=0.999,
        nhead=2,
        dim_ff=16,
        dropout=0.0,
        predictor_num_layers=1,
        context_tokens=context_tokens,
        horizon_blocks=horizon_blocks,
    )

    batch_size = 2
    total_tokens = context_tokens + horizon_tokens
    x_full = torch.randn(batch_size, total_tokens, 6)
    t_full = torch.randn(batch_size, total_tokens, 4)

    pred, target = model(x_full, t_full)

    assert pred.shape == (batch_size, horizon_tokens, 8)
    assert target.shape == (batch_size, horizon_tokens, 8)
    assert pred[:, model.horizon_slices["near"], :].shape[1] == 1
    assert pred[:, model.horizon_slices["med"], :].shape[1] == 4
    assert pred[:, model.horizon_slices["far"], :].shape[1] == 12


def test_multihorizon_uses_absolute_future_patch_indices():
    context_tokens = 4
    horizon_blocks = {
        "near": [1, 1],
        "med": [2, 3],
        "far": [4, 6],
    }
    horizon_tokens = 6

    context_enc = _build_encoder()
    target_enc = _build_encoder()
    model = JEPA(
        context_enc=context_enc,
        target_enc=target_enc,
        d_model=8,
        ema_tau_min=0.99,
        ema_tau_max=0.999,
        nhead=2,
        dim_ff=16,
        dropout=0.0,
        predictor_num_layers=1,
        context_tokens=context_tokens,
        horizon_blocks=horizon_blocks,
    )

    batch_size = 3
    total_tokens = context_tokens + horizon_tokens
    x_full = torch.randn(batch_size, total_tokens, 6)
    t_full = torch.randn(batch_size, total_tokens, 4)

    _pred, _target = model(x_full, t_full)

    expected_context = torch.arange(context_tokens).unsqueeze(0).expand(batch_size, -1)
    expected_future = torch.arange(context_tokens, total_tokens).unsqueeze(0).expand(batch_size, -1)

    assert torch.equal(context_enc.last_patch_indices, expected_context)
    assert torch.equal(target_enc.last_patch_indices, expected_future)
