import torch

from src.models.time_series.patchTransformer import PatchTSTEncoder


def test_add_positional_embeddings_uses_absolute_patch_indices_for_visible_tokens():
    encoder = PatchTSTEncoder(
        patch_len=2,
        d_model=8,
        n_features=3,
        n_time_features=2,
        nhead=2,
        num_layers=1,
        dim_ff=16,
        dropout=0.0,
    )

    tok = torch.zeros(1, 2, 8)
    patch_indices = torch.tensor([[1, 4]], dtype=torch.long)
    positioned = encoder._add_positional_embeddings(tok, patch_indices=patch_indices)
    expected = encoder.get_patch_positional_embeddings(patch_indices)
    assert torch.allclose(positioned, expected)


def test_add_positional_embeddings_defaults_to_dense_positions_for_full_sequence():
    encoder = PatchTSTEncoder(
        patch_len=2,
        d_model=8,
        n_features=3,
        n_time_features=2,
        nhead=2,
        num_layers=1,
        dim_ff=16,
        dropout=0.0,
    )

    tok = torch.zeros(1, 4, 8)
    positioned = encoder._add_positional_embeddings(tok)
    expected = encoder.posenc(tok)

    assert torch.allclose(positioned, expected)
