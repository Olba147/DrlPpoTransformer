import math, torch
import torch.nn as nn
import torch.nn.functional as F

def sincos(x: torch.Tensor, period: float) -> torch.Tensor:
    # x: any shape, returns (..., 2) = [sin, cos]
    angle = 2 * math.pi * x / period
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [B, K]
        returns: [B, K, D]
        """
        base = self.pe.expand(indices.size(0), -1, -1)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, base.size(-1))
        return base.gather(1, gather_idx)

class PatchTSTEncoder(nn.Module):
    """
    Inputs:
      X_patch         : [B, N, patch_len*6]          (flattened price features per patch)
      time_cont_patch : [B, N, patch_len*2]          (flattened [weekday, minute_of_day] per patch step)
    Time encoding:
      - weekday       -> sin/cos with period 7
      - minute_of_day -> sin/cos with period 1440
      => per-step time fea = 4 dims; per-patch flattened time fea = patch_len*4
    Fusion:
      proj_price: Linear(patch_len*6 -> d_model)
      proj_time : Linear(patch_len*4 -> d_model)
      token = proj_price(Xp) + proj_time(Tp)

    Outputs:
      - task='embedding' -> [B, d_model]
    """
    def __init__(
        self,
        patch_len: int,
        d_model: int = 512,          # transformer hidden dim
        n_features: int = 9,
        n_time_features: int = 2,
        nhead: int = 8,              # num of attention heads   
        num_layers: int = 4,         # num of transformer blocks
        dim_ff: int = 512,           # FFN hidden dim
        dropout: float = 0.1,
        num_assets: int | None = None,
    ):
        super().__init__()
        self.patch_len = patch_len

        self.n_features = n_features
        self.n_time_features = n_time_features

        self.proj_price = nn.Linear(patch_len * self.n_features, d_model)
        self.proj_time  = nn.Linear(patch_len * self.n_time_features * 2, d_model)  # 4 = sin/cos weekday + sin/cos minute
        self.time_gate  = nn.Parameter(torch.tensor(0.1))    # optional learnable scale
        self.asset_emb = nn.Embedding(num_assets, d_model) if num_assets is not None else None
        self.asset_gate = nn.Parameter(torch.tensor(0.1)) if num_assets is not None else None

        self.posenc = PositionalEncoding(d_model, max_len=10000)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

        self.head = nn.Identity()

    def get_patch_positional_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        return self.posenc.gather(indices.long())

    def _add_positional_embeddings(
        self,
        tok: torch.Tensor,
        patch_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if patch_indices is None:
            batch_size, num_tokens = tok.shape[:2]
            patch_indices = torch.arange(num_tokens, device=tok.device).unsqueeze(0).expand(batch_size, -1)

        if patch_indices.dim() != 2:
            raise ValueError(
                f"patch_indices must have shape [B, N], got {tuple(patch_indices.shape)}"
            )

        patch_pos = self.get_patch_positional_embeddings(patch_indices)
        if tok.size(1) != patch_pos.size(1):
            raise ValueError(
                "Token count must match patch_indices length when patch_indices are provided."
            )
        return tok + patch_pos

    def _encode_time_patch(self, time_cont_patch: torch.Tensor) -> torch.Tensor:
        """
        time_cont_patch: [B, N, patch_len*2] with (weekday, minute_of_day) flattened per step
        Returns:
          Tp_flat: [B, N, patch_len*4] with sin/cos for both (weekday, minute)
        """
        B, N, PM2 = time_cont_patch.shape
        P = self.patch_len
        assert PM2 == P * 2, f"Expected last dim patch_len*2, got {PM2} for patch_len {P}"

        tc = time_cont_patch.view(B, N, P, 2)          # [B, N, P, 2]
        wd = tc[..., 0]                                 # [B, N, P]
        mod = tc[..., 1]                                # minute_of_day [0..1439], [B, N, P]

        wd_sc  = sincos(wd, 7.0)                        # [B, N, P, 2]
        mod_sc = sincos(mod, 1440.0)                    # [B, N, P, 2]
        time_sc = torch.cat([wd_sc, mod_sc], dim=-1)    # [B, N, P, 4]
        return time_sc.reshape(B, N, P * 4)             # [B, N, P*4]

    def forward(
        self,
        X_patch: torch.Tensor,
        time_cont_patch: torch.Tensor,
        asset_id: torch.Tensor | None = None,
        patch_indices: torch.Tensor | None = None,
        return_tokens: bool = False,
    ):
        """
        X_patch        : [B, N, patch_len*6]
        time_cont_patch: [B, N, patch_len*2]
        asset_id: [B]
        patch_indices: [B, N] original patch indices for masked/context-only inputs
        Returns:
          pooled embedding -> [B, d_model]
          token sequence   -> [B, N, d_model] when return_tokens=True
        """
        # Project price & time separately, then fuse
        Tp_flat = self._encode_time_patch(time_cont_patch)   # [B, N, P*4]
        tok = self.proj_price(X_patch) + self.time_gate * self.proj_time(Tp_flat)  # [B, N, D]

        if self.asset_emb is not None and asset_id is not None:
            if asset_id.dim() > 1:
                asset_id = asset_id.view(-1)
            asset_vec = self.asset_emb(asset_id.long())       # [B, D]
            tok = tok + self.asset_gate * asset_vec.unsqueeze(1)

        tok = self._add_positional_embeddings(tok, patch_indices=patch_indices)
        z = self.encoder(tok)                                 # [B, T, D]
        z = self.final_norm(z)

        if return_tokens:
            return z
        return z.mean(dim=1)
