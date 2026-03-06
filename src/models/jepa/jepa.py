import torch
import torch.nn as nn


class JEPA(nn.Module):
    def __init__(
        self,
        context_enc,
        target_enc,
        d_model,
        ema_tau_min,
        ema_tau_max,
        nhead,
        dim_ff,
        dropout,
        predictor_num_layers=2,
        context_tokens=64,
        horizon_blocks=None,
    ):
        super().__init__()
        self.context_enc = context_enc
        self.target_enc = target_enc

        self.ema_tau_min = ema_tau_min
        self.ema_tau_max = ema_tau_max
        self.context_tokens = int(context_tokens)
        if self.context_tokens <= 0:
            raise ValueError(f"context_tokens must be > 0, got {self.context_tokens}")

        if horizon_blocks is None:
            horizon_blocks = {
                "near": [1, 1],
                "med": [2, 5],
                "far": [6, 17],
            }
        self.horizon_slices, self.horizon_tokens = self._build_horizon_slices(horizon_blocks)
        self.target_queries = nn.Parameter(torch.empty(1, self.horizon_tokens, d_model))
        nn.init.normal_(self.target_queries, std=0.02)

        predictor_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerDecoder(
            predictor_layer,
            num_layers=int(predictor_num_layers),
        )
        self.predictor_norm = nn.LayerNorm(d_model)

        self.target_enc.load_state_dict(self.context_enc.state_dict())
        for p in self.target_enc.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _build_horizon_slices(horizon_blocks: dict[str, list[int]]) -> tuple[dict[str, slice], int]:
        if not horizon_blocks:
            raise ValueError("horizon_blocks must be a non-empty dict.")

        parsed: list[tuple[str, int, int]] = []
        for name, bounds in horizon_blocks.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"horizon_blocks[{name}] must be [start, end], got {bounds}.")
            start, end = int(bounds[0]), int(bounds[1])
            if start < 1 or end < start:
                raise ValueError(
                    f"horizon_blocks[{name}] must satisfy 1 <= start <= end, got [{start}, {end}]."
                )
            parsed.append((name, start, end))

        parsed.sort(key=lambda x: x[1])
        if parsed[0][1] != 1:
            raise ValueError("horizon blocks must start at 1 and be contiguous.")

        prev_end = 0
        slices: dict[str, slice] = {}
        for name, start, end in parsed:
            if start != prev_end + 1:
                raise ValueError(
                    "horizon blocks must be contiguous without gaps/overlaps; "
                    f"expected start {prev_end + 1}, got {start} for block '{name}'."
                )
            slices[name] = slice(start - 1, end)
            prev_end = end

        return slices, prev_end

    def forward(self, X_full, T_full, asset_id=None):
        B, N = X_full.shape[:2]
        required_tokens = self.context_tokens + self.horizon_tokens
        if N < required_tokens:
            raise ValueError(
                f"Expected at least {required_tokens} patched tokens "
                f"(context_tokens={self.context_tokens}, horizon_tokens={self.horizon_tokens}), got {N}."
            )

        X_ctx = X_full[:, :self.context_tokens]
        T_ctx = T_full[:, :self.context_tokens]
        X_future = X_full[:, self.context_tokens:required_tokens]
        T_future = T_full[:, self.context_tokens:required_tokens]

        context_indices = torch.arange(
            self.context_tokens, device=X_full.device
        ).unsqueeze(0).expand(B, -1)
        future_indices = torch.arange(
            self.context_tokens, required_tokens, device=X_full.device
        ).unsqueeze(0).expand(B, -1)

        z_context = self.context_enc(
            X_ctx,
            T_ctx,
            asset_id=asset_id,
            patch_indices=context_indices,
            return_tokens=True,
        )

        with torch.no_grad():
            self.target_enc.eval()
            z_target = self.target_enc(
                X_future,
                T_future,
                asset_id=asset_id,
                patch_indices=future_indices,
                return_tokens=True,
            )

        query_tokens = self.target_queries.expand(B, self.horizon_tokens, -1)
        query_tokens = query_tokens + self.context_enc.get_patch_positional_embeddings(future_indices)

        pred_future = self.predictor_norm(
            self.predictor(
                tgt=query_tokens,
                memory=z_context,
            )
        )
        return pred_future, z_target

    @torch.no_grad()
    def ema_update(self, decay):
        for pt, pc in zip(self.target_enc.parameters(), self.context_enc.parameters()):
            pt.data.mul_(decay).add_(pc.data, alpha=1.0 - decay)
