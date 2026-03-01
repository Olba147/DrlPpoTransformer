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
        mask_ratio=0.5,
    ):
        super().__init__()
        self.context_enc = context_enc
        self.target_enc = target_enc

        self.ema_tau_min = ema_tau_min
        self.ema_tau_max = ema_tau_max
        self.mask_ratio = float(mask_ratio)

        self.mask_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        predictor_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            predictor_layer,
            num_layers=int(predictor_num_layers),
        )
        self.predictor_norm = nn.LayerNorm(d_model)

        self.target_enc.load_state_dict(self.context_enc.state_dict())
        for p in self.target_enc.parameters():
            p.requires_grad_(False)

    def generate_masks(self, B: int, N: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        num_masked = int(N * self.mask_ratio)
        num_masked = max(1, min(N - 1, num_masked))

        perm = torch.argsort(torch.rand(B, N, device=device), dim=1)
        target_indices = perm[:, :num_masked]
        context_indices = perm[:, num_masked:]
        target_indices, _ = torch.sort(target_indices, dim=1)
        context_indices, _ = torch.sort(context_indices, dim=1)
        return context_indices, target_indices

    @staticmethod
    def _gather_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        return x.gather(1, gather_idx)

    def forward_masked(self, X_ctx, T_ctx, asset_id=None):
        B, N = X_ctx.shape[:2]
        context_indices, target_indices = self.generate_masks(B, N, X_ctx.device)

        X_ctx_visible = self._gather_tokens(X_ctx, context_indices)
        T_ctx_visible = self._gather_tokens(T_ctx, context_indices)

        z_context = self.context_enc(
            X_ctx_visible,
            T_ctx_visible,
            asset_id=asset_id,
            return_tokens=True,
        )

        with torch.no_grad():
            self.target_enc.eval()
            z_target_full = self.target_enc(
                X_ctx,
                T_ctx,
                asset_id=asset_id,
                return_tokens=True,
            )
            z_target = self._gather_tokens(z_target_full, target_indices)

        num_masked = target_indices.size(1)
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        pos_tokens = self.context_enc.get_patch_positional_embeddings(target_indices)
        masked_targets = mask_tokens + pos_tokens

        predictor_input = torch.cat([z_context, masked_targets], dim=1)
        pred_all = self.predictor_norm(self.predictor(predictor_input))
        pred_masked = pred_all[:, -num_masked:, :]
        return pred_masked, z_target

    def forward(self, X_ctx, T_ctx, asset_id=None):
        return self.forward_masked(X_ctx, T_ctx, asset_id=asset_id)

    @torch.no_grad()
    def ema_update(self, decay):
        for pt, pc in zip(self.target_enc.parameters(), self.context_enc.parameters()):
            pt.data.mul_(decay).add_(pc.data, alpha=1.0 - decay)
