import torch
import torch.nn as nn
import torch.nn.functional as F

class JEPA(nn.Module):
    def __init__(self, context_enc, target_enc, d_model, ema_start, ema_end):
        super().__init__()
        self.context_enc = context_enc
        self.target_enc  = target_enc
        
        self.ema_start = ema_start
        self.ema_end   = ema_end

        # projector with normalization
        def make_projector():
            return nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        self.proj_online = make_projector()
        self.proj_target = make_projector()   # EMA-updated copy
        
        # small predictor split to inject action between layers
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.GELU(),
            nn.Linear(2*d_model, d_model),
        )

        # make sure that target encoder is initialized the same as context encoder
        self.target_enc.load_state_dict(self.context_enc.state_dict())
        for p in self.target_enc.parameters():
            p.requires_grad_(False)
        for p in self.proj_target.parameters():
            p.requires_grad_(False)

        self.proj_target.load_state_dict(self.proj_online.state_dict())

    def forward(self, X_ctx, T_ctx, X_tgt, T_tgt, asset_id=None):

        z_c = self.proj_online(self.context_enc(X_ctx, T_ctx, asset_id=asset_id))        # [B, D]
        with torch.no_grad():
            self.target_enc.eval()
            self.proj_target.eval()
            z_t = self.proj_target(self.target_enc(X_tgt, T_tgt, asset_id=asset_id))     # [B, D]
        p_c = self.predictor(z_c)                                                        # [B, D]
        return F.normalize(p_c, dim=-1), F.normalize(z_t, dim=-1), p_c, z_t
        # return p_c, z_t

    @torch.no_grad()
    def ema_update(self, decay):

        # take the last index of ema_rate if epoch > len(ema_rate)
        for pt, pc in zip(self.target_enc.parameters(), self.context_enc.parameters()):
            pt.data.mul_(decay).add_(pc.data, alpha=1.0 - decay)        
        for pt, pc in zip(self.proj_target.parameters(), self.proj_online.parameters()):
            pt.data.mul_(decay).add_(pc.data, alpha=1.0 - decay)
