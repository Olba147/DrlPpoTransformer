import torch
import torch.nn as nn
import torch.nn.functional as F

class JEPA(nn.Module):
    def __init__(self, context_enc, target_enc, d_model):
        super().__init__()
        self.context_enc = context_enc
        self.target_enc  = target_enc

        # projector with normalization
        def make_projector():
            return nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        self.proj_online = make_projector()
        self.proj_target = make_projector()   # EMA-updated copy
        
        # small predictor
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.proj_target.load_state_dict(self.proj_online.state_dict())

    def forward(self, X_ctx, T_ctx, X_tgt, T_tgt):

        #print shapes
        print("x_ctx", X_ctx.shape, "T_ctx", T_ctx.shape, "x_tgt", X_tgt.shape, "T_tgt", T_tgt.shape)

        z_c = self.proj_online(self.context_enc(X_ctx, T_ctx))        # [B, D]
        with torch.no_grad():
            z_t = self.proj_target(self.target_enc(X_tgt, T_tgt))     # [B, D]
        p_c = self.predictor(z_c)
        #return F.normalize(p_c, dim=-1), F.normalize(z_t, dim=-1), p_c, z_t
        return p_c, z_t

    @torch.no_grad()
    def ema_update(self, decay=0.999):
        for pt, pc in zip(self.target_enc.parameters(), self.context_enc.parameters()):
            pt.data.mul_(decay).add_(pc.data, alpha=1.0 - decay)
        for pt, pc in zip(self.proj_target.parameters(), self.proj_online.parameters()):
            pt.data.mul_(decay).add_(pc.data, alpha=1.0 - decay)