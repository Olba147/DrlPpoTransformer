import torch
import torch.nn as nn
import torch.nn.functional as F

class JEPA(nn.Module):
    def __init__(self, context_enc, target_enc, d_model, ema_start, ema_end, n_epochs, action_dim: int | None = None):
        super().__init__()
        self.context_enc = context_enc
        self.target_enc  = target_enc
        self.action_dim = action_dim
        
        self.ema_rate = torch.linspace(ema_start, ema_end, n_epochs)

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
        
        # action-conditioning 
        self.action_proj = None
        if action_dim is not None:
            self.action_proj = nn.Linear(action_dim, d_model)

        # small predictor split to inject action between layers
        self.predictor_fc1 = nn.Linear(d_model, d_model)
        self.predictor_act = nn.GELU()
        self.predictor_fc2 = nn.Linear(d_model, d_model)

        self.proj_target.load_state_dict(self.proj_online.state_dict())

    def forward(self, X_ctx, T_ctx, X_tgt, T_tgt, action=None, asset_id=None):

        z_c = self.proj_online(self.context_enc(X_ctx, T_ctx, asset_id=asset_id))        # [B, D]
        with torch.no_grad():
            z_t = self.proj_target(self.target_enc(X_tgt, T_tgt, asset_id=asset_id))     # [B, D]
        h = self.predictor_act(self.predictor_fc1(z_c))
        if self.action_proj is not None and action is not None:
            if action.dim() == 1:
                action = action.unsqueeze(-1)
            h = h + self.action_proj(action)
        p_c = self.predictor_fc2(h)
        #return F.normalize(p_c, dim=-1), F.normalize(z_t, dim=-1), p_c, z_t
        return p_c, z_t

    @torch.no_grad()
    def ema_update(self, epoch):

        # take the last index of ema_rate if epoch > len(ema_rate)
        if epoch >= len(self.ema_rate):
            epoch = len(self.ema_rate) -1
        decay = self.ema_rate[epoch]
        for pt, pc in zip(self.target_enc.parameters(), self.context_enc.parameters()):
            pt.data.mul_(decay).add_(pc.data, alpha=1.0 - decay)
        for pt, pc in zip(self.proj_target.parameters(), self.proj_online.parameters()):
            pt.data.mul_(decay).add_(pc.data, alpha=1.0 - decay)
