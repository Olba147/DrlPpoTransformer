import torch
import torch.nn as nn
import torch.nn.functional as F

class JEPA(nn.Module):

    def __init__(self, context_enc, target_enc, d_model):
        """
        Args:
            input_size (int): Dimension of input
            hidden_size (int): Dimension of hidden layer
            output_size (int): Dimension of output
        """

        super(JEPA, self).__init__()

        self.context_enc = context_enc
        self.target_enc = target_enc

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, X_ctx, T_ctx, X_tgt, T_tgt):
        """
        Args:
            X_ctx (Tensor): [B, N, D]
            T_ctx (Tensor): [B, N, D]
            X_tgt (Tensor): [B, N, D]
            T_tgt (Tensor): [B, N, D]
        Returns:
            h_context (Tensor): [B, D]
            h_target (Tensor): [B, D]
            h_pred (Tensor): [B, D]
        """

        h_c  = self.context_enc(X_ctx, T_ctx)   # [B, D], accepts variable token counts
        with torch.no_grad():
            h_t = self.target_enc(X_tgt, T_tgt)  # [B, D], stop-grad target branch
        h_pred = self.mlp(h_c)                # [B, D]
        h_context = F.normalize(h_pred, dim=-1)
        h_target = F.normalize(h_t, dim=-1)
        return h_context, h_target, h_pred
    

    def ema_update(self, decay=0.999):
        with torch.no_grad():
            for pt, pc in zip(self.target_enc.parameters(), self.context_enc.parameters()):
                pt.data.mul_(decay).add_(pc.data, alpha=1.0 - decay)
        return