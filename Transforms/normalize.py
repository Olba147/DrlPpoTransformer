# Transforms/normalize_features.py
import torch

class NormalizeFeatures:
    """Z-score per feature using TRAIN stats. Works on dict batches with keys:
       'X_ctx':[B,L,C], 'X_tgt':[B,L,C]. Keeps stats for reuse (val/test/inference)."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.detach()
        self.std  = std.detach().clamp_min(1e-6)

    def __call__(self, batch: dict) -> dict:
        out = dict(batch)
        for k in ("X_ctx", "X_tgt"):
            if k in out:
                X = out[k]
                out[k] = (X - self.mean.to(X.device)) / self.std.to(X.device)
        out["stats"] = {"mean": self.mean, "std": self.std}
        return out

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.std.to(X.device) + self.mean.to(X.device)


@torch.no_grad()
def fit_feature_stats(train_loader, key="X_ctx"):
    """Compute per-feature mean/std over TRAIN (after BuildOHLCVFeatures, before patching)."""
    m_sum = None
    m2_sum = None
    n_total = 0
    for batch in train_loader:
        X = batch[key]                     # [B,L,C]
        Xv = X.reshape(-1, X.size(-1))     # [B*L, C]
        n_b = Xv.size(0)
        m = Xv.mean(dim=0)                 # [C]
        v = Xv.var(dim=0, unbiased=False)  # [C]
        if m_sum is None:
            m_sum = m * n_b
            m2_sum = (v + m**2) * n_b
        else:
            m_sum += m * n_b
            m2_sum += (v + m**2) * n_b
        n_total += n_b
    mean = m_sum / n_total
    var  = m2_sum / n_total - mean**2
    std  = var.clamp_min(1e-12).sqrt()
    return mean, std