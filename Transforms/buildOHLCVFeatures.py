import pandas as pd
import torch

class BuildOHLCVFeatures:
    """
    From raw windows → engineered features + time_cont (weekday, minute_of_day).
    Outputs (replaces raw keys with):
      X_ctx: [Lx, 6] = [log_open, log_high, log_low, log_ret_close, log1p(vol), log1p(qvol)]
      X_tgt: [Ly, 6]
      time_cont:      [Lx, 2] = [weekday, minute_of_day]
      time_cont_tgt:  [Ly, 2]
    """
    def __call__(self, sample):
        Xc = sample["X_ctx"]; Xt = sample["X_tgt"]
        tmc = sample["t_ctx"].to(torch.float64)
        tmt = sample["t_tgt"].to(torch.float64)

        def build(X, time_ms):
            # prices
            o = torch.log(X[:, 0].clamp_min(1e-12))
            h = torch.log(X[:, 1].clamp_min(1e-12))
            l = torch.log(X[:, 2].clamp_min(1e-12))
            c = torch.log(X[:, 3].clamp_min(1e-12))

            v  = torch.log1p(X[:, 4].clamp_min(0.0))
            qv = torch.log1p(X[:, 5].clamp_min(0.0))

            X_feat = torch.stack([o, h, l, c, v, qv], dim=-1).to(torch.float32)

            # time features from ms (UTC)
            ts = pd.to_datetime(time_ms.cpu().numpy(), unit="ms", utc=True)
            weekday = torch.as_tensor(ts.dayofweek.values, dtype=torch.float32, device=X.device)
            minute  = torch.as_tensor((ts.hour.values * 60 + ts.minute.values), dtype=torch.float32, device=X.device)
            time_cont = torch.stack([weekday, minute], dim=-1)  # [L, 2]
            return X_feat, time_cont

        X_ctx, time_ctx = build(Xc, tmc)
        X_tgt, time_tgt = build(Xt, tmt)

        out = {
            "X_ctx": X_ctx, "X_tgt": X_tgt,
            "t_ctx": time_ctx, "t_tgt": time_tgt
        }
        return out