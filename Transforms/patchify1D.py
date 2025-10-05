import torch


class Patchify1D:
    """
    Inputs:
      X_ctx: [Lx, C], time_cont: [Lx, 2]
      X_tgt: [Ly, C], time_cont_tgt: [Ly, 2]
    Outputs:
      X_ctx: [N_ctx, P*C]           time_cont: [N_ctx, P*2]
      X_tgt: [N_tgt, P*C]           time_cont_tgt: [N_tgt, P*2]
    """
    def __init__(self, patch_len: int = 16, stride: int = 16, flatten: bool = True):
        self.P, self.S, self.flatten = patch_len, stride, flatten

    def _patch_one(self, X):  # X: [L, C]
        L, C = X.shape
        patches = X.unfold(0, self.P, self.S)  # [N, P, C]
        return patches.reshape(patches.size(0), -1) if self.flatten else patches  # [N, P*C] or [N,P,C]

    def __call__(self, sample):
        Xc, Xt = sample["X_ctx"], sample["X_tgt"]
        Tc, Tt = sample["time_cont"], sample["time_cont_tgt"]

        def patch_both(X, T):
            Xp = self._patch_one(X)
            Tp = self._patch_one(T)   # T: [L,2] -> [N, P*2] if flatten
            return Xp, Tp

        Xp_ctx, Tp_ctx = patch_both(Xc, Tc)
        Xp_tgt, Tp_tgt = patch_both(Xt, Tt)

        return {"X": Xp_ctx, "X_tgt": Xp_tgt, "time_cont": Tp_ctx, "time_cont_tgt": Tp_tgt}