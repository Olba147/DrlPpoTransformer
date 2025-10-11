import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# variance regularization for JEPA latent space loss function
def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    z: [B, D] pre-norm latents (no F.normalize)
    Returns scalar variance penalty: encourage per-dim std >= gamma
    """
    # center per-dim
    zc = z - z.mean(dim=0, keepdim=True)
    # numerical stability
    std = torch.sqrt(zc.var(dim=0, unbiased=False) + eps)
    # hinge on (gamma - std)
    return F.relu(gamma - std).pow(2).mean()