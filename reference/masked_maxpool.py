"""
Masked global max-pool — plaintext reference for the MPC path (spec §6).

After 3 dense GCN layers + ReLU, padding nodes carry a nonzero bias offset,
so a naive max over all Nmax rows would leak/mix padding into the pooled
vector. We first clamp padding rows to NEG_LARGE using the secret binary
mask, then take the column-wise max over all Nmax rows.

MPC realisation (spec §5):
    Xsel = gpu_select(X, mask, NEG_LARGE)   # secret select: real→X, pad→NEG_LARGE
    pool = gpu_maxpool(Xsel over nodes)     # column-wise max → (feat_dim,)

NEG_LARGE must be small enough to never win the max, but stay within the
fixed-point representable range so it does not overflow (spec §10 risk).
"""
import numpy as np


def masked_global_max_pool(X: np.ndarray,
                           mask: np.ndarray,
                           neg_large: float = -1e9) -> np.ndarray:
    """Column-wise max over real-atom rows only.

    Args:
        X:         (nmax, feat_dim) node embeddings after GCN+ReLU.
        mask:      (nmax,) binary — 1 for real atoms, 0 for padding.
        neg_large: value written to padding rows so they never win the max.

    Returns:
        (feat_dim,) pooled feature vector = max over real-atom rows.
    """
    m = mask.reshape(-1, 1)                      # (nmax, 1)
    # select: real rows keep X, padding rows become neg_large
    Xsel = m * X + (1.0 - m) * neg_large         # gpu_select analogue
    return Xsel.max(axis=0)                       # gpu_maxpool over nodes
