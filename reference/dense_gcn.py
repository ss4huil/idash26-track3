"""
Dense GCN layer — plaintext reference for the MPC path (spec §5, §6).

A single GCN layer, equivalent to PyG's GCNConv with add_self_loops=True and
normalize=True, but expressed as two dense matmuls on the fixed-Nmax tensors
that get secret-shared into MPC:

    XW  = X @ W^T          # secret × public   (gpu_matmul)
    out = A_hat @ XW + b   # secret × secret   (gpu_matmul), broadcast bias

Because A_hat = D^{-1/2}(A+I)D^{-1/2} was pre-computed in plaintext (spec §5),
no reciprocal-sqrt is needed inside MPC. Padding rows/cols of A_hat are 0, so
padding nodes only carry the broadcast bias (later suppressed by the masked
max-pool, spec §6).
"""
import numpy as np


def dense_gcn_layer(X: np.ndarray,
                    A_hat: np.ndarray,
                    W: np.ndarray,
                    b: np.ndarray) -> np.ndarray:
    """One dense GCN layer.

    Args:
        X:     (nmax, in_ch)   node features (padding rows are 0).
        A_hat: (nmax, nmax)    symmetric-normalised adjacency (padding rows/cols 0).
        W:     (out_ch, in_ch) GCNConv linear weight (PyG layout: out × in).
        b:     (out_ch,)       bias, broadcast over nodes.

    Returns:
        (nmax, out_ch) node embeddings = A_hat @ (X @ W^T) + b.
    """
    XW = X @ W.T          # (nmax, out_ch)   secret × public
    out = A_hat @ XW      # (nmax, out_ch)   secret × secret
    out = out + b         # broadcast bias over all nodes
    return out
