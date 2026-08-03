"""
Fixed-point forward pass — the golden reference for the MPC-secured portion.

`FixedAffinity` wraps a float `AffinityModel` and runs its *secured* layers
(drug GCN path + fusion FC) over a bw-bit integer ring at a fixed scale, using
exactly the arithmetic the GPU-MPC backend performs (one truncation per matmul
via arithmetic right shift, then reduction into the signed bw-bit ring).

  bw=64 (default) : Z_{2^64} — matches the production 64-bit backend.
  bw=32           : Z_{2^32} — narrower ring for testing half-width compute.

Split of labour (spec §3 privacy model):
  • drug GCN path + fusion FC  → fixed-point int64   (secret-shared in MPC)
  • protein GatedCNN           → float, public        (plaintext)
The public protein vector is quantised at the fusion boundary.
"""
import numpy as np

from reference.fixedpoint import to_fixed, from_fixed, fixed_matmul, SCALE
from reference.dense_graph import smile_to_dense_graph
from reference.affinity_model import seq_cat


def _relu_fx(x):
    return np.maximum(x, np.int64(0))


class FixedAffinity:
    def __init__(self, model, scale: int = SCALE, bw: int = 64):
        self.m  = model
        self.s  = int(scale)
        self.bw = int(bw)
        # Signed bw-bit sentinel for masked max-pool positions.
        # Kept 2 bits away from the ring floor so bw=64 matches the previous
        # -(1<<62) constant, and bw=32 gives -(1<<30) — well inside [-2^31,2^31).
        self._neg_sentinel = np.int64(-(1 << (self.bw - 2)))
        q = lambda a: to_fixed(a, self.s)
        # pre-quantise every public weight once
        self.gcn_fx     = [(q(W), q(b)) for (W, b) in model.gcn]
        self.drug_fc_fx = [(q(W), q(b)) for (W, b) in model.drug_fc]
        self.fusion_fx  = [(q(W), q(b)) for (W, b) in model.fusion]

    # ── ring reduction ────────────────────────────────────────────────────────
    def _wrap(self, x: np.ndarray) -> np.ndarray:
        """Reduce int64 values into the signed bw-bit ring (no-op for bw=64)."""
        if self.bw >= 64:
            return x
        modulus = np.int64(1) << self.bw        # 2^bw  (fits in int64 for bw<64)
        half    = modulus >> np.int64(1)         # 2^(bw-1)
        x = np.asarray(x, dtype=np.int64)
        x_mod = x & (modulus - np.int64(1))     # unsigned mod via bitmask
        return np.where(x_mod >= half, x_mod - modulus, x_mod)

    # ── fixed-point primitives ────────────────────────────────────────────────
    def _linear(self, x_fx, W_fx, b_fx):
        """y = x @ W.T + b  in fixed-point. x:(...,in) W:(out,in) → (...,out)."""
        x2 = np.atleast_2d(x_fx)                           # (n, in)
        y  = self._wrap(fixed_matmul(x2, W_fx.T, self.s))  # truncate then wrap
        y  = self._wrap(y + b_fx)                           # bias add then wrap
        return y.reshape(x_fx.shape[:-1] + (W_fx.shape[0],)) if x_fx.ndim > 1 \
            else y.reshape(W_fx.shape[0])

    def _gcn_layer(self, H_fx, A_fx, W_fx, b_fx):
        """A_hat @ (X @ W.T) + b, all fixed-point (two truncations)."""
        XW  = self._wrap(fixed_matmul(H_fx, W_fx.T, self.s))  # (N, out) @ s
        out = self._wrap(fixed_matmul(A_fx, XW,    self.s))    # (N, out) @ s
        return self._wrap(out + b_fx)                           # bias @ s

    # ── secured drug path (fixed-point) ─────────────────────────────────────────
    def _drug_path_fx(self, X, A_hat, mask):
        X_fx = to_fixed(X, self.s)
        A_fx = to_fixed(A_hat, self.s)
        H = X_fx
        for (W_fx, b_fx) in self.gcn_fx:
            H = _relu_fx(self._gcn_layer(H, A_fx, W_fx, b_fx))
        # masked global max-pool over atoms: real atoms keep value, padded rows
        # are forced to the negative sentinel then max'd away.
        keep = (np.asarray(mask).reshape(-1, 1) != 0)
        masked = np.where(keep, H, self._neg_sentinel)
        pooled = masked.max(axis=0)                    # (376,) @ s
        # Drug_FCs: 376→1024 (relu) → 128
        W0, b0 = self.drug_fc_fx[0]
        h = _relu_fx(self._linear(pooled, W0, b0))     # (1024,)
        W1, b1 = self.drug_fc_fx[1]
        return self._linear(h, W1, b1)                 # (128,) PMVO @ s

    # ── secured fusion path (fixed-point) ────────────────────────────────────────
    def _fusion_fx(self, pmvo_fx, pvec_fx):
        h = np.concatenate([pmvo_fx, pvec_fx])         # (256,) drug first @ s
        n = len(self.fusion_fx)
        for k, (W, b) in enumerate(self.fusion_fx):
            h = self._linear(h, W, b)
            if k < n - 1:
                h = _relu_fx(h)
        return h                                       # (1,) @ s

    # ── public protein path (float) ──────────────────────────────────────────────
    def _protein_vec_fx(self, protein_seq):
        import torch
        enc = torch.tensor(seq_cat(protein_seq), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            pvec = self.m.gated(enc).numpy().squeeze(0)  # (128,) float, public
        return to_fixed(pvec, self.s)                    # quantise at MPC boundary

    # ── public API ────────────────────────────────────────────────────────────
    def predict(self, X, A_hat, mask, protein_seq):
        pmvo_fx = self._drug_path_fx(X, A_hat, mask)
        pvec_fx = self._protein_vec_fx(protein_seq)
        out_fx = self._fusion_fx(pmvo_fx, pvec_fx)
        return float(from_fixed(out_fx, self.s)[0])

    def predict_batch(self, pairs, nmax=138):
        out = []
        for smile, protein in pairs:
            X, A_hat, mask = smile_to_dense_graph(smile, nmax)
            out.append(self.predict(X, A_hat, mask, protein))
        return np.array(out, dtype=np.float64)
