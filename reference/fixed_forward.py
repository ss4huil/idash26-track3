"""
Fixed-point forward pass — the golden reference for the MPC-secured portion.

`FixedAffinity` wraps a float `AffinityModel` and runs its *secured* layers
(drug GCN path + fusion FC) over a 64-bit integer ring at a fixed scale, using
exactly the arithmetic the GPU-MPC backend performs (int64, one truncation per
matmul via arithmetic right shift). This lets us measure quantisation error
against the float model before any C++ is written.

Split of labour (spec §3 privacy model):
  • drug GCN path + fusion FC  → fixed-point int64   (secret-shared in MPC)
  • protein GatedCNN           → float, public        (plaintext)
The public protein vector is quantised at the fusion boundary, i.e. exactly
where it is secret-shared into the MPC fusion network.
"""
import numpy as np

from reference.fixedpoint import to_fixed, from_fixed, fixed_matmul, SCALE
from reference.dense_graph import smile_to_dense_graph
from reference.affinity_model import seq_cat

# most-negative sentinel for masked positions in the fixed-point max-pool.
# int64 floor, kept away from the true minimum to avoid accidental overflow on
# any downstream add (there is none before the max, but be safe).
_NEG_SENTINEL = np.int64(-(1 << 62))


def _relu_fx(x):
    return np.maximum(x, np.int64(0))


class FixedAffinity:
    def __init__(self, model, scale: int = SCALE):
        self.m = model
        self.s = int(scale)
        q = lambda a: to_fixed(a, self.s)
        # pre-quantise every public weight once
        self.gcn_fx     = [(q(W), q(b)) for (W, b) in model.gcn]
        self.drug_fc_fx = [(q(W), q(b)) for (W, b) in model.drug_fc]
        self.fusion_fx  = [(q(W), q(b)) for (W, b) in model.fusion]

    # ── fixed-point primitives ────────────────────────────────────────────────
    def _linear(self, x_fx, W_fx, b_fx):
        """y = x @ W.T + b  in fixed-point. x:(...,in) W:(out,in) → (...,out)."""
        x2 = np.atleast_2d(x_fx)                       # (n, in)
        y = fixed_matmul(x2, W_fx.T, self.s)           # (n, out) @ scale s
        y = y + b_fx                                   # bias already @ scale s
        return y.reshape(x_fx.shape[:-1] + (W_fx.shape[0],)) if x_fx.ndim > 1 \
            else y.reshape(W_fx.shape[0])

    def _gcn_layer(self, H_fx, A_fx, W_fx, b_fx):
        """A_hat @ (X @ W.T) + b, all fixed-point (two truncations)."""
        XW = fixed_matmul(H_fx, W_fx.T, self.s)        # (N, out) @ s
        out = fixed_matmul(A_fx, XW, self.s)           # (N, out) @ s
        return out + b_fx                              # broadcast bias @ s

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
        masked = np.where(keep, H, _NEG_SENTINEL)
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
