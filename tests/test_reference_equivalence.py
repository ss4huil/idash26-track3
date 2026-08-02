"""
TDD – equivalence to the OFFICIAL secure reference (flax_secure_deepdtagen.py).

The official GraphEncoder normalises the adjacency *inside* each GCN layer over
the padded matrix (adds self-loops to real nodes only, eps=1e-6 in the rsqrt,
zeroes padding rows after the linear). My dense path instead pre-computes
A_hat = D^{-1/2}(A+I)D^{-1/2} on the real-atom block in plaintext (an MPC
acceleration lever that avoids secure rsqrt).

This test ports the reference GraphEncoder's *affinity* path to numpy and proves
my `AffinityModel.drug_path` produces the same PMVO vector under identical
weights — i.e. the acceleration lever is numerically faithful. The key
invariant: padding rows never couple into real rows (padding columns of the
normalised adjacency are zero) and the masked pool ignores them, so my omission
of the reference's explicit padding-row zeroing is safe.

Run:  python3 -m pytest idash/mpc/tests/test_reference_equivalence.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from rdkit import Chem

from reference.affinity_model import AffinityModel
from reference.dense_graph import smile_to_dense_graph, FEAT_DIM

NMAX = 138
REF_EPS = 1e-6


def _relu(x):
    return np.maximum(x, 0.0)


def _raw_adj(smile, nmax):
    """Raw adjacency WITHOUT self-loops (the reference NPZ format)."""
    mol = Chem.MolFromSmiles(smile)
    n = mol.GetNumAtoms()
    a = np.zeros((nmax, nmax), dtype=np.float64)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        a[i, j] = 1.0
        a[j, i] = 1.0
    mask = np.zeros(nmax, dtype=bool)
    mask[:n] = True
    return a, mask


def _ref_gcn_layer(x, raw_adj, node_mask, kernel, bias, eps=REF_EPS):
    """Exact numpy port of flax GCNLayer.__call__ (kernel is (in,out))."""
    n = raw_adj.shape[-1]
    eye = np.eye(n)
    real = node_mask.astype(np.float64)
    a = raw_adj * real[:, None] * real[None, :]
    a = a + eye * real[:, None]                      # self-loop, real nodes only
    degree = a.sum(axis=-1)
    inv = np.where(degree > 0, 1.0 / np.sqrt(degree + eps), 0.0)
    a_norm = a * inv[:, None] * inv[None, :]
    out = a_norm @ x
    out = out @ kernel + bias
    return out * real[:, None]                        # zero padding rows


def _ref_drug_encoder(drug_x, raw_adj, node_mask, gcn_k, gcn_b, dfc_k, dfc_b):
    """Reference GraphEncoder affinity path → PMVO (128,)."""
    h = drug_x
    for W, b in zip(gcn_k, gcn_b):
        h = _relu(_ref_gcn_layer(h, raw_adj, node_mask, W, b))
    x = h  # last layer already relu'd above
    pooled = np.where(node_mask[:, None], x, -1.0e6).max(axis=0)   # (376,)
    p = _relu(pooled @ dfc_k[0] + dfc_b[0])                        # (1024,)
    return p @ dfc_k[1] + dfc_b[1]                                 # (128,)


class TestDrugEncoderEquivalence:
    @pytest.mark.parametrize("smile", [
        "CCO", "CC(=O)Nc1ccc(O)cc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "c1ccccc1", "O=C(O)c1ccccc1",
    ])
    def test_pmvo_matches_reference(self, smile):
        m = AffinityModel.from_random(feat_dim=FEAT_DIM, seed=7)

        # my path: dense A_hat (real-block normalised, no eps) + (out,in) weights
        X, A_hat, mask = smile_to_dense_graph(smile, NMAX)
        mine = m.drug_path(X, A_hat, mask)

        # reference oracle: raw adj + (in,out) kernels (transpose of mine),
        # normalise-inside-layer with eps, explicit padding-row zeroing
        raw_adj, node_mask = _raw_adj(smile, NMAX)
        gcn_k = [W.T for (W, b) in m.gcn]        # (out,in) → (in,out)
        gcn_b = [b for (W, b) in m.gcn]
        dfc_k = [W.T for (W, b) in m.drug_fc]
        dfc_b = [b for (W, b) in m.drug_fc]
        ref = _ref_drug_encoder(X.astype(np.float64), raw_adj, node_mask,
                                gcn_k, gcn_b, dfc_k, dfc_b)

        # eps=1e-6 + float32 storage → small but nonzero diff; must be tight.
        np.testing.assert_allclose(mine, ref, rtol=2e-3, atol=2e-3)

    def test_padding_rows_do_not_affect_real_output(self):
        """Corrupting padding-row features leaves the pooled PMVO unchanged —
        confirms padding never couples into the masked real output."""
        m = AffinityModel.from_random(feat_dim=FEAT_DIM, seed=11)
        X, A_hat, mask = smile_to_dense_graph("CC(=O)Nc1ccc(O)cc1", NMAX)
        base = m.drug_path(X, A_hat, mask)

        Xc = X.copy()
        pad = mask == 0
        Xc[pad] = np.random.default_rng(0).standard_normal((pad.sum(), FEAT_DIM))
        corrupted = m.drug_path(Xc, A_hat, mask)

        np.testing.assert_allclose(base, corrupted, rtol=1e-6, atol=1e-6)
