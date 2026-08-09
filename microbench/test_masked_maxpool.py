"""
TDD – RED phase: tests for masked_global_max_pool.

Padding nodes in X[:, c_size:] become nonzero after GCN layers (because
A_hat rows are 0 → XW rows are 0 → but bias offset makes them nonzero
after ReLU).  We must apply the binary mask to clamp those rows to NEG_LARGE
before taking global max-pool so they never win.

masked_global_max_pool(X, mask, neg_large) → (feat_dim,)

Run:  python3 -m pytest idash/mpc/tests/test_masked_maxpool.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
from torch_geometric.nn import global_max_pool as pyg_gmp

from reference.dense_graph  import smile_to_dense_graph
from reference.masked_maxpool import masked_global_max_pool   # not yet implemented


NEG_LARGE = -1e9     # plaintext reference; in MPC this must fit in fixed-point range


class TestMaskedGlobalMaxPool:
    """masked_global_max_pool(X, mask, neg_large) ≡ gmp for real-atom embeddings."""

    def test_return_shape(self):
        X = np.random.randn(16, 376).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32); mask[:3] = 1
        out = masked_global_max_pool(X, mask, NEG_LARGE)
        assert out.shape == (376,), f"expected (376,), got {out.shape}"

    def test_selects_max_of_real_atoms_only(self):
        """Pool of 3 real atoms among 16 should match max of those 3 rows."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((16, 8)).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32); mask[:3] = 1

        # poison padding rows with large values to confirm they're ignored
        X[3:] = 1e6

        out = masked_global_max_pool(X, mask, NEG_LARGE)
        expected = X[:3].max(axis=0)
        np.testing.assert_allclose(out, expected, atol=1e-5,
                                   err_msg="padding rows should be excluded")

    def test_matches_pyg_gmp_after_gcn(self):
        """Pool result == PyG gmp on the real-atom rows."""
        rng = np.random.default_rng(2)
        smile = "CC(=O)Oc1ccccc1C(=O)O"   # aspirin, 13 atoms
        X_dense, A_hat, mask = smile_to_dense_graph(smile, nmax=32)
        c_size = int(mask.sum())

        # Simulate embeddings: run 3 dense GCN layers + ReLU so padding rows
        # have nonzero values (bias contribution) — the real use-case.
        from reference.dense_gcn import dense_gcn_layer
        H = X_dense.copy()
        for (in_ch, out_ch) in [(94, 188), (188, 282), (282, 376)]:
            W = rng.standard_normal((out_ch, in_ch)).astype(np.float32)
            b = rng.standard_normal(out_ch).astype(np.float32)
            H = np.maximum(dense_gcn_layer(H, A_hat, W, b), 0)   # ReLU

        # Our masked pool
        out_dense = masked_global_max_pool(H, mask, NEG_LARGE)   # (376,)

        # PyG reference: gmp on the real-atom rows only
        h_real = torch.tensor(H[:c_size])
        batch  = torch.zeros(c_size, dtype=torch.long)
        out_pyg = pyg_gmp(h_real, batch).numpy().squeeze(0)       # (376,)

        np.testing.assert_allclose(out_dense, out_pyg, atol=1e-5,
                                   err_msg="masked max pool must match PyG gmp")

    def test_single_atom_molecule(self):
        """Edge case: molecule with 1 atom → pool equals that atom's row."""
        X = np.random.randn(16, 8).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32); mask[0] = 1
        out = masked_global_max_pool(X, mask, NEG_LARGE)
        np.testing.assert_allclose(out, X[0], atol=1e-6)

    def test_output_not_influenced_by_neg_large_threshold(self):
        """NEG_LARGE should be small enough to never appear in the output."""
        X = np.zeros((16, 4), dtype=np.float32)
        X[:3] = 0.1   # real atoms have positive values
        mask = np.zeros(16, dtype=np.float32); mask[:3] = 1
        for neg_large in [-1e9, -1e6, -1e4]:
            out = masked_global_max_pool(X, mask, neg_large)
            assert np.all(out >= 0.0), \
                f"NEG_LARGE={neg_large} leaked into pool output"

    def test_all_real_atoms_nmax(self):
        """When c_size == nmax no padding exists — pool equals global max."""
        nmax = 8
        X = np.random.randn(nmax, 16).astype(np.float32)
        mask = np.ones(nmax, dtype=np.float32)
        out = masked_global_max_pool(X, mask, NEG_LARGE)
        np.testing.assert_allclose(out, X.max(axis=0), atol=1e-6)
