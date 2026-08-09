"""
TDD – RED phase: tests for smile_to_dense_graph.

Run:  cd /home/jiang/master && python -m pytest idash/mpc/tests/test_dense_graph.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

# ── the module under test (does not exist yet → tests will fail) ──────────────
from reference.dense_graph import smile_to_dense_graph, FEAT_DIM  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
#  Constants matching the spec (§6)
# ─────────────────────────────────────────────────────────────────────────────
NMAX = 128
# FEAT_DIM imported from module — actual dim atom_features produces.
# NOTE: model.py declares node_feature=94; to be reconciled once weights arrive.


class TestSmileToDenseGraph:
    """smile_to_dense_graph(smile, nmax) → (X, A_hat, mask)"""

    def test_returns_three_arrays(self):
        result = smile_to_dense_graph("CCO", NMAX)
        assert len(result) == 3, "must return (X, A_hat, mask)"

    def test_output_shapes_ethanol(self):
        """Ethanol (CCO) has 3 heavy atoms."""
        X, A_hat, mask = smile_to_dense_graph("CCO", NMAX)
        assert X.shape     == (NMAX, FEAT_DIM), f"X shape wrong: {X.shape}"
        assert A_hat.shape == (NMAX, NMAX),     f"A_hat shape wrong: {A_hat.shape}"
        assert mask.shape  == (NMAX,),           f"mask shape wrong: {mask.shape}"

    def test_mask_marks_real_atoms_ethanol(self):
        """First 3 rows = real atoms → mask[:3] == 1, rest == 0."""
        _, _, mask = smile_to_dense_graph("CCO", NMAX)
        assert mask.sum() == 3,            f"expected 3 real atoms, got {int(mask.sum())}"
        assert np.all(mask[:3] == 1),      "first 3 positions should be masked 1"
        assert np.all(mask[3:]  == 0),     "padding positions should be masked 0"

    def test_mask_binary(self):
        _, _, mask = smile_to_dense_graph("CCO", NMAX)
        assert set(np.unique(mask)).issubset({0, 1}), "mask must be binary 0/1"

    def test_padding_rows_zero(self):
        """Padding atom rows in X must be all-zero."""
        X, _, mask = smile_to_dense_graph("CCO", NMAX)
        c_size = int(mask.sum())
        assert np.allclose(X[c_size:], 0), "padding rows in X must be zeros"

    def test_feature_rows_normalized(self):
        """Real atom feature rows are L1-normalised (sum ≈ 1)."""
        X, _, mask = smile_to_dense_graph("CCO", NMAX)
        c_size = int(mask.sum())
        row_sums = X[:c_size].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            f"feature rows should sum to 1, got {row_sums}"

    def test_a_hat_symmetric(self):
        """Symmetric normalisation D^{-1/2}(A+I)D^{-1/2} must be symmetric."""
        _, A_hat, _ = smile_to_dense_graph("CCO", NMAX)
        assert np.allclose(A_hat, A_hat.T, atol=1e-6), "A_hat must be symmetric"

    def test_a_hat_self_loops_present(self):
        """Diagonal of A_hat for real atoms must be non-zero (self-loop added)."""
        _, A_hat, mask = smile_to_dense_graph("CCO", NMAX)
        c_size = int(mask.sum())
        diag = np.diag(A_hat)[:c_size]
        assert np.all(diag > 0), "self-loops → A_hat diagonal must be positive"

    def test_a_hat_padding_rows_zero(self):
        """Padding rows/cols in A_hat must be zero (isolated padding nodes)."""
        _, A_hat, mask = smile_to_dense_graph("CCO", NMAX)
        c_size = int(mask.sum())
        assert np.allclose(A_hat[c_size:, :], 0), "padding rows in A_hat must be 0"
        assert np.allclose(A_hat[:, c_size:], 0), "padding cols in A_hat must be 0"

    def test_larger_molecule(self):
        """Aspirin (C9H8O4) has 13 heavy atoms."""
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        X, A_hat, mask = smile_to_dense_graph(aspirin, NMAX)
        assert mask.sum() == 13, f"aspirin should have 13 heavy atoms, got {int(mask.sum())}"
        assert X.shape     == (NMAX, FEAT_DIM)
        assert A_hat.shape == (NMAX, NMAX)

    def test_nmax_respected(self):
        """With nmax=16, shapes should use 16, not 128."""
        X, A_hat, mask = smile_to_dense_graph("CCO", nmax=16)
        assert X.shape     == (16, FEAT_DIM)
        assert A_hat.shape == (16, 16)
        assert mask.shape  == (16,)

    def test_deterministic(self):
        """Same SMILES → identical output on two calls."""
        r1 = smile_to_dense_graph("c1ccccc1", NMAX)
        r2 = smile_to_dense_graph("c1ccccc1", NMAX)
        for a, b in zip(r1, r2):
            assert np.allclose(a, b), "smile_to_dense_graph must be deterministic"
