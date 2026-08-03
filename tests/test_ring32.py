"""
TDD: 32-bit ring variant of the fixed-point forward pass.

The GPU-MPC backend can run over Z_{2^32} instead of Z_{2^64}. A 32-bit ring
halves share/comm width and can be much faster, but shrinks the safe magnitude
window and coarsens resolution. `FixedAffinity(model, scale, bw=32)` runs
*exactly* the arithmetic the 32-bit backend performs: every ring element is
reduced into the signed 32-bit range after each op (matmul truncation, bias
add), and the masked-pool sentinel must live inside the ring.

Each test names the break it catches:
  * bw32 output not reduced mod 2^32          -> test_bw32_wraps_output_into_int32_range
  * -(1<<62) sentinel overflows a 32-bit ring -> test_bw32_sentinel_stays_in_ring
  * 32-bit ring diverges from 64-bit on small -> test_bw32_matches_bw64_on_small_values
  * 32-bit precision fails the iDASH ≤2% gate -> test_bw32_preserves_labels / mae

Run:  ~/.pyenv/versions/3.8.7/bin/python -m pytest idash/mpc/tests/test_ring32.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from reference.affinity_model import AffinityModel
from reference.fixed_forward import FixedAffinity
from reference.dense_graph import smile_to_dense_graph
from reference.metrics import binarize

NMAX = 138
INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1


def _pairs(n):
    smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "CCC", "OCC", "CCCl", "CCBr"]
    prot = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    return [(smiles[i % len(smiles)], prot) for i in range(n)]


class TestRing32Semantics:
    def test_bw32_wraps_output_into_int32_range(self):
        # Break: a bw=32 forward pass that leaves values int64-wide (no mod 2^32
        # reduction) would emit fixed-point intermediates outside [-2^31, 2^31).
        m = AffinityModel.from_random(feat_dim=94, seed=1)
        fm = FixedAffinity(m, scale=12, bw=32)
        X, A_hat, mask = smile_to_dense_graph("CC(=O)O", NMAX)
        pmvo_fx = fm._drug_path_fx(X, A_hat, mask)     # raw fixed-point (128,)
        assert pmvo_fx.min() >= INT32_MIN
        assert pmvo_fx.max() <= INT32_MAX

    def test_bw32_sentinel_stays_in_ring(self):
        # Break: the 64-bit masked-pool sentinel -(1<<62) is > 2^31, so in a
        # 32-bit ring it wraps to a bogus value and corrupts the max-pool.
        m = AffinityModel.from_random(feat_dim=94, seed=2)
        fm = FixedAffinity(m, scale=12, bw=32)
        assert INT32_MIN <= fm._neg_sentinel <= INT32_MAX

    def test_bw32_matches_bw64_on_small_values(self):
        # Break: an incorrect ring simulation would make bw=32 disagree with
        # bw=64 even when no value ever approaches 2^31 (small weights, scale 12).
        m = AffinityModel.from_random(feat_dim=94, seed=3)
        pairs = _pairs(4)
        y64 = FixedAffinity(m, scale=12, bw=64).predict_batch(pairs, NMAX)
        y32 = FixedAffinity(m, scale=12, bw=32).predict_batch(pairs, NMAX)
        # magnitudes here stay far below 2^31 at scale 12, so the two rings must
        # give identical results (no wraparound triggered).
        np.testing.assert_allclose(y32, y64, rtol=0, atol=1e-9)


class TestRing32AccuracyGate:
    def test_bw32_preserves_labels_at_scale12(self):
        # Break: 32-bit quantisation error flips a binarised label vs the float
        # model — that is exactly the iDASH graded-metric failure.
        m = AffinityModel.from_random(feat_dim=94, seed=4)
        pairs = _pairs(8)
        yf = m.predict_batch(pairs, nmax=NMAX)
        y32 = FixedAffinity(m, scale=12, bw=32).predict_batch(pairs, NMAX)
        thr = float(np.median(yf))
        np.testing.assert_array_equal(binarize(yf, thr), binarize(y32, thr))

    def test_bw32_mean_abs_error_small_at_scale13(self):
        # Break: 32-bit accumulated truncation error exceeds the affinity budget.
        # Scale=12 has MAE≈0.030 for both bw=32 and bw=64 (resolution limit, not
        # a ring-width issue). At scale=13 the truncation noise drops to ~0.013.
        m = AffinityModel.from_random(feat_dim=94, seed=5)
        pairs = _pairs(8)
        yf = m.predict_batch(pairs, nmax=NMAX)
        y32 = FixedAffinity(m, scale=13, bw=32).predict_batch(pairs, NMAX)
        mae = float(np.mean(np.abs(yf - y32)))
        assert mae < 0.02, f"bw=32 scale=13 MAE too large: {mae}"
