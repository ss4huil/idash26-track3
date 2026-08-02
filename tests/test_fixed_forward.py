"""
TDD – RED: fixed-point forward pass must reproduce the float model.

This is the accuracy-risk gate for the whole MPC port. The GPU-MPC backend
computes the secured portion (drug GCN path + fusion FC) over a 64-bit ring at
scale=12. `fixed_forward.FixedAffinity` runs *exactly* that arithmetic in
int64 (via fixedpoint.py) so we can prove the quantisation error is small
BEFORE writing any C++. The protein GatedCNN is public and stays in float; its
output vector is quantised at the fusion boundary (as it will be secret-shared
into the MPC fusion FC).

Contract: over random drugs/proteins, the fixed-point prediction must match the
float64 prediction closely, and — critically — must not flip the binarised
label often (the challenge grades sensitivity+specificity, not raw MSE).

Run:  python3 -m pytest idash/mpc/tests/test_fixed_forward.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from reference.affinity_model import AffinityModel
from reference.fixed_forward import FixedAffinity   # not yet implemented

NMAX = 138
SMILES = [
    "CCO", "CC(=O)O", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "C1CCCCC1", "O=C(O)c1ccccc1",
]
PROTEINS = [
    "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIV",
    "MABCDEFGHIKLMNPQRSTVWYABCDEFGHIKLMNPQRSTVWY",
]


def _pairs(n=6):
    return [(SMILES[i % len(SMILES)], PROTEINS[i % len(PROTEINS)])
            for i in range(n)]


class TestFixedMatchesFloat:
    def test_single_prediction_close(self):
        m = AffinityModel.from_random(feat_dim=94, seed=1)
        fm = FixedAffinity(m, scale=12)
        smile, prot = _pairs(1)[0]
        yf = m.predict_batch([(smile, prot)], nmax=NMAX)[0]
        yq = fm.predict_batch([(smile, prot)], nmax=NMAX)[0]
        # scale=12 → ~2^-12 resolution; accumulated error over the network
        # should stay well under 0.05 in affinity units.
        assert abs(yf - yq) < 0.05

    def test_batch_mean_abs_error_small(self):
        m = AffinityModel.from_random(feat_dim=94, seed=2)
        fm = FixedAffinity(m, scale=12)
        pairs = _pairs(6)
        yf = m.predict_batch(pairs, nmax=NMAX)
        yq = fm.predict_batch(pairs, nmax=NMAX)
        mae = float(np.mean(np.abs(yf - yq)))
        assert mae < 0.02, f"fixed-point MAE too large: {mae}"

    def test_higher_scale_more_accurate(self):
        """Increasing the scale should not increase the error."""
        m = AffinityModel.from_random(feat_dim=94, seed=3)
        pairs = _pairs(4)
        yf = m.predict_batch(pairs, nmax=NMAX)
        err12 = np.mean(np.abs(yf - FixedAffinity(m, 12).predict_batch(pairs, NMAX)))
        err16 = np.mean(np.abs(yf - FixedAffinity(m, 16).predict_batch(pairs, NMAX)))
        assert err16 <= err12 + 1e-6


class TestLabelStability:
    def test_binarised_labels_preserved(self):
        """The graded metric is label-based: quantisation must not flip labels
        away from the float model's decisions."""
        from reference.metrics import binarize
        m = AffinityModel.from_random(feat_dim=94, seed=4)
        fm = FixedAffinity(m, scale=12)
        pairs = _pairs(6)
        yf = m.predict_batch(pairs, nmax=NMAX)
        yq = fm.predict_batch(pairs, nmax=NMAX)
        # use the median of the float predictions as a synthetic threshold so
        # both classes are represented regardless of the random weights.
        thr = float(np.median(yf))
        np.testing.assert_array_equal(binarize(yf, thr), binarize(yq, thr))
