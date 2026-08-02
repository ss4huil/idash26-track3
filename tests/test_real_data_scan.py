"""
Robustness scan of graph construction against the REAL challenge CSVs.

These tests confirm two design decisions hold on actual data:
  • Nmax = 138 never truncates a real molecule (max atom count < 138)
  • Every SMILES in both test sets parses into a valid dense graph
    (no unparseable atoms, no NaNs, correct feature dim)

Skipped automatically if the challenge CSVs are not present.

Run:  python3 -m pytest idash/mpc/tests/test_real_data_scan.py -v
"""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from reference.dense_graph import smile_to_dense_graph, FEAT_DIM, count_atoms

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "project", "test")
_KIBA  = os.path.join(_ROOT, "kiba_test.csv")
_DAVIS = os.path.join(_ROOT, "davis_test.csv")
NMAX = 138


def _unique_smiles(path):
    seen = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            seen.add(row["compound_iso_smiles"])
    return sorted(seen)


@pytest.mark.skipif(not os.path.exists(_KIBA), reason="kiba_test.csv absent")
class TestKibaRealData:
    def test_all_smiles_parse_and_fit_nmax(self):
        smiles = _unique_smiles(_KIBA)
        assert len(smiles) > 0
        max_atoms = 0
        for s in smiles:
            n = count_atoms(s)
            max_atoms = max(max_atoms, n)
            assert 0 < n <= NMAX, f"{s} has {n} atoms (> Nmax={NMAX})"
        assert max_atoms <= NMAX
        print(f"\n[KIBA] {len(smiles)} unique SMILES, max atoms = {max_atoms}")

    def test_sample_graphs_are_finite(self):
        smiles = _unique_smiles(_KIBA)[:50]
        for s in smiles:
            X, A_hat, mask = smile_to_dense_graph(s, NMAX)
            assert X.shape == (NMAX, FEAT_DIM)
            assert A_hat.shape == (NMAX, NMAX)
            assert np.all(np.isfinite(X))
            assert np.all(np.isfinite(A_hat))
            assert mask.sum() == count_atoms(s)


@pytest.mark.skipif(not os.path.exists(_DAVIS), reason="davis_test.csv absent")
class TestDavisRealData:
    def test_all_smiles_parse_and_fit_nmax(self):
        smiles = _unique_smiles(_DAVIS)
        assert len(smiles) > 0
        max_atoms = max(count_atoms(s) for s in smiles)
        for s in smiles:
            assert 0 < count_atoms(s) <= NMAX
        print(f"\n[DAVIS] {len(smiles)} unique SMILES, max atoms = {max_atoms}")

    def test_sample_graphs_are_finite(self):
        smiles = _unique_smiles(_DAVIS)[:50]
        for s in smiles:
            X, A_hat, mask = smile_to_dense_graph(s, NMAX)
            assert np.all(np.isfinite(X))
            assert np.all(np.isfinite(A_hat))
