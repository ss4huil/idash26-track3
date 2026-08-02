"""
TDD – RED: CSV-driven reference runner (run_experiment.py integration layer).

`csv_runner.run_from_csv` reads the challenge CSV, runs float + fixed-point
predictions with a given model, computes the accuracy-gate metrics for both,
and returns a structured result dict. This is the bridge that ties every Python
reference module together and will be the gold reference the MPC C++ output is
compared against.

Tests use a random model (no weights needed) over a synthetic CSV to verify:
  • column parsing and dataset threshold detection
  • correct metric computation pipeline end-to-end
  • qualification gate evaluation
  • fixed-point accuracy column present and numerical

Run:  python3 -m pytest idash/mpc/tests/test_csv_runner.py -v
"""
import sys, os, io, csv, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from reference.affinity_model import AffinityModel
from reference.fixed_forward  import FixedAffinity
from reference.csv_runner      import run_from_csv   # not yet implemented

# Tiny synthetic CSV rows (real-looking SMILES + protein snippets)
_SMILES = [
    "CCO", "CC(=O)O", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
]
_PROTEIN = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGG"


def _make_csv(path, n=5, affinity_vals=None, dataset="kiba"):
    """Write a minimal challenge-format CSV."""
    if affinity_vals is None:
        affinity_vals = np.random.default_rng(0).uniform(5, 20, n).tolist()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["compound_iso_smiles", "target_smiles", "target_sequence", "affinity"])
        for i in range(n):
            w.writerow([_SMILES[i % len(_SMILES)],
                        _SMILES[i % len(_SMILES)],   # target_smiles = same (unused)
                        _PROTEIN,
                        affinity_vals[i]])
    return path


class TestCsvRunnerBasic:
    def test_returns_dict_with_required_keys(self, tmp_path):
        csv_path = _make_csv(str(tmp_path / "kiba_test.csv"), n=4)
        m  = AffinityModel.from_random(seed=0)
        fm = FixedAffinity(m, scale=12)
        result = run_from_csv(csv_path, m, fm)
        for key in ("dataset", "threshold", "float_acc", "fixed_acc",
                    "qualified", "n_samples", "float_preds", "fixed_preds"):
            assert key in result, f"missing key: {key}"

    def test_kiba_threshold_inferred_from_filename(self, tmp_path):
        p = _make_csv(str(tmp_path / "kiba_test.csv"), n=3)
        m  = AffinityModel.from_random(seed=1)
        fm = FixedAffinity(m, scale=12)
        r = run_from_csv(p, m, fm)
        assert r["threshold"] == 12.1

    def test_davis_threshold_inferred(self, tmp_path):
        p = _make_csv(str(tmp_path / "davis_test.csv"), n=3)
        m  = AffinityModel.from_random(seed=2)
        fm = FixedAffinity(m, scale=12)
        r = run_from_csv(p, m, fm)
        assert r["threshold"] == 7.0

    def test_n_samples_matches_csv_rows(self, tmp_path):
        p = _make_csv(str(tmp_path / "kiba_test.csv"), n=5)
        m  = AffinityModel.from_random(seed=3)
        fm = FixedAffinity(m, scale=12)
        r = run_from_csv(p, m, fm)
        assert r["n_samples"] == 5

    def test_preds_are_numpy_arrays_of_right_length(self, tmp_path):
        p = _make_csv(str(tmp_path / "kiba_test.csv"), n=4)
        m  = AffinityModel.from_random(seed=4)
        fm = FixedAffinity(m, scale=12)
        r = run_from_csv(p, m, fm)
        assert len(r["float_preds"]) == 4
        assert len(r["fixed_preds"]) == 4

    def test_accuracy_values_in_0_1(self, tmp_path):
        p = _make_csv(str(tmp_path / "kiba_test.csv"), n=5)
        m  = AffinityModel.from_random(seed=5)
        fm = FixedAffinity(m, scale=12)
        r = run_from_csv(p, m, fm)
        assert 0.0 <= r["float_acc"] <= 1.0
        assert 0.0 <= r["fixed_acc"] <= 1.0

    def test_qualified_is_bool(self, tmp_path):
        p = _make_csv(str(tmp_path / "kiba_test.csv"), n=4)
        m  = AffinityModel.from_random(seed=6)
        fm = FixedAffinity(m, scale=12)
        r = run_from_csv(p, m, fm)
        assert isinstance(r["qualified"], bool)


class TestCsvRunnerMetrics:
    def test_float_acc_matches_direct_metric(self, tmp_path):
        from reference.metrics import sens_spec_accuracy
        affs = [14.0, 5.0, 13.5, 3.0, 15.0]   # kiba threshold=12.1
        p = _make_csv(str(tmp_path / "kiba_test.csv"), n=5, affinity_vals=affs)
        m  = AffinityModel.from_random(seed=7)
        fm = FixedAffinity(m, scale=12)
        r  = run_from_csv(p, m, fm)
        expected = sens_spec_accuracy(affs, r["float_preds"], 12.1)
        assert abs(r["float_acc"] - expected) < 1e-9

    def test_qualification_consistent_with_gate(self, tmp_path):
        from reference.metrics import is_qualified
        p = _make_csv(str(tmp_path / "kiba_test.csv"), n=5)
        m  = AffinityModel.from_random(seed=8)
        fm = FixedAffinity(m, scale=12)
        r  = run_from_csv(p, m, fm)
        assert r["qualified"] == is_qualified(r["float_acc"], r["fixed_acc"])
