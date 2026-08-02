"""
TDD – RED: reference-compatible NPZ export (spec §8 / BumbleBee driver format).

The official BumbleBee driver loads a single NPZ file per dataset with arrays:
  drug_x    (N, 138, 94)  float32  L1-normalised atom features (NO pre-norm A_hat)
  adj       (N, 138, 138) float32  RAW adjacency WITHOUT self-loops
  node_mask (N, 138)      bool     True for real atoms
  protein   (N, 1000)     int32    encoded protein (seq_cat integers)
  y         (N, 1)        float32  affinity label

This module produces that format from (SMILES, protein_str, y) tuples so we
can run our predictions through the official driver for apples-to-apples
comparison once weights land.

Run:  python3 -m pytest idash/mpc/tests/test_export_npz.py -v
"""
import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from reference.export_npz import to_npz_record, export_npz  # not yet implemented

NMAX = 138
PAIRS = [
    ("CCO",                           "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIG", 7.5),
    ("CC(=O)Nc1ccc(O)cc1",            "MABCDEFGHIKLMNPQRST", 5.2),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "QWERTYUIOP", 12.3),
]


class TestNpzRecord:
    def test_drug_x_shape(self):
        rec = to_npz_record(*PAIRS[0], nmax=NMAX)
        assert rec["drug_x"].shape == (NMAX, 94)

    def test_drug_x_dtype_float32(self):
        rec = to_npz_record(*PAIRS[0], nmax=NMAX)
        assert rec["drug_x"].dtype == np.float32

    def test_adj_raw_no_self_loop(self):
        """Reference NPZ stores raw adjacency, no self-loops (added by GCNLayer)."""
        rec = to_npz_record(*PAIRS[0], nmax=NMAX)
        adj = rec["adj"]
        # diagonal must be zero everywhere (no self-loops in export)
        assert np.all(np.diag(adj) == 0), "NPZ adj must not contain self-loops"

    def test_adj_shape_and_dtype(self):
        rec = to_npz_record(*PAIRS[0], nmax=NMAX)
        assert rec["adj"].shape == (NMAX, NMAX)
        assert rec["adj"].dtype == np.float32

    def test_adj_symmetric(self):
        rec = to_npz_record(*PAIRS[2], nmax=NMAX)  # caffeine, more bonds
        np.testing.assert_array_equal(rec["adj"], rec["adj"].T)

    def test_node_mask_bool_and_shape(self):
        rec = to_npz_record(*PAIRS[0], nmax=NMAX)
        assert rec["node_mask"].shape == (NMAX,)
        assert rec["node_mask"].dtype == bool

    def test_node_mask_correct_count(self):
        """CCO = 3 atoms (C, C, O) → 3 True entries."""
        rec = to_npz_record("CCO", "AAA", 1.0, nmax=NMAX)
        assert rec["node_mask"].sum() == 3

    def test_protein_shape_int32(self):
        rec = to_npz_record(*PAIRS[0], nmax=NMAX)
        assert rec["protein"].shape == (1000,)
        assert rec["protein"].dtype == np.int32

    def test_protein_nonzero_for_known_aa(self):
        """Encoding maps known amino acid letters to nonzero integers."""
        rec = to_npz_record("CCO", "MKKFF", 0.0, nmax=NMAX)
        assert rec["protein"][:5].sum() > 0

    def test_y_float32(self):
        rec = to_npz_record(*PAIRS[0], nmax=NMAX)
        assert rec["y"].dtype == np.float32

    def test_drug_x_l1_normalised(self):
        """Real-atom rows must be L1-normalised (each sums to ~1.0)."""
        rec = to_npz_record("c1ccccc1", "AAA", 0.0, nmax=NMAX)
        mask = rec["node_mask"]
        row_sums = rec["drug_x"][mask].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_padding_rows_zero(self):
        rec = to_npz_record("CCO", "AAA", 0.0, nmax=NMAX)
        pad = ~rec["node_mask"]
        assert np.all(rec["drug_x"][pad] == 0.0)
        assert np.all(rec["adj"][pad, :] == 0.0)
        assert np.all(rec["adj"][:, pad] == 0.0)


class TestExportBatch:
    def test_export_npz_shapes(self, tmp_path):
        out = str(tmp_path / "test.npz")
        export_npz(PAIRS, out, nmax=NMAX)
        d = np.load(out, allow_pickle=False)
        n = len(PAIRS)
        assert d["drug_x"].shape    == (n, NMAX, 94)
        assert d["adj"].shape       == (n, NMAX, NMAX)
        assert d["node_mask"].shape == (n, NMAX)
        assert d["protein"].shape   == (n, 1000)
        assert d["y"].shape         == (n, 1)

    def test_export_npz_dtypes(self, tmp_path):
        out = str(tmp_path / "dtypes.npz")
        export_npz(PAIRS, out, nmax=NMAX)
        d = np.load(out, allow_pickle=False)
        assert d["drug_x"].dtype    == np.float32
        assert d["adj"].dtype       == np.float32
        assert d["node_mask"].dtype == bool
        assert d["protein"].dtype   == np.int32
        assert d["y"].dtype         == np.float32

    def test_round_trip_preserves_values(self, tmp_path):
        out = str(tmp_path / "rt.npz")
        export_npz(PAIRS, out, nmax=NMAX)
        d = np.load(out, allow_pickle=False)
        recs = [to_npz_record(*p, nmax=NMAX) for p in PAIRS]
        for i, rec in enumerate(recs):
            np.testing.assert_array_equal(d["drug_x"][i],    rec["drug_x"])
            np.testing.assert_array_equal(d["adj"][i],       rec["adj"])
            np.testing.assert_array_equal(d["node_mask"][i], rec["node_mask"])
            np.testing.assert_array_equal(d["y"][i, 0],      rec["y"])
