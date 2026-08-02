"""
TDD – RED: additive secret-sharing of confidential drug inputs (P1 side).

`share_data.split_shares(x_float, scale, seed)` quantises a float array to the
64-bit ring and splits it into two additive shares over Z_{2^64}:

    share1 + share2 ≡ round(x * 2^scale)   (mod 2^64)

`share_data.share_drug_graph(smile, out_prefix, scale)` writes the three
confidential drug tensors (node features X, normalised adjacency A_hat, node
mask) as `*_share1.dat` / `*_share2.dat` little-endian u64 files — mirroring
GPU-MPC's writeSharesCpu (share_data.cpp). Protein is public and NOT shared.

Run:  python3 -m pytest idash/mpc/tests/test_share_data.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from reference.share_data import (          # not yet implemented
    split_shares, reconstruct, share_drug_graph, U64_MOD,
)

SCALE = 24


class TestSplitShares:
    def test_shares_reconstruct_to_fixed_point(self):
        x = np.array([1.5, -2.25, 0.0, 3.125], dtype=np.float64)
        s1, s2 = split_shares(x, scale=SCALE, seed=0)
        recon = reconstruct(s1, s2, scale=SCALE)
        assert np.allclose(recon, x, atol=2.0 ** -SCALE * 2)

    def test_shares_are_uint64(self):
        x = np.array([0.1, 0.2, 0.3])
        s1, s2 = split_shares(x, scale=SCALE, seed=1)
        assert s1.dtype == np.uint64
        assert s2.dtype == np.uint64

    def test_shares_look_random_not_trivial(self):
        # share1 should not just equal the plaintext (masked by random pad)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        s1, _ = split_shares(x, scale=SCALE, seed=2)
        fixed = np.rint(x * (1 << SCALE)).astype(np.int64).astype(np.uint64)
        assert not np.array_equal(s1, fixed)

    def test_additive_mod_2_64(self):
        x = np.array([-1.0, 7.0, -3.5], dtype=np.float64)
        s1, s2 = split_shares(x, scale=SCALE, seed=3)
        summed = (s1.astype(object) + s2.astype(object)) % U64_MOD
        fixed = (np.rint(x * (1 << SCALE)).astype(np.int64).astype(object)) % U64_MOD
        assert list(summed) == list(fixed)

    def test_different_seeds_give_different_shares(self):
        x = np.array([1.0, 2.0, 3.0])
        a1, _ = split_shares(x, scale=SCALE, seed=10)
        b1, _ = split_shares(x, scale=SCALE, seed=11)
        assert not np.array_equal(a1, b1)


class TestShareDrugGraph:
    def test_writes_six_share_files(self, tmp_path):
        prefix = str(tmp_path / "drug0")
        share_drug_graph("CCO", prefix, scale=SCALE, nmax=138, seed=0)
        for tensor in ("x", "adj", "mask"):
            assert os.path.exists(f"{prefix}_{tensor}_share1.dat")
            assert os.path.exists(f"{prefix}_{tensor}_share2.dat")

    def test_reconstructed_x_matches_graph(self, tmp_path):
        from reference.dense_graph import smile_to_dense_graph
        prefix = str(tmp_path / "d")
        share_drug_graph("CC(=O)O", prefix, scale=SCALE, nmax=138, seed=5)
        X, A_hat, mask = smile_to_dense_graph("CC(=O)O", 138)
        s1 = np.fromfile(f"{prefix}_x_share1.dat", dtype="<u8")
        s2 = np.fromfile(f"{prefix}_x_share2.dat", dtype="<u8")
        recon = reconstruct(s1, s2, scale=SCALE).reshape(X.shape)
        assert np.allclose(recon, X, atol=2.0 ** -SCALE * 4)

    def test_reconstructed_adj_matches(self, tmp_path):
        from reference.dense_graph import smile_to_dense_graph
        prefix = str(tmp_path / "d")
        share_drug_graph("c1ccccc1", prefix, scale=SCALE, nmax=138, seed=6)
        _, A_hat, _ = smile_to_dense_graph("c1ccccc1", 138)
        s1 = np.fromfile(f"{prefix}_adj_share1.dat", dtype="<u8")
        s2 = np.fromfile(f"{prefix}_adj_share2.dat", dtype="<u8")
        recon = reconstruct(s1, s2, scale=SCALE).reshape(A_hat.shape)
        assert np.allclose(recon, A_hat, atol=2.0 ** -SCALE * 4)
