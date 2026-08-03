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

    def test_mask_is_emitted_pre_tiled(self, tmp_path):
        # C++ masked-max-pool multiplies mask*(H:(nmax,376)); sytorch _Mul
        # forbids broadcast, so the share must already be tiled (nmax, 376).
        from reference.dense_graph import smile_to_dense_graph
        prefix = str(tmp_path / "d")
        POOL = 376
        share_drug_graph("CC(=O)O", prefix, scale=SCALE, nmax=138,
                         seed=7, pool_dim=POOL)
        _, _, mask = smile_to_dense_graph("CC(=O)O", 138)
        expected = np.broadcast_to(mask.reshape(138, 1), (138, POOL))
        s1 = np.fromfile(f"{prefix}_mask_share1.dat", dtype="<u8")
        s2 = np.fromfile(f"{prefix}_mask_share2.dat", dtype="<u8")
        assert s1.size == 138 * POOL
        recon = reconstruct(s1, s2, scale=SCALE).reshape(138, POOL)
        assert np.allclose(recon, expected, atol=2.0 ** -SCALE * 4)
        # every column is an identical copy of the (nmax,) node mask
        assert np.array_equal(recon[:, 0], recon[:, POOL - 1])


class TestBW32Shares:
    """32-bit ring: bw=32 must emit 4-byte-per-element files (u32) and
    reconstruct correctly over Z_{2^32}."""

    SCALE32 = 12   # must satisfy scale < bw=32; Q19.12 for 32-bit ring

    def test_split_shares_bw32_dtype(self):
        # Break: bw=32 with u64 dtype wastes half the bandwidth and the C++
        # side reads wrong-width elements.
        x = np.array([1.5, -2.25, 0.0, 3.0])
        s1, s2 = split_shares(x, scale=self.SCALE32, seed=0, bw=32)
        assert s1.dtype == np.uint32, f"expected uint32, got {s1.dtype}"
        assert s2.dtype == np.uint32

    def test_split_shares_bw32_reconstruct(self):
        # Break: incorrect ring arithmetic would fail round-trip reconstruction.
        x = np.array([1.5, -2.25, 0.0, 3.0, -1.0])
        s1, s2 = split_shares(x, scale=self.SCALE32, seed=1, bw=32)
        recon = reconstruct(s1, s2, scale=self.SCALE32, bw=32)
        assert np.allclose(recon, x, atol=2.0 ** -self.SCALE32 * 2)

    def test_share_drug_graph_bw32_file_size(self, tmp_path):
        # Break: bw=32 share files must be 4*N bytes (not 8*N).
        prefix = str(tmp_path / "d")
        share_drug_graph("CCO", prefix, scale=self.SCALE32, nmax=138,
                         seed=0, bw=32)
        # X: (138, 94) → 138*94 = 12972 u32 elements → 51888 bytes
        expected_x = 138 * 94 * 4
        actual = os.path.getsize(f"{prefix}_x_share1.dat")
        assert actual == expected_x, \
            f"bw=32 x_share1.dat: expected {expected_x} bytes, got {actual}"

    def test_share_drug_graph_bw32_reconstructed(self, tmp_path):
        # Break: reconstructed values must match the quantised graph within one LSB.
        from reference.dense_graph import smile_to_dense_graph
        prefix = str(tmp_path / "d")
        share_drug_graph("CC(=O)O", prefix, scale=self.SCALE32, nmax=138,
                         seed=2, bw=32)
        X, _, _ = smile_to_dense_graph("CC(=O)O", 138)
        s1 = np.fromfile(f"{prefix}_x_share1.dat", dtype="<u4")
        s2 = np.fromfile(f"{prefix}_x_share2.dat", dtype="<u4")
        recon = reconstruct(s1, s2, scale=self.SCALE32, bw=32).reshape(X.shape)
        assert np.allclose(recon, X, atol=2.0 ** -self.SCALE32 * 4)
