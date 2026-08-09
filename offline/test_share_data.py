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
        s1, s2 = split_shares(x, scale=SCALE, seed=1, bw=64)
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
        s1, s2 = split_shares(x, scale=SCALE, seed=3, bw=64)
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
        out_dir = str(tmp_path)
        share_drug_graph("CCO", out_dir, scale=SCALE, nmax=138, seed=0, bw=64)
        for tensor in ("x", "adj", "mask"):
            assert os.path.exists(os.path.join(out_dir, f"{tensor}_share0.dat"))
            assert os.path.exists(os.path.join(out_dir, f"{tensor}_share1.dat"))

    def test_reconstructed_x_matches_graph(self, tmp_path):
        from reference.dense_graph import smile_to_dense_graph
        out_dir = str(tmp_path)
        share_drug_graph("CC(=O)O", out_dir, scale=SCALE, nmax=138, seed=5, bw=64)
        X, A_hat, mask = smile_to_dense_graph("CC(=O)O", 138)
        s0 = np.fromfile(os.path.join(out_dir, "x_share0.dat"), dtype="<u8")
        s1 = np.fromfile(os.path.join(out_dir, "x_share1.dat"), dtype="<u8")
        recon = reconstruct(s0, s1, scale=SCALE, bw=64).reshape(X.shape)
        assert np.allclose(recon, X, atol=2.0 ** -SCALE * 4)

    def test_reconstructed_adj_matches(self, tmp_path):
        from reference.dense_graph import smile_to_dense_graph
        out_dir = str(tmp_path)
        share_drug_graph("c1ccccc1", out_dir, scale=SCALE, nmax=138, seed=6, bw=64)
        _, A_hat, _ = smile_to_dense_graph("c1ccccc1", 138)
        s0 = np.fromfile(os.path.join(out_dir, "adj_share0.dat"), dtype="<u8")
        s1 = np.fromfile(os.path.join(out_dir, "adj_share1.dat"), dtype="<u8")
        recon = reconstruct(s0, s1, scale=SCALE, bw=64).reshape(A_hat.shape)
        assert np.allclose(recon, A_hat, atol=2.0 ** -SCALE * 4)

    def test_mask_is_emitted_pre_tiled(self, tmp_path):
        # C++ masked-max-pool multiplies mask*(H:(nmax,376)); sytorch _Mul
        # forbids broadcast, so the share must already be tiled (nmax, 376).
        from reference.dense_graph import smile_to_dense_graph
        out_dir = str(tmp_path)
        POOL = 376
        share_drug_graph("CC(=O)O", out_dir, scale=SCALE, nmax=138,
                         seed=7, pool_dim=POOL, bw=64)
        _, _, mask = smile_to_dense_graph("CC(=O)O", 138)
        expected = np.broadcast_to(mask.reshape(138, 1), (138, POOL))
        s0 = np.fromfile(os.path.join(out_dir, "mask_share0.dat"), dtype="<u8")
        s1 = np.fromfile(os.path.join(out_dir, "mask_share1.dat"), dtype="<u8")
        assert s0.size == 138 * POOL
        recon = reconstruct(s0, s1, scale=SCALE, bw=64).reshape(138, POOL)
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
        out_dir = str(tmp_path)
        share_drug_graph("CCO", out_dir, scale=self.SCALE32, nmax=138,
                         seed=0, bw=32)
        # X: (138, 94) → 138*94 = 12972 u32 elements → 51888 bytes
        expected_x = 138 * 94 * 4
        actual = os.path.getsize(os.path.join(out_dir, "x_share0.dat"))
        assert actual == expected_x, \
            f"bw=32 x_share0.dat: expected {expected_x} bytes, got {actual}"

    def test_share_drug_graph_bw32_reconstructed(self, tmp_path):
        # Break: reconstructed values must match the quantised graph within one LSB.
        from reference.dense_graph import smile_to_dense_graph
        out_dir = str(tmp_path)
        share_drug_graph("CC(=O)O", out_dir, scale=self.SCALE32, nmax=138,
                         seed=2, bw=32)
        X, _, _ = smile_to_dense_graph("CC(=O)O", 138)
        s0 = np.fromfile(os.path.join(out_dir, "x_share0.dat"), dtype="<u4")
        s1 = np.fromfile(os.path.join(out_dir, "x_share1.dat"), dtype="<u4")
        recon = reconstruct(s0, s1, scale=self.SCALE32, bw=32).reshape(X.shape)
        assert np.allclose(recon, X, atol=2.0 ** -self.SCALE32 * 4)


# Task 2: verify share-file naming contract matches C++ loader
from reference import mpc_config


class TestCppContract:
    """Verify share files match the C++ loader contract (0-based, prefix-free)."""

    def test_share_files_use_zero_based_cpp_names(self, tmp_path):
        out = str(tmp_path)
        share_drug_graph("CCO", out, bw=32, scale=12, seed=1)
        for tensor in ("x", "adj", "mask"):
            for party in (0, 1):
                assert os.path.exists(os.path.join(out, mpc_config.share_filename(tensor, party)))

    def test_shares_reconstruct_to_fixed_point(self, tmp_path):
        # Break: bw=32 shares must reconstruct to the original graph via the
        # real reconstruct() helper, exercising the 0-based naming contract.
        from reference.dense_graph import smile_to_dense_graph
        out = str(tmp_path)
        share_drug_graph("CCO", out, bw=32, scale=12, seed=1)
        X, _, _ = smile_to_dense_graph("CCO", 138)
        s0 = np.fromfile(os.path.join(out, "x_share0.dat"), dtype="<u4")
        s1 = np.fromfile(os.path.join(out, "x_share1.dat"), dtype="<u4")
        recon = reconstruct(s0, s1, scale=12, bw=32).reshape(X.shape)
        assert np.allclose(recon, X, atol=2.0 ** -12 * 4)
