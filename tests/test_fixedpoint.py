"""
TDD – RED phase: tests for fixed-point quantisation (spec §7).

The MPC layer computes over 64-bit integers in fixed-point (scale = 12 by
default). This module is the plaintext golden reference for that arithmetic:
it must reproduce exactly what the GPU-MPC C++ layer does so we can diff the
two. Key operations:

    to_fixed(x, scale)       float  → int64 fixed-point   round(x * 2^scale)
    from_fixed(x, scale)     int64  → float               x / 2^scale
    fixed_matmul(A, B, s)    (A @ B) with one truncation by 2^s (arith. shift)

Run:  python3 -m pytest idash/mpc/tests/test_fixedpoint.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from reference.fixedpoint import (   # not yet implemented
    to_fixed, from_fixed, fixed_matmul, SCALE, BITWIDTH,
)


class TestQuantiseRoundTrip:
    """to_fixed / from_fixed must round-trip within quantisation error."""

    def test_scale_default_is_12(self):
        assert SCALE == 12, "spec fixes default scale = 12"

    def test_bitwidth_is_64(self):
        assert BITWIDTH == 64, "spec fixes 64-bit ring"

    def test_to_fixed_dtype_is_int64(self):
        xf = to_fixed(np.array([1.0, -2.5, 0.0], dtype=np.float32), SCALE)
        assert xf.dtype == np.int64, f"fixed-point must be int64, got {xf.dtype}"

    def test_to_fixed_value(self):
        """round(x * 2^scale)."""
        xf = to_fixed(np.array([1.0, 0.5, -0.25], dtype=np.float64), 12)
        expected = np.array([4096, 2048, -1024], dtype=np.int64)
        np.testing.assert_array_equal(xf, expected)

    def test_round_trip_small_values(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000).astype(np.float64)
        xr = from_fixed(to_fixed(x, SCALE), SCALE)
        # quantisation step is 2^-12 ≈ 2.4e-4; round-trip error is half that
        np.testing.assert_allclose(xr, x, atol=2 ** -(SCALE + 1) + 1e-9)

    def test_round_trip_preserves_zero(self):
        assert from_fixed(to_fixed(np.zeros(5), SCALE), SCALE).tolist() == [0.0] * 5

    def test_to_fixed_rounds_half_correctly(self):
        # 0.5 * 2^1 = 1.0 exactly; test a value that needs rounding
        # 0.1 * 4096 = 409.6 → rounds to 410
        xf = to_fixed(np.array([0.1]), 12)
        assert xf[0] == 410, f"expected 410, got {xf[0]}"


class TestFixedMatmul:
    """fixed_matmul(A, B, scale) = truncate(A @ B, scale), matching MPC."""

    def test_matches_float_matmul_within_tolerance(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((8, 16)).astype(np.float64) * 0.5
        B = rng.standard_normal((16, 4)).astype(np.float64) * 0.5

        # float reference
        C_float = A @ B

        # fixed-point path
        A_fx = to_fixed(A, SCALE)
        B_fx = to_fixed(B, SCALE)
        C_fx = fixed_matmul(A_fx, B_fx, SCALE)      # already truncated once
        C_dequant = from_fixed(C_fx, SCALE)

        # accumulation error grows with inner dim; tolerance ~ K * 2^-scale
        np.testing.assert_allclose(C_dequant, C_float, atol=16 * 2 ** -SCALE)

    def test_output_is_int64(self):
        A_fx = to_fixed(np.ones((2, 3)), SCALE)
        B_fx = to_fixed(np.ones((3, 2)), SCALE)
        C = fixed_matmul(A_fx, B_fx, SCALE)
        assert C.dtype == np.int64

    def test_output_shape(self):
        A_fx = to_fixed(np.ones((5, 7)), SCALE)
        B_fx = to_fixed(np.ones((7, 3)), SCALE)
        C = fixed_matmul(A_fx, B_fx, SCALE)
        assert C.shape == (5, 3)

    def test_truncation_is_arithmetic_shift(self):
        """Truncation must be arithmetic (floor toward -inf) shift by scale.

        A single product a*b at scale s has scale 2s; truncating by s brings
        it back to scale s. We check the exact integer semantics used by MPC:
        (A @ B) >> scale  with arithmetic shift.
        """
        # a = 2.0 (=8192 fx), b = 3.0 (=12288 fx); product = 100663296
        # >> 12 = 24576 fx = 6.0
        A_fx = to_fixed(np.array([[2.0]]), 12)
        B_fx = to_fixed(np.array([[3.0]]), 12)
        C = fixed_matmul(A_fx, B_fx, 12)
        assert C[0, 0] == 24576, f"expected 24576 (=6.0), got {C[0,0]}"
        assert from_fixed(C, 12)[0, 0] == 6.0
