"""
Fixed-point quantisation — plaintext golden reference for the MPC layer (spec §7).

The GPU-MPC backend computes over a 64-bit integer ring in fixed-point with a
fixed scale (default 12). This module reproduces that arithmetic exactly so the
plaintext reference and the C++ MPC output can be diffed bit-for-bit:

    to_fixed(x, scale)     : float  → int64   round(x * 2^scale)
    from_fixed(x, scale)   : int64  → float   x / 2^scale
    fixed_matmul(A, B, s)  : (A @ B) truncated once by 2^s (arithmetic shift)

Truncation is an arithmetic right shift (floor toward -inf), matching the
local-truncation semantics used by the FSS protocol (gpu_local_truncate.h /
gpu_truncate.h). Sign is preserved because numpy's >> on int64 is arithmetic.
"""
import numpy as np

# ── ring / scale constants (spec §7) ─────────────────────────────────────────
BITWIDTH = 64          # 64-bit ring (u64 in the C++ backend)
SCALE    = 12          # fixed-point fractional bits


def to_fixed(x, scale: int = SCALE) -> np.ndarray:
    """Quantise float → int64 fixed-point: round(x * 2^scale).

    Uses round-half-away-from-zero via np.round (banker's rounding in numpy is
    round-half-to-even; we match the C++ `llround` which is half-away-from-zero
    by adding/subtracting 0.5 before truncation).
    """
    x = np.asarray(x, dtype=np.float64)
    scaled = x * float(1 << scale)
    # round half away from zero (matches C++ llround)
    fixed = np.where(scaled >= 0,
                     np.floor(scaled + 0.5),
                     np.ceil(scaled - 0.5))
    return fixed.astype(np.int64)


def from_fixed(x, scale: int = SCALE) -> np.ndarray:
    """Dequantise int64 fixed-point → float: x / 2^scale."""
    x = np.asarray(x, dtype=np.int64)
    return x.astype(np.float64) / float(1 << scale)


def truncate(x, scale: int = SCALE) -> np.ndarray:
    """Arithmetic right shift by `scale` (floor toward -inf), int64.

    Matches the MPC local-truncation: a value at scale 2s is brought back to
    scale s by dropping the low `scale` bits. numpy >> on a signed dtype is an
    arithmetic shift, so the sign is preserved and negatives floor correctly.
    """
    x = np.asarray(x, dtype=np.int64)
    return x >> np.int64(scale)


def fixed_matmul(A, B, scale: int = SCALE) -> np.ndarray:
    """Fixed-point matrix multiply with one truncation.

    A (m,k) and B (k,n) are int64 at scale `scale`; their integer product is at
    scale 2*scale, so we truncate once by `scale` to return to scale `scale`.

    The integer accumulation A @ B is done in Python-object arithmetic to avoid
    int64 overflow during the sum, then truncated and cast back to int64 (the
    final result is guaranteed to fit the ring for realistic model magnitudes).
    """
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    # accumulate in object dtype (arbitrary precision) to avoid overflow, then
    # arithmetic-shift-truncate. Python's >> on ints floors toward -inf.
    prod = A.astype(object) @ B.astype(object)      # scale = 2*scale
    trunc = np.vectorize(lambda v: v >> scale)(prod)  # floor toward -inf
    return trunc.astype(np.int64)
