"""
Additive secret-sharing of confidential drug inputs — the P1 (data owner) side.

Mirrors GPU-MPC's `writeSharesCpu` (experiments/*/share_data.cpp): each
plaintext value is quantised to the 64-bit ring at a fixed scale, then split
into two additive shares over Z_{2^64}:

    share1 = random pad
    share2 = (fixed - pad)  mod 2^64
    share1 + share2 ≡ fixed  (mod 2^64)

Only the confidential drug tensors are shared here — node features X, the
normalised adjacency A_hat, and the node mask. Protein sequences are public and
are NOT shared (P2 evaluates the GatedCNN on the cleartext protein and shares
only the resulting Pvec at the fusion boundary).

Shares are written as little-endian uint64 (`<u8`) raw files, one pair per
tensor: `<prefix>_<tensor>_share1.dat` and `<prefix>_<tensor>_share2.dat`.
"""
import numpy as np

from reference.dense_graph import smile_to_dense_graph

U64_MOD = 1 << 64
DEFAULT_SCALE = 24


def _to_ring_u64(x_float: np.ndarray, scale: int) -> np.ndarray:
    """Quantise to fixed-point then reinterpret two's-complement int64 as u64."""
    fixed = np.rint(np.asarray(x_float, dtype=np.float64) * (1 << scale)).astype(np.int64)
    return fixed.astype(np.uint64)            # bit-preserving reinterpretation


def split_shares(x_float, scale: int = DEFAULT_SCALE, seed: int = 0):
    """Split a float array into two additive u64 shares over Z_{2^64}.

    Returns (share1, share2) flat uint64 arrays such that
    share1 + share2 ≡ round(x*2^scale) (mod 2^64).
    """
    x = np.asarray(x_float, dtype=np.float64).ravel()
    fixed_u64 = _to_ring_u64(x, scale)

    rng = np.random.default_rng(seed)
    # full-width random pad in [0, 2^64)
    pad = rng.integers(0, U64_MOD, size=fixed_u64.shape, dtype=np.uint64)

    # share2 = fixed - pad  (mod 2^64); uint64 subtraction wraps correctly
    share1 = pad
    share2 = fixed_u64 - pad                  # numpy uint64 wraps mod 2^64
    return share1, share2


def reconstruct(share1, share2, scale: int = DEFAULT_SCALE) -> np.ndarray:
    """Inverse of split_shares — recombine shares to a float array."""
    s1 = np.asarray(share1, dtype=np.uint64)
    s2 = np.asarray(share2, dtype=np.uint64)
    summed = s1 + s2                          # wraps mod 2^64
    as_int = summed.astype(np.int64)          # reinterpret as two's complement
    return as_int.astype(np.float64) / (1 << scale)


def _write_pair(x_float, prefix: str, tensor: str, scale: int, seed: int):
    s1, s2 = split_shares(x_float, scale=scale, seed=seed)
    s1.astype("<u8").tofile(f"{prefix}_{tensor}_share1.dat")
    s2.astype("<u8").tofile(f"{prefix}_{tensor}_share2.dat")


def share_drug_graph(smile: str, out_prefix: str,
                     scale: int = DEFAULT_SCALE, nmax: int = 138, seed: int = 0):
    """Secret-share the confidential drug graph of `smile` to share files.

    Writes 3 tensor pairs (x, adj, mask). Uses a distinct sub-seed per tensor so
    the random pads are independent.
    """
    X, A_hat, mask = smile_to_dense_graph(smile, nmax)
    _write_pair(X,     out_prefix, "x",    scale, seed + 0)
    _write_pair(A_hat, out_prefix, "adj",  scale, seed + 1)
    _write_pair(mask,  out_prefix, "mask", scale, seed + 2)
    return {"nmax": nmax, "scale": scale,
            "shapes": {"x": list(X.shape), "adj": list(A_hat.shape),
                       "mask": list(mask.shape)}}
