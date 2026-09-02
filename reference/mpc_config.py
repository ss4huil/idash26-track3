"""Single source of truth for the DeepDTAGen MPC fixed-point + file-naming
contract. Both the Python offline pipeline and the C++/CUDA online binary must
agree on these values (see gpu_mpc/deepdtagen_inference.cu)."""

BW = 64            # ring Z_{2^64} — v2 production path (LocalARS truncation
                   # requires bw=64; see design doc §2). BW=32 is test-only.
SCALE = 12         # Q51.12 fixed-point fractional bits (Q20.12 at bw=32)
NMAX = 138         # padded graph nodes
FEAT_DIM = 94      # node feature width
POOL_DIM = 376     # final GCN width == pooled embedding width

PROTEIN_EMB_FILE = "protein_emb.dat"

def share_filename(tensor: str, party: int) -> str:
    """0-based, prefix-free name the C++ loader reads (deepdtagen_inference.cu)."""
    assert party in (0, 1)
    return f"{tensor}_share{party}.dat"

def key_filename(bw: int = BW, scale: int = SCALE) -> str:
    """Must equal the C++ expName: 'DeepDTAGen_' + bw + '_' + scale."""
    return f"DeepDTAGen_{bw}_{scale}"
