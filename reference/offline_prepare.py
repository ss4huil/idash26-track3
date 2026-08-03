"""Offline driver (Phase A): turn one CSV row into every artifact the online
C++/CUDA binary needs.

Per sample, into `<out_dir>/sample_<row_idx>/`:
  * `{x,adj,mask}_share{0,1}.dat` — additive shares of the CONFIDENTIAL drug
    graph (P1's input; the mask leaks the atom count, so it is shared too).
  * `protein_emb.dat` — the PUBLIC 128-d GatedCNN embedding in fixed point.
    Protein sequences are public, so this path runs in plaintext here and only
    enters MPC as a public constant at the fusion boundary.

Once per run, into `<out_dir>/`:
  * `weights.bin` (+ `weights.bin.json` manifest) — the public fixed-point blob
    of the MPC-secured layers, shared by every sample in the run.

The weight blob is dumped from `AffinityModel.from_pth` (numpy (W, b) groups),
NOT from `official_baseline_data.load_model` — the latter returns a
`(torch DeepDTAGen, device)` pair, which `dump_mpc_weights` cannot walk.
"""
import os

import numpy as np

from reference import mpc_config, share_data, protein_plaintext, export_weights
from reference.affinity_model import AffinityModel
from baseline import official_baseline_data as ob

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
WEIGHTS_FILE = "weights.bin"


def model_pth(dataset: str) -> str:
    """Path to the released checkpoint the fixed-point blob is dumped from."""
    return os.path.normpath(os.path.join(MODEL_DIR, f"deepdtagen_model_{dataset}.pth"))


def export_run_weights(dataset: str, out_dir: str,
                       scale: int = mpc_config.SCALE) -> str:
    """Dump the shared fixed-point weight blob once per run; return its path."""
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, WEIGHTS_FILE)
    if not (os.path.exists(weights_path) and os.path.exists(weights_path + ".json")):
        export_weights.dump_mpc_weights(AffinityModel.from_pth(model_pth(dataset)),
                                        weights_path, scale=scale)
    return weights_path


def prepare_sample(dataset: str, csv_path: str, row_idx: int, out_dir: str,
                   scale: int = mpc_config.SCALE,
                   bw: int = mpc_config.BW) -> dict:
    """Write all online artifacts for CSV row `row_idx`; return a manifest."""
    row = ob.dataset_row(csv_path, row_idx)
    sample_dir = os.path.join(out_dir, f"sample_{row_idx}")

    # confidential drug graph -> additive shares (per-sample seed keeps pads
    # independent across samples in one run)
    share_data.share_drug_graph(row["smile"], sample_dir, scale=scale,
                                nmax=mpc_config.NMAX, seed=row_idx,
                                pool_dim=mpc_config.POOL_DIM, bw=bw)

    # public protein embedding -> fixed-point constant
    ds = ob.build_dataset(dataset, csv_path, limit=row_idx + 1)
    sample = ds[row_idx]
    pvec = protein_plaintext.protein_embedding(dataset, sample)
    protein_path = protein_plaintext.export_protein_emb(pvec, sample_dir,
                                                        scale=scale, bw=bw)

    weights_path = export_run_weights(dataset, out_dir, scale=scale)

    return {
        "sample_dir":   sample_dir,
        "weights_path": weights_path,
        "protein_emb_path": protein_path,
        "smile":        row["smile"],
        "protein_seq":  row["protein_seq"],
        "y":            float(np.asarray(sample.y).reshape(-1)[0]),
        "bw":           int(bw),
        "scale":        int(scale),
        "nmax":         mpc_config.NMAX,
        "dataset":      dataset,
        "row_idx":      int(row_idx),
    }
