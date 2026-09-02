"""Smoke-test data prep: no pretrained .pth available, so use a randomly
initialised AffinityModel (seed=0) to generate weights.bin + secret shares,
then print the plaintext prediction for comparison with the MPC output.

Avoids `baseline.official_baseline_data` (hardcoded external DeepDTAGen path)
by reading the CSV directly and inlining protein_emb export.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

torch.manual_seed(0)  # GatedCNN init is torch-based; seed it for reproducibility

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from reference.affinity_model import AffinityModel
from reference.fixed_forward import FixedAffinity
from reference import export_weights, share_data, mpc_config, dense_graph

SCALE, BW = 12, 32

def main():
    out_dir = os.path.join(ROOT, "gpu_mpc", "smoke")
    sample_dir = os.path.join(out_dir, "sample_0")
    os.makedirs(sample_dir, exist_ok=True)

    model = AffinityModel.from_random(feat_dim=mpc_config.FEAT_DIM, seed=0)

    weights_path = os.path.join(out_dir, "weights.bin")
    export_weights.dump_mpc_weights(model, weights_path, scale=SCALE)
    print("weights:", weights_path)

    r = pd.read_csv(os.path.join(ROOT, "data", "davis_test.csv")).iloc[0]
    smile, protein_seq, y = str(r["compound_iso_smiles"]), str(r["target_sequence"]), float(r["affinity"])

    share_data.share_drug_graph(smile, sample_dir, scale=SCALE,
                                nmax=mpc_config.NMAX, seed=0,
                                pool_dim=mpc_config.POOL_DIM, bw=BW)

    # inline of protein_plaintext.export_protein_emb (avoids baseline import)
    pvec = model.protein_path(protein_seq)
    fixed = np.rint(np.asarray(pvec, np.float64) * (1 << SCALE)).astype(np.int64)
    ring = np.mod(fixed, np.int64(1) << BW).astype("<u4")
    ring.tofile(os.path.join(sample_dir, mpc_config.PROTEIN_EMB_FILE))
    print("shares + protein_emb ->", sample_dir)

    X, A_hat, mask = dense_graph.smile_to_dense_graph(smile, nmax=mpc_config.NMAX)
    print("plaintext affinity (float):", model.predict(X, A_hat, mask, protein_seq))
    fx32 = FixedAffinity(model, scale=SCALE, bw=BW)
    print("plaintext affinity (fx32): ", fx32.predict(X, A_hat, mask, protein_seq))
    print("label affinity:            ", y)

if __name__ == "__main__":
    main()
