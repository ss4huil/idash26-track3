#!/usr/bin/env python3
"""
Validate MPC inference accuracy for a raw-adj batch (any dataset) by comparing
MPC output against the fixed-point plaintext reference.

The reference mirrors exactly what the MPC fusion consumed:
  - drug path: FixedAffinity._drug_path_fx over the *normalised* dense graph
    (smile_to_dense_graph) — the MPC computes the same normalisation online
    from the raw 0/1 adjacency (DDG_SECURE_ADJ_NORM=1 path);
  - protein vector: read back from the batch's protein_emb.dat (the actual
    fixed-point public input party 1 fed into the fusion FCs).

Usage:
  python validate_batch_fixed.py <batch_name> <mpc_output_log>

Example:
  python validate_batch_fixed.py timing_b8_raw_kiba /tmp/kb_k_0.log
"""
import os
import re
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from reference.affinity_model import AffinityModel
from reference.dense_graph import smile_to_dense_graph
from reference.fixed_forward import FixedAffinity
from reference.fixedpoint import from_fixed
from reference import mpc_config

TOL = 0.005  # per-sample |MPC - fixed reference| tolerance


def read_protein_fx(batch_dir, B, scale, bw):
    """Read protein_emb.dat → (B, 128) signed int64 fixed-point ring values."""
    raw = np.fromfile(os.path.join(batch_dir, "protein_emb.dat"),
                      dtype=f"<u{bw // 8}").reshape(B, 128).astype(np.int64)
    modulus = np.int64(1) << bw
    half = modulus >> 1
    return np.where(raw >= half, raw - modulus, raw)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    batch_name = sys.argv[1]
    mpc_log = sys.argv[2]

    batch_dir = os.path.join("gpu_mpc", batch_name)
    with open(os.path.join(batch_dir, "batch_manifest.json")) as f:
        manifest = json.load(f)

    dataset = manifest["dataset"]
    row_indices = manifest["row_indices"]
    smiles_list = manifest["smiles"]
    scale = manifest["scale"]
    bw = manifest["bw"]
    nmax = manifest["nmax"]
    B = len(row_indices)

    print(f"[validate] Batch: {batch_name}, dataset: {dataset}, B={B}, "
          f"scale={scale}, bw={bw}")

    # Parse MPC output
    with open(mpc_log) as f:
        log_content = f.read()
    mpc_vals = [None] * B
    for m in re.finditer(r'AFFINITY\[(\d+)\]=(-?[\d.]+)', log_content):
        idx = int(m.group(1))
        if idx < B:
            mpc_vals[idx] = float(m.group(2))
    if any(v is None for v in mpc_vals):
        print("[ERROR] failed to parse all AFFINITY[i] lines from log")
        sys.exit(1)

    # Fixed-point reference
    model_pth = os.path.join(os.path.dirname(__file__), "../..",
                             "model", f"deepdtagen_model_{dataset}.pth")
    fm = FixedAffinity(AffinityModel.from_pth(model_pth), scale=scale, bw=bw)
    pvec_fx = read_protein_fx(batch_dir, B, scale, bw)

    ref_vals = []
    for i, smile in enumerate(smiles_list):
        X, A_hat, mask = smile_to_dense_graph(smile, nmax)
        pmvo_fx = fm._drug_path_fx(X, A_hat, mask)
        out_fx = fm._fusion_fx(pmvo_fx, pvec_fx[i])
        ref_vals.append(float(from_fixed(out_fx, scale)[0]))
    ref_vals = np.array(ref_vals)
    mpc_vals = np.array(mpc_vals)

    errors = np.abs(mpc_vals - ref_vals)
    max_err = errors.max()

    print(f"\n=== {dataset} MPC vs fixed-point reference ===")
    print(f"{'i':>2} {'row':>4} {'MPC':>12} {'reference':>12} {'|err|':>10}")
    for i in range(B):
        flag = "OK " if errors[i] <= TOL else "FAIL"
        print(f"{i:>2} {row_indices[i]:>4} {mpc_vals[i]:>12.6f} "
              f"{ref_vals[i]:>12.6f} {errors[i]:>10.6f} {flag}")

    # garbage / constant-output sanity check
    distinct = len(np.unique(np.round(mpc_vals, 4)))
    print(f"\nmax |err| = {max_err:.6f}  (tolerance {TOL})")
    print(f"distinct MPC outputs: {distinct}/{B}")

    passed = max_err <= TOL and distinct > 1
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
