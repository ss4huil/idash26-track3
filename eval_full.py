#!/usr/bin/env python
"""
Full-dataset MPC accuracy evaluator for iDASH Track 3.

Optimized: caches drug-path PMVO per unique SMILES and protein Pvec per
unique sequence, so each row only requires the cheap fusion FC pass.

Usage:
  ~/.pyenv/versions/3.8.7/bin/python idash/mpc/eval_full.py
"""
import sys, os, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
from reference.affinity_model import AffinityModel
from reference.fixed_forward  import FixedAffinity, _relu_fx
from reference.dense_graph    import smile_to_dense_graph
from reference.fixedpoint     import to_fixed, from_fixed, fixed_matmul
from reference.metrics        import (threshold_for, sensitivity, specificity,
                                      sens_spec_accuracy, is_qualified)

NMAX = 138
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
TEST_DIR  = os.path.join(os.path.dirname(__file__), "../project/test")

CONFIGS = [
    ("davis", 24, 64),
    ("davis", 12, 32),
    ("kiba",  24, 64),
    ("kiba",  12, 32),
]


def run_dataset(dataset, scale, bw):
    pth = os.path.join(MODEL_DIR, f"deepdtagen_model_{dataset}.pth")
    csv_path = os.path.join(TEST_DIR, f"{dataset}_test.csv")
    if not (os.path.exists(pth) and os.path.exists(csv_path)):
        print(f"  [{dataset} bw={bw}] SKIP: files missing"); return

    m  = AffinityModel.from_pth(pth)
    fm = FixedAffinity(m, scale=scale, bw=bw)
    thr = threshold_for(dataset)
    n_fusion = len(m.fusion)

    # --- Read rows ---
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["compound_iso_smiles"], r["target_sequence"],
                         float(r["affinity"])))
    true_labels = np.array([y for _, _, y in rows], dtype=np.float64)

    print(f"  [{dataset} bw={bw} scale={scale}] {len(rows)} rows, "
          f"{len(set(s for s,_,_ in rows))} SMILES, "
          f"{len(set(p for _,p,_ in rows))} proteins ...")
    sys.stdout.flush()

    # --- Cache drug-path outputs per unique SMILES ---
    t0 = time.perf_counter()
    pmvo_float_cache = {}     # smiles -> (128,) float
    pmvo_fixed_cache = {}     # smiles -> (128,) int64
    for sm, _, _ in rows:
        if sm not in pmvo_float_cache:
            X, A, mask = smile_to_dense_graph(sm, NMAX)
            pmvo_float_cache[sm] = m.drug_path(X, A, mask)
            pmvo_fixed_cache[sm] = fm._drug_path_fx(X, A, mask)
    dt_drug = time.perf_counter() - t0
    print(f"    drug-path cache: {len(pmvo_float_cache)} entries in {dt_drug:.1f}s")
    sys.stdout.flush()

    # --- Cache protein-path outputs per unique sequence ---
    t0 = time.perf_counter()
    pvec_float_cache = {}     # seq -> (128,) float
    pvec_fixed_cache = {}     # seq -> (128,) int64
    for _, seq, _ in rows:
        if seq not in pvec_float_cache:
            pvec_float_cache[seq] = m.protein_path(seq)
            pvec_fixed_cache[seq] = fm._protein_vec_fx(seq)
    dt_prot = time.perf_counter() - t0
    print(f"    protein cache:   {len(pvec_float_cache)} entries in {dt_prot:.1f}s")
    sys.stdout.flush()

    # --- Run fusion over all rows (cheap: just FC layers) ---
    t0 = time.perf_counter()
    yf = np.empty(len(rows), dtype=np.float64)
    yq = np.empty(len(rows), dtype=np.float64)
    for i, (sm, seq, _) in enumerate(rows):
        # float fusion
        hf = np.concatenate([pmvo_float_cache[sm], pvec_float_cache[seq]])
        for k, (W, b) in enumerate(m.fusion):
            hf = hf @ W.T + b
            if k < n_fusion - 1: hf = np.maximum(hf, 0)
        yf[i] = float(hf[0])
        # fixed fusion
        yq[i] = float(from_fixed(
            fm._fusion_fx(pmvo_fixed_cache[sm], pvec_fixed_cache[seq]), scale)[0])
    dt_fuse = time.perf_counter() - t0

    # --- Metrics ---
    fa = sens_spec_accuracy(true_labels, yf, thr)
    qa = sens_spec_accuracy(true_labels, yq, thr)
    sens_f = sensitivity(true_labels, yf, thr)
    spec_f = specificity(true_labels, yf, thr)
    sens_q = sensitivity(true_labels, yq, thr)
    spec_q = specificity(true_labels, yq, thr)
    drop = (fa - qa) * 100
    ok = is_qualified(fa, qa)

    print(f"\n  === {dataset.upper()} bw={bw} scale={scale} (n={len(rows)}) ===")
    print(f"    Float : acc={fa:.4f}  sens={sens_f:.4f}  spec={spec_f:.4f}")
    print(f"    Fixed : acc={qa:.4f}  sens={sens_q:.4f}  spec={spec_q:.4f}")
    print(f"    Drop  : {drop:+.2f}pp  gate: {'✅ QUALIFIED' if ok else '❌ FAILED (>2pp)'}")
    print(f"    Times : drug={dt_drug:.1f}s  prot={dt_prot:.1f}s  fusion={dt_fuse:.2f}s\n")
    sys.stdout.flush()
    return {"dataset": dataset, "bw": bw, "scale": scale,
            "n": len(rows), "float_acc": fa, "fixed_acc": qa, "drop_pp": drop,
            "qualified": ok}


if __name__ == "__main__":
    print("=== iDASH Track 3 MPC Framework Accuracy Evaluation ===\n")
    results = []
    for ds, sc, bw in CONFIGS:
        r = run_dataset(ds, sc, bw)
        if r: results.append(r)

    print("\n=== SUMMARY ===")
    print(f"{'Dataset':10} {'BW':4} {'Scale':6} {'Float':7} {'Fixed':7} {'Drop':8} {'Gate'}")
    print("-" * 60)
    for r in results:
        verdict = "PASS ✅" if r["qualified"] else "FAIL ❌"
        print(f"{r['dataset']:10} {r['bw']:<4} {r['scale']:<6} "
              f"{r['float_acc']:.4f}  {r['fixed_acc']:.4f}  "
              f"{r['drop_pp']:+5.2f}pp  {verdict}")
