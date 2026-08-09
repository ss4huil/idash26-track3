#!/usr/bin/env python3
"""
Benchmark plaintext affinity-only inference time for single samples.
Uses AffinityModel (GCN + fusion FC) — the exact path MPC computes,
NOT the generative decoder. Timed on the same davis_test rows the MPC
samples were generated from (0, 3, 5, 9).
"""
import sys, time, csv
import numpy as np

sys.path.insert(0, '/home/jiang/master/idash/mpc')

from reference.affinity_model import AffinityModel
from reference.dense_graph import smile_to_dense_graph

CSV = "/home/jiang/master/idash/project/DeepDTAGen/data/davis_test.csv"
PTH = "/home/jiang/master/idash/mpc/model/deepdtagen_model_davis.pth"
NMAX = 138
ROWS = [0, 3, 5, 9]

def main():
    print(f"Loading affinity model: {PTH}")
    model = AffinityModel.from_pth(PTH)

    rows = list(csv.DictReader(open(CSV, newline="")))

    print("\n=== Plaintext single-sample inference time (affinity-only path) ===")
    print("3 warmup + 20 measured runs per sample\n")

    all_times = []
    for idx in ROWS:
        r = rows[idx]
        smile = r["compound_iso_smiles"]
        protein = r["target_sequence"]
        golden = float(r["affinity"])

        X, A_hat, mask = smile_to_dense_graph(smile, NMAX)

        for _ in range(3):
            _ = model.predict(X, A_hat, mask, protein)

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            pred = model.predict(X, A_hat, mask, protein)
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms, std_ms = np.mean(times), np.std(times)
        all_times.extend(times)
        print(f"row {idx:>2}: {mean_ms:7.2f} +/- {std_ms:5.2f} ms  "
              f"(pred={pred:.3f}, golden={golden:.3f})")

    m, s = np.mean(all_times), np.std(all_times)
    print(f"\nOverall: {m:.2f} +/- {s:.2f} ms/sample  ({m/1000:.4f} s/sample)")

if __name__ == "__main__":
    main()
