#!/usr/bin/env python3
"""
Aggregate validation across all multi-batch davis runs.

Reads davis_multibatch_results.log (tagged AFFINITY outputs) and each batch's
manifest, compares MPC output vs CSV golden labels, reports aggregate MAE/RMSE.

Usage:
  python aggregate_davis_validation.py <num_batches>
"""
import sys
import os
import json
import re
import numpy as np
import pandas as pd

num_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 5

# Paths: script is in scripts/dev_tools/, mpc root is two levels up
MPC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GPU_MPC = os.path.join(MPC_ROOT, "gpu_mpc")
CSV = os.path.join(MPC_ROOT, "data", "davis_test.csv")

df = pd.read_csv(CSV)
golden_all = df['affinity'].values

all_mpc = []
all_golden = []
all_rows = []

for batch in range(num_batches):
    batch_name = f"davis_mb_{batch}"
    manifest_path = os.path.join(GPU_MPC, batch_name, "batch_manifest.json")
    log_path = f"/tmp/mb_{batch}.log"

    if not os.path.exists(manifest_path) or not os.path.exists(log_path):
        print(f"[skip] Batch {batch}: missing manifest or log")
        continue

    with open(manifest_path) as f:
        manifest = json.load(f)
    row_indices = manifest["row_indices"]
    B = len(row_indices)

    with open(log_path) as f:
        log = f.read()

    mpc = [None] * B
    for m in re.finditer(r'AFFINITY\[(\d+)\]=([\d.]+)', log):
        idx, val = int(m.group(1)), float(m.group(2))
        if idx < B:
            mpc[idx] = val

    if any(v is None for v in mpc):
        print(f"[warn] Batch {batch}: incomplete outputs ({mpc.count(None)} missing)")
        continue

    for i in range(B):
        all_mpc.append(mpc[i])
        all_golden.append(golden_all[row_indices[i]])
        all_rows.append(row_indices[i])

all_mpc = np.array(all_mpc)
all_golden = np.array(all_golden)
errors = np.abs(all_mpc - all_golden)

N = len(errors)
if N == 0:
    print("[ERROR] No valid results to aggregate")
    sys.exit(1)

mae = np.mean(errors)
rmse = np.sqrt(np.mean(errors ** 2))

# Pearson correlation (predictive quality metric for affinity)
if N > 1 and np.std(all_mpc) > 0 and np.std(all_golden) > 0:
    pearson = np.corrcoef(all_mpc, all_golden)[0, 1]
else:
    pearson = float('nan')

print(f"\n{'='*50}")
print(f"AGGREGATE DAVIS VALIDATION ({N} samples)")
print(f"{'='*50}")
print(f"MAE:     {mae:.4f} pKd")
print(f"RMSE:    {rmse:.4f} pKd")
print(f"Pearson: {pearson:.4f}")
print(f"Max err: {errors.max():.4f} pKd (row={all_rows[np.argmax(errors)]})")

print(f"\nError percentiles:")
for p in [50, 75, 90, 95, 99]:
    print(f"  {p}th: {np.percentile(errors, p):.4f} pKd")

large = errors > 1.0
print(f"\nSamples with error > 1.0 pKd: {large.sum()}/{N} ({100*large.mean():.1f}%)")
if large.any():
    print("  Worst offenders:")
    worst = np.argsort(errors)[::-1][:5]
    for i in worst:
        print(f"    row={all_rows[i]:4d}: golden={all_golden[i]:5.2f}, " +
              f"mpc={all_mpc[i]:5.2f}, err={errors[i]:.3f}")

print(f"\n{'='*50}")
if mae < 0.5:
    print(f"✅ PASS: MAE={mae:.4f} < 0.5 pKd — excellent MPC fidelity")
elif mae < 1.0:
    print(f"⚠️  MARGINAL: MAE={mae:.4f} — acceptable but review large errors")
else:
    print(f"❌ FAIL: MAE={mae:.4f} >= 1.0 pKd")
print(f"{'='*50}")
