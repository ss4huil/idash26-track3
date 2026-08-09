#!/usr/bin/env python3
"""
Validate MPC inference accuracy on a batch from davis test set by comparing
MPC output with golden CSV labels.

Usage:
  python validate_davis_batch.py <batch_name> <mpc_output_log> <golden_csv>

Example:
  BATCH=4 ./run_local_2pc.sh test_batch_4 /tmp/keys /weights.bin > mpc_output.log
  python validate_davis_batch.py test_batch_4 mpc_output.log \
      /home/jiang/master/idash/project/DeepDTAGen/data/davis_test.csv
"""
import sys
import json
import re
import numpy as np
import pandas as pd

if len(sys.argv) < 4:
    print(__doc__)
    sys.exit(1)

batch_name = sys.argv[1]
mpc_output_log = sys.argv[2]
golden_csv = sys.argv[3]

# Load batch manifest to get row indices
manifest_path = f"gpu_mpc/{batch_name}/batch_manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

row_indices = manifest["row_indices"]
B = len(row_indices)

print(f"[validate] Batch: {batch_name}, size: {B}")
print(f"[validate] Row indices: {row_indices}")

# Parse MPC output from log file (AFFINITY[i]=value or AFFINITY=value for B=1)
with open(mpc_output_log) as f:
    log_content = f.read()

mpc_affinities = [None] * B
# Match AFFINITY[0]=5.064, AFFINITY[1]=8.402, etc.
for match in re.finditer(r'AFFINITY\[(\d+)\]=([\d.]+)', log_content):
    idx = int(match.group(1))
    value = float(match.group(2))
    if idx < B:
        mpc_affinities[idx] = value

# Also handle B=1 case: AFFINITY=8.402
if B == 1 and mpc_affinities[0] is None:
    match = re.search(r'^AFFINITY=([\d.]+)', log_content, re.MULTILINE)
    if match:
        mpc_affinities[0] = float(match.group(1))

mpc_affinities = np.array(mpc_affinities, dtype=float)

if np.isnan(mpc_affinities).any():
    print(f"[ERROR] Failed to parse {np.isnan(mpc_affinities).sum()} MPC outputs")
    sys.exit(1)

# Load golden affinities from CSV
df = pd.read_csv(golden_csv)
golden_batch = df.iloc[row_indices]['affinity'].values

# Compute metrics
errors = np.abs(mpc_affinities - golden_batch)
mae = np.mean(errors)
rmse = np.sqrt(np.mean(errors ** 2))
max_error = np.max(errors)
max_error_idx = np.argmax(errors)

print(f"\n=== Validation Results ===")
print(f"Samples: {B}")
print(f"MAE:  {mae:.4f} pKd")
print(f"RMSE: {rmse:.4f} pKd")
print(f"Max error: {max_error:.4f} pKd at batch_idx={max_error_idx} " +
      f"(row={row_indices[max_error_idx]}, golden={golden_batch[max_error_idx]:.3f}, " +
      f"mpc={mpc_affinities[max_error_idx]:.3f})")

# Show all samples
print(f"\n=== Per-sample breakdown ===")
for i in range(B):
    err = errors[i]
    status = "✓" if err < 1.0 else "⚠️"
    print(f"  [{i}] row={row_indices[i]:4d} | golden={golden_batch[i]:5.2f} | " +
          f"mpc={mpc_affinities[i]:5.2f} | error={err:5.3f} {status}")

# Show distribution of errors
if B > 4:
    percentiles = [50, 75, 90, 95]
    print(f"\nError percentiles:")
    for p in percentiles:
        print(f"  {p}th: {np.percentile(errors, p):.4f} pKd")

# Summary status
if mae < 0.5:
    print(f"\n✅ PASS: MAE={mae:.4f} < 0.5 pKd (excellent MPC fidelity)")
elif mae < 1.0:
    print(f"\n⚠️  MARGINAL: MAE={mae:.4f} in [0.5, 1.0) pKd (acceptable drift)")
else:
    print(f"\n❌ FAIL: MAE={mae:.4f} >= 1.0 pKd (excessive error)")

sys.exit(0 if mae < 1.0 else 1)
