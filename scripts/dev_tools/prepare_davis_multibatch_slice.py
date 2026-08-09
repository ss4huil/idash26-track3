#!/usr/bin/env python3
"""
Prepare one slice of a multi-batch davis validation: stratified sampling with offset.

Usage:
  python prepare_davis_multibatch_slice.py <batch_idx> <batch_size> <batch_name>

Samples are stratified across 5 affinity bins (low/med-low/med/med-high/high pKd),
with batch_idx controlling the random seed for reproducible, non-overlapping selection.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from prepare_batch_samples import prepare_batch_samples

if len(sys.argv) < 4:
    print(__doc__)
    sys.exit(1)

batch_idx = int(sys.argv[1])
batch_size = int(sys.argv[2])
batch_name = sys.argv[3]

# Load davis test CSV (idash/mpc/data/, two levels up from scripts/dev_tools/)
_MPC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_path = os.path.join(_MPC_ROOT, "data", "davis_test.csv")
df = pd.read_csv(csv_path)

# Stratified sampling: 5 bins covering pKd range, batch_size/5 samples per bin
pKd_min, pKd_max = df['affinity'].min(), df['affinity'].max()
bins = np.linspace(pKd_min, pKd_max, 6)  # 5 bins
samples_per_bin = max(1, batch_size // 5)

row_indices = []
np.random.seed(42 + batch_idx)  # offset seed by batch_idx for non-overlap

for i in range(len(bins) - 1):
    bin_mask = (df['affinity'] >= bins[i]) & (df['affinity'] < bins[i+1])
    bin_indices = df[bin_mask].index.tolist()

    if len(bin_indices) == 0:
        continue

    # Sample without replacement
    n_sample = min(samples_per_bin, len(bin_indices))
    sampled = np.random.choice(bin_indices, size=n_sample, replace=False)
    row_indices.extend(sampled.tolist())

# Pad to exact batch_size if needed
if len(row_indices) < batch_size:
    remaining = batch_size - len(row_indices)
    all_indices = set(range(len(df)))
    available = list(all_indices - set(row_indices))
    row_indices.extend(np.random.choice(available, size=remaining, replace=False))

row_indices = sorted(int(x) for x in row_indices[:batch_size])

print(f"[prepare] Batch {batch_idx}: selected {len(row_indices)} samples")
print(f"[prepare] Rows: {row_indices}")
selected_affinities = df.iloc[row_indices]['affinity'].values
print(f"[prepare] pKd range: [{selected_affinities.min():.2f}, {selected_affinities.max():.2f}]")

# Prepare batch
out_dir = "/home/jiang/master/idash/mpc/gpu_mpc"
manifest = prepare_batch_samples(
    dataset="davis",
    csv_path=csv_path,
    row_indices=row_indices,
    out_dir=out_dir,
    batch_name=batch_name,
    scale=12,
    bw=32
)

print(f"[prepare] ✓ {batch_name}/ ready for MPC")
