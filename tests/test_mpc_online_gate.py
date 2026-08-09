"""
test_mpc_online_gate.py — single-machine 2PC online accuracy gate (Task 8).

Runs the compiled deepdtagen_inference binary with real trained weights and
secret-shared drug inputs, then asserts the revealed affinity matches the
plaintext baseline within tolerance.

Skip conditions:
  - binary absent (GPU-MPC not built)
  - davis test CSV or model weights absent
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import pytest

# ── paths ──────────────────────────────────────────────────────────────────
# tests/ → mpc/ (one level up)
MPC_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
GPU_DIR  = os.path.join(MPC_DIR, "gpu_mpc")
BINARY   = os.path.join(GPU_DIR, "deepdtagen_inference")
RUN_SH   = os.path.join(GPU_DIR, "run_local_2pc.sh")

from test_data_paths import DAVIS_TEST_CSV
DAVIS_CSV = DAVIS_TEST_CSV
BASELINE_JSON = os.path.join(MPC_DIR, "baseline", "official_baseline_davis.json")

# ── fixtures / marks ────────────────────────────────────────────────────────
_requires_binary = pytest.mark.skipif(
    not os.path.exists(BINARY),
    reason="deepdtagen_inference binary absent — rebuild with make")

_requires_data = pytest.mark.skipif(
    not (os.path.exists(DAVIS_CSV) and os.path.exists(BASELINE_JSON)),
    reason="davis test CSV or baseline JSON absent")


def _load_baseline():
    with open(BASELINE_JSON) as f:
        return json.load(f)


def _prepare_and_run(row_idx: int, tmp_path: str, timeout: int = 600):
    """Prepare sample offline, run 2PC, return revealed affinity float."""
    # import here so the test module can be collected even without these deps
    sys.path.insert(0, MPC_DIR)
    from reference import offline_prepare as op

    m = op.prepare_sample("davis", DAVIS_CSV, row_idx=row_idx,
                          out_dir=tmp_path, scale=12, bw=32)
    sample_dir   = m["sample_dir"]
    weights_path = m["weights_path"]

    key_dir = os.path.join(tmp_path, f"keys_{row_idx}")
    os.makedirs(key_dir, exist_ok=True)

    result = subprocess.run(
        ["bash", RUN_SH, sample_dir, key_dir, weights_path],
        cwd=GPU_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    # Dump both streams to pytest stdout for diagnosis
    if result.stdout:
        print("[run_local_2pc stdout]\n" + result.stdout)
    if result.stderr:
        print("[run_local_2pc stderr]\n" + result.stderr)

    if result.returncode != 0:
        pytest.fail(f"run_local_2pc.sh exited {result.returncode}\n"
                    f"stderr: {result.stderr[-2000:]}")

    # Parse AFFINITY= from combined output (script echoes it on stdout)
    combined = result.stdout + result.stderr
    m_aff = re.search(r"AFFINITY=([+-]?\d+\.\d+)", combined)
    if not m_aff:
        pytest.fail("AFFINITY= line not found in output:\n" + combined[-3000:])

    return float(m_aff.group(1))


# ── single-sample gate ───────────────────────────────────────────────────────
@_requires_binary
@_requires_data
def test_single_sample_reveal_matches_baseline(tmp_path):
    """Revealed affinity for davis row 0 must be within 0.1 of the baseline."""
    baseline = _load_baseline()
    baseline_pred = baseline["predictions"][0]

    reveal = _prepare_and_run(row_idx=0, tmp_path=str(tmp_path))

    abs_diff = abs(reveal - baseline_pred)
    print(f"[gate] reveal={reveal:.6f}  baseline={baseline_pred:.6f}  diff={abs_diff:.6f}")
    assert abs_diff < 0.1, (
        f"Affinity mismatch: reveal={reveal:.6f}, baseline={baseline_pred:.6f}, "
        f"abs_diff={abs_diff:.6f} >= 0.1"
    )


# ── subset gate (N samples) ──────────────────────────────────────────────────
N_SUBSET = 5   # reduced from 20 if per-sample keygen is slow; note explicitly

@_requires_binary
@_requires_data
def test_subset_gate_no_label_flips(tmp_path):
    """
    First N_SUBSET davis rows: zero label flips vs baseline at threshold 7.0,
    and mean |reveal - pred| < 0.05.

    N_SUBSET = 5 (kept small due to per-sample keygen cost; each sample takes
    ~60-120s on the local GPU, so N=20 would exceed practical CI limits).
    """
    baseline  = _load_baseline()
    threshold = baseline["aupr_threshold"]   # 7.0
    preds     = baseline["predictions"]

    abs_diffs = []
    label_flips = 0

    for row_idx in range(N_SUBSET):
        reveal = _prepare_and_run(row_idx=row_idx, tmp_path=str(tmp_path),
                                  timeout=600)
        base_p = preds[row_idx]
        diff   = abs(reveal - base_p)
        abs_diffs.append(diff)

        # label flip: baseline and reveal disagree on which side of threshold
        base_label   = int(base_p  >= threshold)
        reveal_label = int(reveal  >= threshold)
        if base_label != reveal_label:
            label_flips += 1
        print(f"  row={row_idx}  reveal={reveal:.6f}  base={base_p:.6f}"
              f"  diff={diff:.6f}  base_lbl={base_label}  reveal_lbl={reveal_label}")

    mean_diff = sum(abs_diffs) / len(abs_diffs)
    print(f"[subset gate] N={N_SUBSET} label_flips={label_flips}"
          f" mean_abs_diff={mean_diff:.6f}")

    assert label_flips == 0, (
        f"Label flips detected: {label_flips}/{N_SUBSET} samples disagree "
        f"with baseline at threshold {threshold}"
    )
    assert mean_diff < 0.05, (
        f"Mean abs diff {mean_diff:.6f} >= 0.05 over {N_SUBSET} samples"
    )
