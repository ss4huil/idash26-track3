"""
Fixed-point end-to-end accuracy gate (Q20.12 vs official baseline).

Verifies that the Q20.12 fixed-point forward path clears the challenge accuracy
gate against the official BumbleBee baseline JSON, before any C++/CUDA work.

Run: ~/.pyenv/versions/3.8.7/bin/python -m pytest idash/mpc/tests/test_fixed_e2e_gate.py -v
"""
import os
import json
import pytest
import numpy as np

from reference.affinity_model import AffinityModel
from reference.fixed_forward import FixedAffinity
from reference import metrics, mpc_config

# Paths (absolute)
PTH_PATH = "/home/jiang/master/idash/mpc/model/deepdtagen_model_davis.pth"
CSV_PATH = "/home/jiang/master/idash/project/test/davis_test.csv"
BASELINE_JSON = "/home/jiang/master/idash/mpc/baseline/official_baseline_davis.json"

# Test subset size
N = 200


@pytest.mark.skipif(
    not os.path.exists(PTH_PATH) or
    not os.path.exists(CSV_PATH) or
    not os.path.exists(BASELINE_JSON),
    reason="Missing required files (pth, csv, or baseline JSON)"
)
def test_fixed_e2e_accuracy_gate():
    """
    Accuracy gate: Q20.12 fixed-point vs official baseline (first 200 pairs).

    Checks:
    (a) Label flips (fixed vs our-float): quantization issue → SCALE bump if fails
    (b) Label flips (our-float vs baseline): implementation issue → BLOCKER if beyond gate
    (c) Accuracy gate (fixed vs baseline): challenge requirement
    """
    # ── Load baseline predictions and ground truth ────────────────────────────
    with open(BASELINE_JSON) as f:
        baseline = json.load(f)

    threshold = baseline["aupr_threshold"]
    base_pred = np.array(baseline["predictions"][:N], dtype=np.float64)
    gt = np.array(baseline["ground_truth"][:N], dtype=np.float64)

    # ── Load CSV pairs ─────────────────────────────────────────────────────────
    pairs = []
    with open(CSV_PATH) as f:
        f.readline()  # skip header
        for i, line in enumerate(f):
            if i >= N:
                break
            parts = line.strip().split(',')
            smile = parts[0]
            protein_seq = parts[2]
            pairs.append((smile, protein_seq))

    assert len(pairs) == N, f"Expected {N} pairs, got {len(pairs)}"

    # ── Load model and compute predictions ─────────────────────────────────────
    print(f"\nLoading model from {PTH_PATH}")
    model = AffinityModel.from_pth(PTH_PATH, tokenizer_path=None, device="cpu")

    # Float reference (our implementation)
    print(f"Computing float predictions (N={N})...")
    float_pred = model.predict_batch(pairs, nmax=mpc_config.NMAX)

    # Fixed-point Q20.12
    print(f"Computing fixed-point predictions (bw={mpc_config.BW}, scale={mpc_config.SCALE})...")
    fixed_model = FixedAffinity(model, scale=mpc_config.SCALE, bw=mpc_config.BW)
    fx_pred = fixed_model.predict_batch(pairs, nmax=mpc_config.NMAX)

    # ── Compute accuracies ─────────────────────────────────────────────────────
    base_acc = metrics.sens_spec_accuracy(gt, base_pred, threshold)
    float_acc = metrics.sens_spec_accuracy(gt, float_pred, threshold)
    fx_acc = metrics.sens_spec_accuracy(gt, fx_pred, threshold)

    print(f"\nAccuracy @ threshold={threshold}:")
    print(f"  Official baseline: {base_acc:.4f}")
    print(f"  Our float:         {float_acc:.4f}")
    print(f"  Fixed Q{mpc_config.BW-mpc_config.SCALE}.{mpc_config.SCALE}: {fx_acc:.4f}")

    # ── Label flip analysis ────────────────────────────────────────────────────
    base_labels = (base_pred > threshold)
    float_labels = (float_pred > threshold)
    fx_labels = (fx_pred > threshold)

    # Quantization flips: fixed vs our-float (what SCALE bump addresses)
    quant_flips = np.sum(fx_labels != float_labels)
    # Implementation flips: our-float vs baseline (indicates model divergence)
    impl_flips = np.sum(float_labels != base_labels)
    # Total flips: fixed vs baseline (challenge requirement)
    total_flips = np.sum(fx_labels != base_labels)

    print(f"\nLabel flips (N={N}):")
    print(f"  Fixed vs our-float (quantization):  {quant_flips}")
    print(f"  Our-float vs baseline (impl diff):  {impl_flips}")
    print(f"  Fixed vs baseline (total):          {total_flips}")

    # ── Check implementation divergence (BLOCKER if beyond gate) ───────────────
    float_vs_base_qualified = metrics.is_qualified(base_acc, float_acc)
    if not float_vs_base_qualified:
        pytest.fail(
            f"BLOCKER: Our float model diverges from official baseline beyond the gate. "
            f"Base acc={base_acc:.4f}, float acc={float_acc:.4f}. "
            f"This is an implementation issue from earlier tasks, not a quantization issue. "
            f"Do NOT bump SCALE for this."
        )

    # ── Check quantization (SCALE contingency if flips occur) ──────────────────
    if quant_flips > 0:
        pytest.fail(
            f"Quantization flips detected: {quant_flips}/{N} labels differ between "
            f"fixed-point Q{mpc_config.BW-mpc_config.SCALE}.{mpc_config.SCALE} and our float model. "
            f"Contingency: bump mpc_config.SCALE from {mpc_config.SCALE} to {mpc_config.SCALE+1}, "
            f"re-run all Phase A tests, and document in spec §12."
        )

    # ── Check accuracy gate (challenge requirement) ────────────────────────────
    fx_vs_base_qualified = metrics.is_qualified(base_acc, fx_acc)
    assert fx_vs_base_qualified, (
        f"Fixed-point fails accuracy gate vs official baseline: "
        f"base_acc={base_acc:.4f}, fx_acc={fx_acc:.4f}, "
        f"margin={metrics.QUALIFY_MARGIN:.2f}"
    )

    print(f"\n✓ Fixed-point Q{mpc_config.BW-mpc_config.SCALE}.{mpc_config.SCALE} clears accuracy gate")
    print(f"  Base: {base_acc:.4f}, Fixed: {fx_acc:.4f} (margin: {metrics.QUALIFY_MARGIN:.2f})")
