"""
CSV-driven reference runner — bridges Python reference modules to the
challenge evaluation format.

`run_from_csv(csv_path, float_model, fixed_model)` reads the challenge CSV
(compound_iso_smiles, target_smiles, target_sequence, affinity), runs both the
float and fixed-point affinity models over every row, computes the challenge
accuracy gate (average sensitivity + specificity), and returns a result dict.

The dataset name and binarisation threshold are inferred from the CSV filename
(contains "kiba" → 12.1, otherwise → 7.0). An explicit `dataset` kwarg
overrides the inference.

Intended use:
  • As the gold-reference runner for the plaintext accuracy baseline.
  • As the comparison target that the MPC C++ output is diffed against.
"""
import csv
import os
import time
from typing import Optional

import numpy as np

from reference.affinity_model import AffinityModel
from reference.fixed_forward  import FixedAffinity
from reference.metrics        import threshold_for, sens_spec_accuracy, is_qualified, THRESHOLDS
from reference.dense_graph    import smile_to_dense_graph

NMAX = 138


def _detect_dataset(path: str) -> str:
    name = os.path.basename(path).lower()
    for ds in THRESHOLDS:
        if ds in name:
            return ds
    return "davis"                   # default fallback


def run_from_csv(csv_path: str,
                 float_model: AffinityModel,
                 fixed_model: FixedAffinity,
                 dataset: Optional[str] = None,
                 nmax: int = NMAX) -> dict:
    """Run both models over a challenge CSV and return a result dict.

    Returns:
        dataset       (str)    inferred or explicit dataset name
        threshold     (float)  binarisation threshold used
        n_samples     (int)    number of evaluated rows
        float_preds   (ndarray float64)  per-row float predictions
        fixed_preds   (ndarray float64)  per-row fixed-point predictions
        true_labels   (ndarray float64)  per-row ground-truth affinity
        float_acc     (float)  float-model sens+spec accuracy
        fixed_acc     (float)  fixed-model sens+spec accuracy
        qualified     (bool)   True iff fixed_acc within 2pt of float_acc
        float_time_s  (float)  elapsed seconds for float predictions
        fixed_time_s  (float)  elapsed seconds for fixed-point predictions
    """
    ds   = dataset or _detect_dataset(csv_path)
    thr  = threshold_for(ds)

    smiles_col = "compound_iso_smiles"
    seq_col    = "target_sequence"
    aff_col    = "affinity"

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row[smiles_col], row[seq_col], float(row[aff_col])))

    n = len(rows)
    true_labels  = np.array([y for _, _, y in rows], dtype=np.float64)
    float_preds  = np.empty(n, dtype=np.float64)
    fixed_preds  = np.empty(n, dtype=np.float64)

    # float pass — cache (X, A_hat, mask) per unique SMILES
    graph_cache: dict = {}
    t0 = time.perf_counter()
    for i, (smile, protein, _) in enumerate(rows):
        if smile not in graph_cache:
            graph_cache[smile] = smile_to_dense_graph(smile, nmax)
        X, A_hat, mask = graph_cache[smile]
        float_preds[i] = float_model.predict(X, A_hat, mask, protein)
    float_time = time.perf_counter() - t0

    # fixed-point pass — reuse same graph cache
    t0 = time.perf_counter()
    for i, (smile, protein, _) in enumerate(rows):
        X, A_hat, mask = graph_cache[smile]
        fixed_preds[i] = fixed_model.predict(X, A_hat, mask, protein)
    fixed_time = time.perf_counter() - t0

    float_acc = sens_spec_accuracy(true_labels, float_preds, thr)
    fixed_acc = sens_spec_accuracy(true_labels, fixed_preds, thr)

    return {
        "dataset":      ds,
        "threshold":    thr,
        "n_samples":    n,
        "float_preds":  float_preds,
        "fixed_preds":  fixed_preds,
        "true_labels":  true_labels,
        "float_acc":    float_acc,
        "fixed_acc":    fixed_acc,
        "qualified":    is_qualified(float_acc, fixed_acc),
        "float_time_s": float_time,
        "fixed_time_s": fixed_time,
    }


def print_report(result: dict) -> None:
    """Print a human-readable experiment summary to stdout."""
    r = result
    print(f"\n=== {r['dataset'].upper()} (threshold={r['threshold']}) ===")
    print(f"  n_samples    : {r['n_samples']}")
    print(f"  float  acc   : {r['float_acc']:.4f}  ({r['float_time_s']:.1f}s)")
    print(f"  fixed  acc   : {r['fixed_acc']:.4f}  ({r['fixed_time_s']:.1f}s)")
    print(f"  drop         : {(r['float_acc'] - r['fixed_acc']) * 100:.2f}pp")
    verdict = "✅ QUALIFIED" if r["qualified"] else "❌ FAILED (>2pp drop)"
    print(f"  gate         : {verdict}\n")
