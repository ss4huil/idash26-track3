"""
Challenge accuracy gate for iDASH Track-3 (see idash/describe.md).

Grading metric = average of sensitivity and specificity of the binarised DTI
prediction. Continuous affinity is binarised at a dataset-specific threshold,
matching the official flax reference `summarize_metrics`:

    positive  ⟺  affinity > threshold          (strictly greater)

A submission qualifies if its accuracy is at most 2 percentage points below the
plaintext original.
"""
import numpy as np

# dataset → binarisation threshold (flax_secure_deepdtagen.summarize_metrics)
THRESHOLDS = {
    "kiba": 12.1,
    "davis": 7.0,
    "bindingdb": 7.0,
}

# maximum allowed accuracy drop, in absolute points (2%)
QUALIFY_MARGIN = 0.02


def threshold_for(dataset: str) -> float:
    """Binarisation threshold for a named dataset (case-insensitive)."""
    return THRESHOLDS[dataset.strip().lower()]


def binarize(affinity, threshold: float) -> np.ndarray:
    """positive (1) iff affinity strictly greater than threshold."""
    return (np.asarray(affinity, dtype=np.float64) > threshold).astype(np.int64)


def _confusion(y_true, y_pred, threshold):
    yt = binarize(y_true, threshold)
    yp = binarize(y_pred, threshold)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    return tp, fn, tn, fp


def sensitivity(y_true, y_pred, threshold: float) -> float:
    """TP / (TP + FN) — recall of the positive (high-affinity) class."""
    tp, fn, _, _ = _confusion(y_true, y_pred, threshold)
    denom = tp + fn
    return tp / denom if denom else 0.0


def specificity(y_true, y_pred, threshold: float) -> float:
    """TN / (TN + FP) — recall of the negative (low-affinity) class."""
    _, _, tn, fp = _confusion(y_true, y_pred, threshold)
    denom = tn + fp
    return tn / denom if denom else 0.0


def sens_spec_accuracy(y_true, y_pred, threshold: float) -> float:
    """Challenge accuracy = mean(sensitivity, specificity)."""
    return (sensitivity(y_true, y_pred, threshold)
            + specificity(y_true, y_pred, threshold)) / 2.0


def is_qualified(original_acc: float, mpc_acc: float,
                 margin: float = QUALIFY_MARGIN) -> bool:
    """Qualified iff MPC accuracy is at most `margin` below the original."""
    return mpc_acc >= original_acc - margin
