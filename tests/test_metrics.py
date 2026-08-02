"""
TDD – RED phase: tests for the challenge accuracy gate (spec §9 / describe.md).

The iDASH Track-3 accuracy metric is the AVERAGE of sensitivity and specificity
of the binarised DTI prediction. Continuous affinity is binarised at a
dataset-specific threshold (KIBA = 12.1, Davis/BindingDB = 7.0), matching the
official flax reference's `summarize_metrics`:

    positive  ⟺  affinity > threshold

A submission is "qualified" if its accuracy is at most 2 percentage points below
the plaintext original on the test data.

    sensitivity = TP / (TP + FN)          (recall of the positive class)
    specificity = TN / (TN + FP)          (recall of the negative class)
    accuracy    = (sensitivity + specificity) / 2

Run:  python3 -m pytest idash/mpc/tests/test_metrics.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from reference.metrics import (   # not yet implemented
    binarize, sensitivity, specificity, sens_spec_accuracy,
    threshold_for, is_qualified, THRESHOLDS,
)


class TestThresholds:
    def test_kiba_threshold(self):
        assert threshold_for("kiba") == 12.1

    def test_davis_threshold(self):
        assert threshold_for("davis") == 7.0

    def test_bindingdb_threshold(self):
        assert threshold_for("bindingdb") == 7.0

    def test_case_insensitive(self):
        assert threshold_for("KIBA") == 12.1


class TestBinarize:
    def test_strictly_greater(self):
        """positive ⟺ affinity > threshold (not >=)."""
        aff = np.array([6.9, 7.0, 7.1])
        np.testing.assert_array_equal(binarize(aff, 7.0), [0, 0, 1])

    def test_returns_int(self):
        b = binarize(np.array([1.0, 20.0]), 12.1)
        assert b.dtype == np.int64 or b.dtype == np.int32


class TestSensitivitySpecificity:
    def test_perfect_prediction(self):
        y = np.array([15.0, 5.0, 13.0, 2.0])      # KIBA: >12.1 → pos
        p = np.array([14.0, 4.0, 20.0, 1.0])
        assert sensitivity(y, p, 12.1) == 1.0
        assert specificity(y, p, 12.1) == 1.0
        assert sens_spec_accuracy(y, p, 12.1) == 1.0

    def test_all_wrong(self):
        y = np.array([15.0, 5.0])                  # pos, neg
        p = np.array([1.0, 20.0])                  # predicted neg, pos
        assert sensitivity(y, p, 12.1) == 0.0
        assert specificity(y, p, 12.1) == 0.0

    def test_sensitivity_definition(self):
        """TP/(TP+FN): of the true positives, how many predicted positive."""
        # true positives: idx 0,1,2 (>7). preds: 0 correct, 1 correct, 2 miss
        y = np.array([8.0, 9.0, 10.0, 1.0])
        p = np.array([8.0, 9.0, 1.0, 1.0])
        # TP=2 (idx0,1), FN=1 (idx2) → 2/3
        assert abs(sensitivity(y, p, 7.0) - 2/3) < 1e-9

    def test_specificity_definition(self):
        """TN/(TN+FP): of the true negatives, how many predicted negative."""
        y = np.array([1.0, 2.0, 3.0, 8.0])         # 3 negatives, 1 positive
        p = np.array([1.0, 9.0, 3.0, 8.0])         # idx1 false positive
        # TN=2 (idx0,2), FP=1 (idx1) → 2/3
        assert abs(specificity(y, p, 7.0) - 2/3) < 1e-9

    def test_accuracy_is_average(self):
        y = np.array([8.0, 9.0, 1.0, 2.0])
        p = np.array([8.0, 1.0, 1.0, 9.0])
        sens = sensitivity(y, p, 7.0)
        spec = specificity(y, p, 7.0)
        assert abs(sens_spec_accuracy(y, p, 7.0) - (sens + spec) / 2) < 1e-12


class TestQualificationGate:
    def test_within_2pct_qualifies(self):
        """<= 2 percentage points below original → qualified."""
        assert is_qualified(original_acc=0.90, mpc_acc=0.885) is True   # 1.5pt drop
        assert is_qualified(original_acc=0.90, mpc_acc=0.88) is True    # exactly 2pt

    def test_more_than_2pct_fails(self):
        assert is_qualified(original_acc=0.90, mpc_acc=0.87) is False   # 3pt drop

    def test_improvement_qualifies(self):
        assert is_qualified(original_acc=0.90, mpc_acc=0.92) is True
