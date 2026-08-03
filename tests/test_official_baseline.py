"""
TDD tests for the OFFICIAL plaintext baseline.

The baseline runs the *original* DeepDTAGen model (idash/project/DeepDTAGen)
with the released pretrained weights on the challenge test CSVs, producing the
per-sample affinities + metrics that the ciphertext MPC pipeline is compared
against. It must NOT depend on our reproduced plaintext model and must NOT need
the training data — only the frozen tokenizer, the released weights, and the
test CSV.

Run:
  PYBIN=/home/jiang/.pyenv/versions/3.8.7/bin/python
  $PYBIN -m pytest idash/mpc/tests/test_official_baseline.py -q
"""
import os
import math
import pytest

from baseline import official_baseline_data as ob

DAVIS_CSV = "/home/jiang/master/idash/project/test/davis_test.csv"
KIBA_CSV = "/home/jiang/master/idash/project/test/kiba_test.csv"

# Vocab sizes frozen into the released checkpoints.
EXPECTED_VOCAB = {"davis": 58, "kiba": 69}


def test_tokenizer_matches_checkpoint_vocab():
    """The frozen tokenizer must match the released checkpoint's vocab size,
    otherwise load_state_dict would corrupt the embeddings."""
    for ds, vocab in EXPECTED_VOCAB.items():
        tok = ob.load_tokenizer(ds)
        assert len(tok) == vocab, f"{ds}: tokenizer {len(tok)} != ckpt {vocab}"


def test_build_dataset_from_test_csv_only():
    """Dataset is built from the test CSV alone (no train data, no processed
    .pt). Each sample carries the affinity label straight from the CSV."""
    import pandas as pd

    n = 8
    ds = ob.build_dataset("davis", DAVIS_CSV, limit=n)
    assert len(ds) == n

    df = pd.read_csv(DAVIS_CSV).head(n)
    for i in range(n):
        sample = ds[i]
        # graph node features present
        assert sample.x.shape[0] > 0
        assert sample.x.shape[1] == 94
        # protein sequence encoded to fixed length 1000
        assert sample.target.shape[-1] == 1000
        # label matches the CSV row
        assert math.isclose(
            float(sample.y.item()), float(df["affinity"].iloc[i]), rel_tol=1e-5
        )


def test_predict_shape_and_finite():
    """Running the released model over a small slice yields one finite affinity
    per sample."""
    n = 8
    preds, trues = ob.predict("davis", DAVIS_CSV, limit=n, batch_size=4)
    assert preds.shape == (n,)
    assert trues.shape == (n,)
    assert all(math.isfinite(float(p)) for p in preds)


def test_chunked_predict_matches_single_shot():
    """Streaming the CSV in chunks must give identical affinities to loading
    it all at once. This is what lets kiba (19653 rows) run in <8GB RAM.

    Affinity = fc(PMVO, Protein_vector); it never touches target_seq padding,
    so per-chunk padding cannot change the result."""
    import numpy as np

    n = 20
    preds_single, trues_single = ob.predict("davis", DAVIS_CSV, limit=n,
                                            batch_size=8, chunk_size=None)
    preds_chunked, trues_chunked = ob.predict("davis", DAVIS_CSV, limit=n,
                                              batch_size=8, chunk_size=7)
    assert preds_chunked.shape == preds_single.shape == (n,)
    assert np.allclose(preds_chunked, preds_single, atol=1e-4)
    assert np.allclose(trues_chunked, trues_single, atol=1e-6)


def test_bounded_cindex_matches_official():
    """The official get_cindex builds N x N matrices (3.1 GB at kiba's N=19653
    -> OOM). Our tiled c-index must return the identical value in bounded
    memory."""
    import numpy as np
    import sys
    sys.path.insert(0, "/home/jiang/master/idash/project/DeepDTAGen")
    import utils as ddg_utils

    rng = np.random.RandomState(0)
    for n in (50, 200, 777):
        y = rng.uniform(5, 12, size=n).astype(np.float64)
        p = (y + rng.normal(0, 0.5, size=n)).astype(np.float64)
        official = ddg_utils.get_cindex(y, p)
        bounded = ob.cindex_bounded(y, p, tile=64)
        # official casts to float32; agreement is float32-level, not exact
        assert abs(official - bounded) < 1e-6, f"n={n}: {official} vs {bounded}"


def test_evaluate_returns_expected_metric_keys():
    """evaluate() returns the full challenge metric set on a small slice."""
    res = ob.evaluate("davis", DAVIS_CSV, limit=16, batch_size=8)
    for key in ["dataset", "n", "mse", "rmse", "pearson", "spearman",
                "cindex", "rm2", "aupr", "predictions", "ground_truth"]:
        assert key in res, f"missing metric key: {key}"
    assert res["dataset"] == "davis"
    assert res["n"] == 16
    assert len(res["predictions"]) == 16
    assert math.isfinite(res["mse"])
