"""
TDD – real pretrained weights: does the MPC (fixed-point) framework run, and
how far does its accuracy fall below the plaintext model?

The GPU-MPC backend computes the drug GCN path + fusion FC over a fixed-point
ring; `FixedAffinity` reproduces that arithmetic *exactly* (one truncation per
matmul, ring reduction into the signed bw-bit range). Model weights are
plaintext, the protein GatedCNN is precomputed offline in the clear — so
FixedAffinity over the REAL weights is the faithful accuracy measurement of the
secure pipeline.

Weights: idash/mpc/model/deepdtagen_model_{davis,kiba,bindingdb}.pth
Test CSVs: idash/project/test/{davis,kiba}_test.csv

Each test names the break it catches:
  * framework can't load real weights          -> test_framework_loads_real_weights
  * fixed-point pipeline crashes / NaNs on real -> test_fixed_pipeline_runs_on_real_sample
  * bw=64 diverges from plaintext beyond 1 LSB  -> test_bw64_tracks_float_on_real_sample
  * MPC accuracy drop exceeds the iDASH 2pt gate-> test_accuracy_gate_{davis,kiba}_subset

Run:  ~/.pyenv/versions/3.8.7/bin/python -m pytest \
        idash/mpc/tests/test_real_weights_accuracy.py -v -s
"""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from reference.affinity_model import AffinityModel
from reference.fixed_forward  import FixedAffinity
from reference.dense_graph    import smile_to_dense_graph
from reference.metrics        import (threshold_for, sensitivity, specificity,
                                      sens_spec_accuracy, is_qualified)

NMAX = 138
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
_TEST_DIR  = os.path.join(os.path.dirname(__file__), "../../project/test")

_PTH = {ds: os.path.join(_MODEL_DIR, f"deepdtagen_model_{ds}.pth")
        for ds in ("davis", "kiba", "bindingdb")}
_CSV = {ds: os.path.join(_TEST_DIR, f"{ds}_test.csv")
        for ds in ("davis", "kiba")}

_HAS = {ds: os.path.exists(p) for ds, p in _PTH.items()}
_requires = lambda ds: pytest.mark.skipif(
    not (_HAS.get(ds) and os.path.exists(_CSV.get(ds, ""))),
    reason=f"{ds} weights or test CSV not available")

# subset size — keeps CI fast; full-set eval lives in the standalone script.
_SUBSET = int(os.environ.get("MPC_ACC_SUBSET", "200"))


def _load_rows(csv_path, limit=None):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["compound_iso_smiles"], r["target_sequence"],
                         float(r["affinity"])))
            if limit and len(rows) >= limit:
                break
    return rows


def _eval_subset(dataset, scale, bw, limit):
    """Run float + fixed-point over a subset; return (float_acc, fixed_acc, thr)."""
    m  = AffinityModel.from_pth(_PTH[dataset])
    fm = FixedAffinity(m, scale=scale, bw=bw)
    thr = threshold_for(dataset)
    rows = _load_rows(_CSV[dataset], limit)

    # cache graphs per unique SMILES and protein vecs per unique sequence
    gcache, pcache = {}, {}
    yt = np.array([y for _, _, y in rows], dtype=np.float64)
    yf = np.empty(len(rows)); yq = np.empty(len(rows))
    for i, (sm, seq, _) in enumerate(rows):
        if sm not in gcache:
            gcache[sm] = smile_to_dense_graph(sm, NMAX)
        if seq not in pcache:
            pcache[seq] = (m.protein_path(seq),        # float Pvec
                           fm._protein_vec_fx(seq))     # fixed Pvec
        X, A, mask = gcache[sm]
        pf, pq = pcache[seq]
        # float
        pmvo_f = m.drug_path(X, A, mask)
        hf = np.concatenate([pmvo_f, pf]); n = len(m.fusion)
        for k, (W, b) in enumerate(m.fusion):
            hf = hf @ W.T + b
            if k < n - 1: hf = np.maximum(hf, 0)
        yf[i] = float(hf[0])
        # fixed-point (MPC-faithful)
        pmvo_q = fm._drug_path_fx(X, A, mask)
        yq[i]  = float(__import__("reference.fixedpoint", fromlist=["from_fixed"])
                       .from_fixed(fm._fusion_fx(pmvo_q, pq), scale)[0])
    return (sens_spec_accuracy(yt, yf, thr),
            sens_spec_accuracy(yt, yq, thr), thr)


class TestFrameworkRunsOnRealWeights:
    @_requires("davis")
    def test_framework_loads_real_weights(self):
        m = AffinityModel.from_pth(_PTH["davis"])
        assert m.feat_dim == 94
        assert len(m.gcn) == 3 and len(m.fusion) == 4

    @_requires("davis")
    def test_fixed_pipeline_runs_on_real_sample(self):
        m  = AffinityModel.from_pth(_PTH["davis"])
        fm = FixedAffinity(m, scale=24, bw=64)
        sm, seq, _ = _load_rows(_CSV["davis"], 1)[0]
        X, A, mask = smile_to_dense_graph(sm, NMAX)
        v = fm.predict(X, A, mask, seq)
        assert np.isfinite(v), "fixed-point pipeline produced non-finite output"

    @_requires("davis")
    def test_bw64_tracks_float_on_real_sample(self):
        m  = AffinityModel.from_pth(_PTH["davis"])
        fm = FixedAffinity(m, scale=24, bw=64)
        sm, seq, _ = _load_rows(_CSV["davis"], 1)[0]
        X, A, mask = smile_to_dense_graph(sm, NMAX)
        vf = m.predict(X, A, mask, seq)
        vq = fm.predict(X, A, mask, seq)
        # Q40.24 fixed-point should track the float model to <0.01 affinity units.
        assert abs(vf - vq) < 0.01, f"bw=64 diverged: float={vf} fixed={vq}"


class TestAccuracyGate:
    @_requires("davis")
    def test_accuracy_gate_davis_subset_bw64(self):
        fa, qa, thr = _eval_subset("davis", scale=24, bw=64, limit=_SUBSET)
        print(f"\n[davis bw64] float={fa:.4f} fixed={qa:.4f} "
              f"drop={(fa-qa)*100:.2f}pp thr={thr}")
        assert is_qualified(fa, qa), f"davis bw64 drop too large: {fa}->{qa}"

    @_requires("kiba")
    def test_accuracy_gate_kiba_subset_bw64(self):
        fa, qa, thr = _eval_subset("kiba", scale=24, bw=64, limit=_SUBSET)
        print(f"\n[kiba bw64] float={fa:.4f} fixed={qa:.4f} "
              f"drop={(fa-qa)*100:.2f}pp thr={thr}")
        assert is_qualified(fa, qa), f"kiba bw64 drop too large: {fa}->{qa}"

    @_requires("davis")
    def test_accuracy_gate_davis_subset_bw32(self):
        # bw=32 ring — scale=12 to stay safely inside [-(1<<31), (1<<31)).
        # Break: 32-bit overflow or excess truncation noise flips binarised labels.
        fa, qa, thr = _eval_subset("davis", scale=12, bw=32, limit=_SUBSET)
        print(f"\n[davis bw32] float={fa:.4f} fixed={qa:.4f} "
              f"drop={(fa-qa)*100:.2f}pp thr={thr}")
        assert is_qualified(fa, qa), f"davis bw32 drop too large: {fa:.4f}->{qa:.4f}"

    @_requires("kiba")
    def test_accuracy_gate_kiba_subset_bw32(self):
        fa, qa, thr = _eval_subset("kiba", scale=12, bw=32, limit=_SUBSET)
        print(f"\n[kiba bw32] float={fa:.4f} fixed={qa:.4f} "
              f"drop={(fa-qa)*100:.2f}pp thr={thr}")
        assert is_qualified(fa, qa), f"kiba bw32 drop too large: {fa:.4f}->{qa:.4f}"
