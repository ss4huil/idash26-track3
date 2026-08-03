"""Offline prepare driver: one CSV row -> every artifact the online binary needs.

`prepare_sample` must write the 6 secret-share files + the public protein
embedding into `<out_dir>/sample_<row_idx>/`, and the public fixed-point weight
blob once per run into `<out_dir>/weights.bin`, then return a manifest.
"""
import os

import pytest

from reference import offline_prepare as op, mpc_config
from baseline import official_baseline_data as ob

DAVIS_CSV = "/home/jiang/master/idash/project/test/davis_test.csv"
DAVIS_PTH = os.path.join(os.path.dirname(__file__), "..", "model",
                         "deepdtagen_model_davis.pth")

_requires_data = pytest.mark.skipif(
    not (os.path.exists(DAVIS_CSV) and os.path.exists(DAVIS_PTH)),
    reason="davis test CSV or released weights absent")


@_requires_data
def test_prepare_sample_writes_all_artifacts(tmp_path):
    m = op.prepare_sample("davis", DAVIS_CSV, row_idx=0, out_dir=str(tmp_path),
                          scale=12, bw=32)
    sd = m["sample_dir"]
    for tensor in ("x", "adj", "mask"):
        for party in (0, 1):
            assert os.path.exists(os.path.join(sd, mpc_config.share_filename(tensor, party)))
    assert os.path.exists(os.path.join(sd, mpc_config.PROTEIN_EMB_FILE))
    assert os.path.exists(m["weights_path"])
    assert m["bw"] == 32 and m["scale"] == 12 and isinstance(m["y"], float)


@_requires_data
def test_manifest_keys_and_row_fields(tmp_path):
    """Manifest carries the interface keys, sourced from the real CSV row."""
    m = op.prepare_sample("davis", DAVIS_CSV, row_idx=1, out_dir=str(tmp_path))
    assert set(m) >= {"sample_dir", "weights_path", "smile", "protein_seq",
                      "y", "bw", "scale"}
    # helper added to official_baseline_data is the single source of row truth
    row = ob.dataset_row(DAVIS_CSV, 1)
    assert m["smile"] == row["smile"] and m["protein_seq"] == row["protein_seq"]
    assert m["y"] == pytest.approx(row["y"])
    assert m["sample_dir"].endswith("sample_1")
    assert m["bw"] == mpc_config.BW and m["scale"] == mpc_config.SCALE


@_requires_data
def test_weights_written_once_and_shares_sized_for_bw(tmp_path):
    """weights.bin is reused across samples with the SAME scale; share files match ring width."""
    m0 = op.prepare_sample("davis", DAVIS_CSV, row_idx=0, out_dir=str(tmp_path),
                           scale=12, bw=32)
    stamp = os.stat(m0["weights_path"]).st_mtime_ns
    m1 = op.prepare_sample("davis", DAVIS_CSV, row_idx=1, out_dir=str(tmp_path),
                           scale=12, bw=32)
    assert m1["weights_path"] == m0["weights_path"]
    assert os.stat(m1["weights_path"]).st_mtime_ns == stamp, "weights re-dumped"
    assert m0["sample_dir"] != m1["sample_dir"]

    # bw=32 -> 4 bytes per ring element
    x0 = os.path.join(m1["sample_dir"], mpc_config.share_filename("x", 0))
    assert os.path.getsize(x0) == mpc_config.NMAX * mpc_config.FEAT_DIM * 4
    mask0 = os.path.join(m1["sample_dir"], mpc_config.share_filename("mask", 0))
    assert os.path.getsize(mask0) == mpc_config.NMAX * mpc_config.POOL_DIM * 4
    pemb = os.path.join(m1["sample_dir"], mpc_config.PROTEIN_EMB_FILE)
    assert os.path.getsize(pemb) == 128 * 4


@_requires_data
def test_weights_scale_mismatch_raises(tmp_path):
    """Calling prepare_sample twice into one out_dir with DIFFERENT scales raises ValueError."""
    op.prepare_sample("davis", DAVIS_CSV, row_idx=0, out_dir=str(tmp_path), scale=12)
    with pytest.raises(ValueError, match=r"weights\.bin scale mismatch.*scale=12.*scale=18"):
        op.prepare_sample("davis", DAVIS_CSV, row_idx=1, out_dir=str(tmp_path), scale=18)
