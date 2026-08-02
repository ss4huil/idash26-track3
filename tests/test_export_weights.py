"""
TDD – RED: fixed-point weight export for the GPU-MPC backend.

`export_weights.dump_mpc_weights(model, out_path, scale=24)` serialises the
MPC-secured layers (GCN×3, Drug_FC×2, fusion×4) into the raw binary format the
GPU-MPC loader expects:

  • int64 little-endian, no header, layers concatenated in forward order
  • each layer emits W (in_features × out_features, row-major) then bias
  • weights quantised as round(w * 2**scale)

A sidecar JSON manifest records per-layer name / shape / offset so both the C++
loader and these tests can verify the byte layout. The GatedCNN is NOT exported
here — the protein path is public and runs in plaintext on P2.

Run:  python3 -m pytest idash/mpc/tests/test_export_weights.py -v
"""
import sys, os, json, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from reference.affinity_model import AffinityModel
from reference.export_weights import dump_mpc_weights, load_mpc_weights   # not yet implemented

SCALE = 24


@pytest.fixture
def model():
    return AffinityModel.from_random(seed=0)


class TestManifest:
    def test_creates_dat_and_manifest(self, model, tmp_path):
        out = str(tmp_path / "weights.dat")
        manifest = dump_mpc_weights(model, out, scale=SCALE)
        assert os.path.exists(out)
        assert os.path.exists(out + ".json")
        assert manifest["scale"] == SCALE
        assert manifest["bitwidth"] == 64

    def test_layer_order_is_forward(self, model, tmp_path):
        out = str(tmp_path / "w.dat")
        m = dump_mpc_weights(model, out, scale=SCALE)
        names = [l["name"] for l in m["layers"]]
        assert names == [
            "gcn.0", "gcn.1", "gcn.2",
            "drug_fc.0", "drug_fc.1",
            "fusion.0", "fusion.1", "fusion.2", "fusion.3",
        ]

    def test_manifest_shapes_are_in_by_out(self, model, tmp_path):
        # weight dumped transposed: (in_features, out_features)
        out = str(tmp_path / "w.dat")
        m = dump_mpc_weights(model, out, scale=SCALE)
        gcn0 = next(l for l in m["layers"] if l["name"] == "gcn.0")
        assert gcn0["W_shape"] == [94, 188]        # (in, out)
        assert gcn0["b_shape"] == [188]

    def test_total_element_count(self, model, tmp_path):
        out = str(tmp_path / "w.dat")
        m = dump_mpc_weights(model, out, scale=SCALE)
        expected = 0
        for name in ("gcn", "drug_fc", "fusion"):
            for (W, b) in getattr(model, name):
                expected += W.size + b.size
        got = os.path.getsize(out) // 8       # int64 = 8 bytes
        assert got == expected
        assert m["total_elements"] == expected


class TestBinaryLayout:
    def test_dtype_is_int64_little_endian(self, model, tmp_path):
        out = str(tmp_path / "w.dat")
        dump_mpc_weights(model, out, scale=SCALE)
        raw = np.fromfile(out, dtype="<i8")
        assert raw.dtype == np.dtype("<i8")
        # first W is gcn.0 transposed → first value = round(W[0,0]*2^scale)
        W0 = model.gcn[0][0]                       # (188, 94)
        expected0 = int(round(float(W0.T.ravel()[0]) * (1 << SCALE)))
        assert raw[0] == expected0

    def test_roundtrip_dequantise_matches(self, model, tmp_path):
        out = str(tmp_path / "w.dat")
        dump_mpc_weights(model, out, scale=SCALE)
        loaded = load_mpc_weights(out)             # dict name -> (W_float, b_float)
        for name, layers in (("gcn", model.gcn),
                             ("drug_fc", model.drug_fc),
                             ("fusion", model.fusion)):
            for i, (W, b) in enumerate(layers):
                key = f"{name}.{i}"
                W_l, b_l = loaded[key]
                # loaded W is (in, out); compare to W.T
                assert np.allclose(W_l, W.T, atol=2.0 ** -SCALE * 4)
                assert np.allclose(b_l, b,   atol=2.0 ** -SCALE * 4)


class TestScale:
    def test_higher_scale_bigger_magnitudes(self, model, tmp_path):
        a = str(tmp_path / "a.dat"); b = str(tmp_path / "b.dat")
        dump_mpc_weights(model, a, scale=12)
        dump_mpc_weights(model, b, scale=24)
        ra = np.fromfile(a, dtype="<i8").astype(np.float64)
        rb = np.fromfile(b, dtype="<i8").astype(np.float64)
        # same weights, 12 more fractional bits → ~2^12 larger integers
        assert np.abs(rb).sum() > np.abs(ra).sum() * 100
