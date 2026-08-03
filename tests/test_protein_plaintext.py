# idash/mpc/tests/test_protein_plaintext.py
import os, numpy as np, pytest
from reference import protein_plaintext as pp, mpc_config

DAVIS_CSV = "/home/jiang/master/idash/project/test/davis_test.csv"

@pytest.mark.skipif(not os.path.exists(DAVIS_CSV), reason="test CSV absent")
def test_protein_embedding_shape_and_finite():
    from baseline import official_baseline_data as ob
    ds = ob.build_dataset("davis", DAVIS_CSV, limit=1)
    pvec = pp.protein_embedding("davis", ds[0])
    assert pvec.shape == (128,)
    assert np.all(np.isfinite(pvec))

def test_export_protein_emb_roundtrip(tmp_path):
    pvec = np.array([0.5, -0.25, 1.0] + [0.0] * 125, dtype=np.float64)
    pp.export_protein_emb(pvec, str(tmp_path), scale=12, bw=32)
    path = os.path.join(str(tmp_path), mpc_config.PROTEIN_EMB_FILE)
    raw = np.fromfile(path, dtype="<u4").astype(np.int64)
    signed = np.where(raw >= (1 << 31), raw - (1 << 32), raw)
    recon = signed.astype(np.float64) / (1 << 12)
    assert np.allclose(recon[:3], [0.5, -0.25, 1.0], atol=1e-3)
