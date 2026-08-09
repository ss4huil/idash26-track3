"""
TDD – RED phase: tests for the full plaintext affinity reference model.

AffinityModel(weights_dir) encapsulates:
  1. Drug path (dense, will be MPC):
       GCN×3 (94→188→282→376) + ReLU + masked max-pool → Drug_FC (→1024→ReLU→128)
  2. Protein path (plaintext, public):
       GatedCNN → Pvec (128)
  3. Fusion (dense, will be MPC):
       concat(PMVO, Pvec) → FC (256→1024→ReLU→512→ReLU→256→ReLU→1)

The reference uses the SAME weights as the original DeepDTAGen model (loaded
from a .pth checkpoint) but applies the dense fixed-Nmax GCN instead of PyG's
sparse GCNConv, so we can verify numerical equivalence.

Tests are split into two groups:
  - Structural tests (no weights needed): shape, dtypes, determinism.
  - Alignment tests (require .pth weights, skipped otherwise): affinity values
    from dense reference must match original sparse model within atol=0.01.

Run:  python3 -m pytest idash/mpc/tests/test_affinity_model.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from reference.dense_graph   import smile_to_dense_graph
from reference.affinity_model import AffinityModel, seq_cat   # not yet implemented

# ── paths ─────────────────────────────────────────────────────────────────────
_PROJ  = os.path.join(os.path.dirname(__file__),
                      "../../../project/DeepDTAGen")
_PTH   = {ds: os.path.join(_PROJ, f"models/deepdtagen_model_{ds}.pth")
          for ds in ("davis", "kiba")}
_TOK   = {ds: os.path.join(_PROJ, f"data/{ds}_tokenizer.pkl")
          for ds in ("davis", "kiba")}

_WEIGHTS_AVAILABLE = any(os.path.exists(p) for p in _PTH.values())
_requires_weights  = pytest.mark.skipif(not _WEIGHTS_AVAILABLE,
                                         reason="pretrained .pth weights not downloaded yet")

# ── small test data ────────────────────────────────────────────────────────────
_ASPIRIN_SMILES  = "CC(=O)Oc1ccccc1C(=O)O"
_EXAMPLE_PROTEIN = "MKTAYIAKQRQISFVKSHFSRQ"    # 22 aa stub (will be 0-padded to 1000)


class TestAffinityModelStructural:
    """Shape and type checks — no pretrained weights needed."""

    def _random_model(self, feat_dim=94):
        """AffinityModel initialised with random weights (no .pth needed)."""
        return AffinityModel.from_random(feat_dim=feat_dim)

    def test_drug_path_pmvo_shape(self):
        model = self._random_model()
        X, A_hat, mask = smile_to_dense_graph(_ASPIRIN_SMILES, nmax=128)
        pmvo = model.drug_path(X, A_hat, mask)   # should return (128,)
        assert pmvo.shape == (128,), f"PMVO shape wrong: {pmvo.shape}"

    def test_protein_path_pvec_shape(self):
        model = self._random_model()
        pvec = model.protein_path(_EXAMPLE_PROTEIN)  # should return (128,)
        assert pvec.shape == (128,), f"Pvec shape wrong: {pvec.shape}"

    def test_predict_scalar_output(self):
        model = self._random_model()
        X, A_hat, mask = smile_to_dense_graph(_ASPIRIN_SMILES, nmax=128)
        aff = model.predict(X, A_hat, mask, _EXAMPLE_PROTEIN)  # scalar or (1,)
        assert np.isscalar(aff) or aff.shape in ((), (1,)), \
            f"predict should return a scalar, got shape {np.array(aff).shape}"

    def test_batch_predict(self):
        """predict_batch on N (smile, protein) pairs → (N,) array."""
        model = self._random_model()
        pairs = [(_ASPIRIN_SMILES, _EXAMPLE_PROTEIN)] * 4
        affs = model.predict_batch(pairs, nmax=128)
        assert affs.shape == (4,), f"batch output shape wrong: {affs.shape}"

    def test_drug_path_deterministic(self):
        model = self._random_model()
        X, A_hat, mask = smile_to_dense_graph(_ASPIRIN_SMILES, nmax=128)
        pmvo1 = model.drug_path(X, A_hat, mask)
        pmvo2 = model.drug_path(X, A_hat, mask)
        np.testing.assert_allclose(pmvo1, pmvo2)


class TestAffinityModelAlignment:
    """Numerical alignment with the original sparse DeepDTAGen model."""

    @_requires_weights
    @pytest.mark.parametrize("dataset", ["davis", "kiba"])
    def test_affinity_matches_original_model(self, dataset):
        """Dense reference must match original PyG model within atol=0.01."""
        import pickle
        from torch_geometric.data import Data
        sys.path.insert(0, _PROJ)
        from model import DeepDTAGen

        # load tokenizer
        with open(_TOK[dataset], "rb") as f:
            tokenizer = pickle.load(f)

        # load original sparse model
        device = "cpu"
        orig_model = DeepDTAGen(tokenizer)
        orig_model.load_state_dict(torch.load(_PTH[dataset], map_location=device))
        orig_model.eval()

        # load dense reference model from same .pth
        ref_model = AffinityModel.from_pth(_PTH[dataset], _TOK[dataset])

        # read a handful of rows from the test CSV
        import pandas as pd
        from test_data_paths import DAVIS_TEST_CSV, KIBA_TRAIN_CSV
        _csv_map = {"davis": DAVIS_TEST_CSV, "kiba": KIBA_TRAIN_CSV}

        test_csv = os.path.join(_PROJ, f"data/{dataset}_test.csv")
        if not os.path.exists(test_csv):
            # fallback: idash/mpc/data/
            test_csv = _csv_map.get(dataset, test_csv)
        df = pd.read_csv(test_csv).head(10)

        from utils import TestbedDataset  # noqa (only available in DeepDTAGen dir)
        for _, row in df.iterrows():
            smile  = row["compound_iso_smiles"]
            target = row["target_sequence"]

            # --- original model affinity (sparse) ---
            from reference.dense_graph import smile_to_dense_graph as s2d, \
                                              atom_features as af
            import networkx as nx
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smile)
            # build PyG Data object
            feats = np.array([af(a) for a in mol.GetAtoms()], dtype=np.float32)
            feats = feats / feats.sum(axis=1, keepdims=True)
            edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
            edges += [[b, a] for a, b in edges]
            ei = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros(2, 0, dtype=torch.long)
            target_enc = seq_cat(target)
            data_obj = Data(
                x=torch.tensor(feats, dtype=torch.float),
                edge_index=ei,
                c_size=torch.tensor([mol.GetNumAtoms()]),
                target=torch.tensor(target_enc, dtype=torch.long).unsqueeze(0),
                target_seq=torch.tensor(target_enc, dtype=torch.long).unsqueeze(0),
                y=torch.tensor([0.0]),
                batch=torch.zeros(mol.GetNumAtoms(), dtype=torch.long),
            )
            with torch.no_grad():
                orig_aff = orig_model(data_obj)[0].item()

            # --- dense reference ---
            X, A_hat, mask = s2d(smile, nmax=128)
            ref_aff = float(ref_model.predict(X, A_hat, mask, target))

            assert abs(ref_aff - orig_aff) < 0.01, \
                f"Affinity mismatch for {smile[:20]!r}: dense={ref_aff:.4f} orig={orig_aff:.4f}"
