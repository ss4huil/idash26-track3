"""
TDD – RED phase: tests for dense_gcn_layer.

The dense GCN layer computes: out = A_hat @ (X @ W^T) + b
This must match PyG's GCNConv (with add_self_loops=True, normalize=True)
for the real-atom rows, since our A_hat = D^{-1/2}(A+I)D^{-1/2} is identical
to gcn_norm applied to the sparse graph.

Run:  python3 -m pytest idash/mpc/tests/test_dense_gcn.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from reference.dense_graph import smile_to_dense_graph, FEAT_DIM
from reference.dense_gcn   import dense_gcn_layer          # not yet implemented

# ── helpers ──────────────────────────────────────────────────────────────────

def _smile_to_pyg_sparse(smile: str):
    """Build a torch_geometric Data object from a SMILES string."""
    import networkx as nx
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    from reference.dense_graph import atom_features
    feats = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    feats = feats / feats.sum(axis=1, keepdims=True)   # L1-normalise (matches create_data.py)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    if not edges:
        edges = [[0, 0]]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    x = torch.tensor(feats, dtype=torch.float)
    return Data(x=x, edge_index=edge_index), c_size


def _conv_weights(conv):
    """Extract (W, b) from a GCNConv as numpy arrays."""
    W = conv.lin.weight.detach().numpy()   # (out_ch, in_ch)
    b = conv.bias.detach().numpy()          # (out_ch,)
    return W, b


# ─────────────────────────────────────────────────────────────────────────────

class TestDenseGcnLayer:
    """dense_gcn_layer(X, A_hat, W, b) ≡ GCNConv on real-atom rows."""

    SMILES = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
    OUT_CH = 32

    def test_return_shape(self):
        from torch_geometric.nn import GCNConv
        conv = GCNConv(FEAT_DIM, self.OUT_CH, bias=True)
        W, b = _conv_weights(conv)
        X, A_hat, mask = smile_to_dense_graph("CCO", nmax=16)
        out = dense_gcn_layer(X, A_hat, W, b)
        assert out.shape == (16, self.OUT_CH), f"expected (16, {self.OUT_CH}), got {out.shape}"

    @pytest.mark.parametrize("smile", SMILES)
    def test_matches_pyg_gcnconv_real_atoms(self, smile):
        """Dense output[:c_size] matches GCNConv(sparse) for the same weights."""
        from torch_geometric.nn import GCNConv
        torch.manual_seed(0)
        conv = GCNConv(FEAT_DIM, self.OUT_CH, add_self_loops=True,
                       normalize=True, bias=True)
        W, b = _conv_weights(conv)

        X, A_hat, mask = smile_to_dense_graph(smile, nmax=128)
        c_size = int(mask.sum())

        # --- reference: sparse PyG ---
        data, _ = _smile_to_pyg_sparse(smile)
        with torch.no_grad():
            ref_out = conv(data.x, data.edge_index).numpy()  # (c_size, out_ch)

        # --- our dense implementation ---
        dense_out = dense_gcn_layer(X, A_hat, W, b)           # (128, out_ch)

        np.testing.assert_allclose(
            dense_out[:c_size], ref_out,
            atol=1e-4, rtol=1e-4,
            err_msg=f"Dense GCN mismatch for '{smile}' at c_size={c_size}",
        )

    def test_padding_rows_zero_before_bias(self):
        """Padding rows in A_hat @ (X @ W^T) must be 0 when bias is 0."""
        rng = np.random.default_rng(0)
        W = rng.standard_normal((self.OUT_CH, FEAT_DIM)).astype(np.float32)
        b = np.zeros(self.OUT_CH, dtype=np.float32)
        X, A_hat, mask = smile_to_dense_graph("CCO", nmax=16)
        c_size = int(mask.sum())
        out = dense_gcn_layer(X, A_hat, W, b)
        assert np.allclose(out[c_size:], 0.0, atol=1e-6), \
            "padding rows should produce 0 when A_hat padding rows are 0 and b=0"

    def test_three_layer_composition_matches_pyg(self):
        """Three successive dense GCN layers should match three GCNConv layers."""
        from torch_geometric.nn import GCNConv
        torch.manual_seed(42)
        smile = "CC(=O)O"    # acetic acid, 4 atoms
        dims = [(FEAT_DIM, 32), (32, 64), (64, 128)]
        convs = [GCNConv(a, b, add_self_loops=True, normalize=True, bias=True)
                 for a, b in dims]

        X, A_hat, mask = smile_to_dense_graph(smile, nmax=32)
        c_size = int(mask.sum())
        data, _ = _smile_to_pyg_sparse(smile)

        # PyG three-layer pass
        with torch.no_grad():
            h = data.x
            for conv in convs:
                h = torch.relu(conv(h, data.edge_index))
            ref_out = h.numpy()   # (c_size, 128)

        # Dense three-layer pass
        H = X.copy()
        for conv in convs:
            W, b = _conv_weights(conv)
            H = dense_gcn_layer(H, A_hat, W, b)
            H = np.maximum(H, 0)   # ReLU
        dense_out = H[:c_size]

        np.testing.assert_allclose(
            dense_out, ref_out, atol=1e-4, rtol=1e-4,
            err_msg="3-layer dense GCN mismatch",
        )
