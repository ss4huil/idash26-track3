"""
Reference-compatible NPZ export for the official BumbleBee driver.

Produces the exact array layout the flax secure driver's `load_npz` expects
(see reference_bumblebee/flax_secure_deepdtagen.py::export_npz_from_deepdtagen):

    drug_x    (N, nmax, 94)   float32  L1-normalised atom features
    adj       (N, nmax, nmax) float32  RAW adjacency, NO self-loops
    node_mask (N, nmax)       bool     True for real atoms
    protein   (N, 1000)       int32    seq_cat-encoded protein
    y         (N, 1)          float32  affinity label

The self-loop and symmetric normalisation are applied *inside* the reference
GCNLayer, so the exported adjacency is the raw bond graph. This differs from
`dense_graph.smile_to_dense_graph`, which pre-computes the normalised A_hat for
the MPC-accelerated path; both are numerically equivalent (see
tests/test_reference_equivalence.py).
"""
import numpy as np
from rdkit import Chem

from reference.dense_graph import atom_features, FEAT_DIM
from reference.affinity_model import seq_cat, _MAX_SEQ_LEN


def to_npz_record(smile: str, protein: str, y: float, nmax: int = 138) -> dict:
    """Build one reference-format record from (SMILES, protein, affinity)."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smile!r}")
    c_size = mol.GetNumAtoms()
    if c_size > nmax:
        raise ValueError(f"molecule has {c_size} atoms > nmax={nmax}")

    # L1-normalised atom features (matches DeepDTAGen/create_data.py data.x)
    drug_x = np.zeros((nmax, FEAT_DIM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        f = atom_features(atom)
        drug_x[i] = f / f.sum()

    # RAW adjacency, no self-loops (self-loop added inside the reference GCNLayer)
    adj = np.zeros((nmax, nmax), dtype=np.float32)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[a, b] = 1.0
        adj[b, a] = 1.0

    node_mask = np.zeros(nmax, dtype=bool)
    node_mask[:c_size] = True

    protein_enc = seq_cat(protein).astype(np.int32)          # (1000,)

    return {
        "drug_x": drug_x,
        "adj": adj,
        "node_mask": node_mask,
        "protein": protein_enc,
        "y": np.float32(y),
    }


def export_npz(pairs, out_path: str, nmax: int = 138) -> str:
    """Export an iterable of (smile, protein, y) to a reference-format NPZ.

    `pairs` items are (smile, protein, y) tuples. Writes `out_path` and returns
    it. Arrays are stacked in input order.
    """
    recs = [to_npz_record(s, p, y, nmax=nmax) for (s, p, y) in pairs]
    drug_x    = np.stack([r["drug_x"] for r in recs]).astype(np.float32)
    adj       = np.stack([r["adj"] for r in recs]).astype(np.float32)
    node_mask = np.stack([r["node_mask"] for r in recs]).astype(bool)
    protein   = np.stack([r["protein"] for r in recs]).astype(np.int32)
    y         = np.asarray([r["y"] for r in recs], dtype=np.float32).reshape(-1, 1)

    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, drug_x=drug_x, adj=adj, node_mask=node_mask,
                        protein=protein, y=y)
    return out_path
