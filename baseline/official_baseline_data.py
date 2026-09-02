"""
OFFICIAL plaintext baseline for the ciphertext MPC reference.

Runs the *original* DeepDTAGen (idash/project/DeepDTAGen) with the released
pretrained weights on the challenge test CSVs. Produces per-sample affinities +
metrics that the MPC pipeline is diffed against.

Design decisions (see conversation):
  * No training data needed. The affinity path is fc(PMVO, Protein_vector),
    fed only by the drug graph + protein sequence. The tokenized SMILES only
    drive the (discarded) generation decoder; we still supply them via the
    frozen tokenizer so the released weights load with the correct vocab size.
  * We reuse the OFFICIAL preprocessing (atom/bond/graph features, seq_cat) and
    OFFICIAL metrics from DeepDTAGen, so this is the released model, not a
    reproduction. create_data.py can't be imported (it reads the train CSVs at
    import time), so its deterministic feature functions are copied verbatim.

Env: pyenv 3.8.7 (matches DeepDTAGen environment.yml: torch 1.12.1+cu102, pyg,
rdkit 2022.09). fairseq 0.10.2 needs the numpy-alias shim below.
"""
import os
import sys
import json
import pickle

import numpy as np

# --- fairseq 0.10.2 uses numpy aliases removed in numpy>=1.24; restore them ---
for _name, _t in [("float", float), ("int", int), ("object", object),
                  ("bool", bool), ("str", str)]:
    if not hasattr(np, _name):
        setattr(np, _name, _t)

import pandas as pd
import networkx as nx
import torch
from rdkit import Chem
from torch_geometric import data as DATA
from torch_geometric.loader import DataLoader
from torch.nn.utils.rnn import pad_sequence

# DeepDTAGen project must be importable for model.py + utils.py
DEEPDTAGEN_DIR = os.environ.get("DEEPDTAGEN_DIR", "/home/ecs-user/idash26/DeepDTAGen")
if DEEPDTAGEN_DIR not in sys.path:
    sys.path.insert(0, DEEPDTAGEN_DIR)

from utils import Tokenizer  # noqa: E402  (official tokenizer + metrics live here)
import utils as ddg_utils    # noqa: E402
from model import DeepDTAGen  # noqa: E402

DATA_DIR = os.path.join(DEEPDTAGEN_DIR, "data")
MODEL_DIR = os.path.join(DEEPDTAGEN_DIR, "models")

# --- OFFICIAL preprocessing, copied verbatim from create_data.py -------------
# (create_data.py cannot be imported: it reads the training CSVs at import.)
_MAX_SEQ_LEN = 1000
_SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
_SEQ_DICT = {v: (i + 1) for i, v in enumerate(_SEQ_VOC)}


def _one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def _one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set] + [x not in allowable_set]


def _atom_features(atom):
    return np.array(
        _one_of_k_encoding_unk(atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
             'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
             'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
             'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
        _one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        _one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        _one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        _one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) +
        _one_of_k_encoding_unk(atom.GetHybridization(),
            [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
             Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
             Chem.rdchem.HybridizationType.SP3D2]) +
        [atom.GetIsAromatic()] +
        [atom.IsInRing()]
    )


def _bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [0, 0, 0, 0, bond.GetBondTypeAsDouble()]
    if bt == Chem.rdchem.BondType.SINGLE:
        bond_feats = [1, 0, 0, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.DOUBLE:
        bond_feats = [0, 1, 0, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.TRIPLE:
        bond_feats = [0, 0, 1, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.AROMATIC:
        bond_feats = [0, 0, 0, 1, bond.GetBondTypeAsDouble()]
    return np.array(bond_feats)


def _smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = _atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edge_feats = _bond_features(bond)
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                      {'edge_feats': edge_feats}))
    g = nx.Graph()
    g.add_edges_from(edges)
    g = g.to_directed()
    edge_index, edge_feats = [], []
    for e1, e2, feats in g.edges(data=True):
        edge_index.append([e1, e2])
        edge_feats.append(feats['edge_feats'])
    return c_size, features, edge_index, edge_feats


def _seq_cat(prot):
    x = np.zeros(_MAX_SEQ_LEN)
    for i, ch in enumerate(prot[:_MAX_SEQ_LEN]):
        x[i] = _SEQ_DICT[ch]
    return x


# --- public API --------------------------------------------------------------
def load_tokenizer(dataset):
    """Load the frozen tokenizer that matches the released checkpoint."""
    with open(os.path.join(DATA_DIR, f"{dataset}_tokenizer.pkl"), "rb") as f:
        return pickle.load(f)


def _build_from_df(df, tokenizer):
    """Build PyG Data objects from a dataframe slice with a given tokenizer.

    `target_seq` is padded to a constant length across THIS slice; since the
    affinity path never reads target_seq, per-slice padding does not change the
    predicted affinity, which is what makes chunked streaming exact.
    """
    drugs = list(df["compound_iso_smiles"])
    prots = list(df["target_sequence"])
    ys = list(df["affinity"])

    tok_smis = [torch.LongTensor(tokenizer.parse(s)) for s in drugs]
    pad_token = Tokenizer.SPECIAL_TOKENS.index("<pad>")
    smi = pad_sequence(tok_smis, batch_first=True, padding_value=pad_token)

    data_list = []
    for i in range(len(df)):
        c_size, features, edge_index, edge_feats = _smile_to_graph(drugs[i])
        target = _seq_cat(prots[i])
        gcn = DATA.Data(
            x=torch.Tensor(np.array(features)),
            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            edge_attr=torch.Tensor(np.array(edge_feats)),
            y=torch.FloatTensor([ys[i]]),
        )
        gcn.target = torch.LongTensor([target])
        gcn.target_seq = torch.LongTensor([smi[i].tolist()])
        gcn.__setitem__("c_size", torch.LongTensor([c_size]))
        data_list.append(gcn)
    return data_list


def build_dataset(dataset, csv_path, limit=None):
    """Build an in-memory list of PyG Data objects straight from the test CSV.

    No training data, no processed .pt cache. Kept for tests/small runs; large
    sets should stream via predict(chunk_size=...).
    """
    tokenizer = load_tokenizer(dataset)
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)
    return _build_from_df(df, tokenizer)


def dataset_row(csv_path, row_idx):
    """Read one test-CSV row's SMILES / protein sequence / affinity label."""
    r = pd.read_csv(csv_path).iloc[row_idx]
    return {"smile": str(r["compound_iso_smiles"]),
            "protein_seq": str(r["target_sequence"]),
            "y": float(r["affinity"])}


def _resolve_device():
    """Pick CUDA only if it can actually launch a kernel.

    torch.cuda.is_available() returns True for GPUs newer than this cu102
    build supports (e.g. RTX 4060, sm_89), but any real op then raises
    'no kernel image is available'. Probe with a tiny op and fall back to CPU.
    Override with BASELINE_DEVICE=cpu|cuda.
    """
    env = os.environ.get("BASELINE_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).cpu()
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def load_model(dataset, device=None):
    """Load DeepDTAGen with the released weights for `dataset`."""
    if device is None:
        device = _resolve_device()
    tokenizer = load_tokenizer(dataset)
    model = DeepDTAGen(tokenizer)
    state = torch.load(os.path.join(MODEL_DIR, f"deepdtagen_model_{dataset}.pth"),
                       map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def _predict_data_list(data_list, model, device, batch_size):
    """Run the model over a list of Data objects; return numpy (preds, trues)."""
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction, _, _, _ = model(batch)
            trues.append(batch.y.view(-1).cpu().numpy())
            preds.append(prediction.view(-1).cpu().numpy())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(trues)


def predict(dataset, csv_path, limit=None, batch_size=128, chunk_size=2000):
    """Return (predictions, ground_truth) numpy arrays over the test CSV.

    Streams the CSV in `chunk_size`-row blocks so memory stays bounded
    (kiba = 19653 rows OOMs at once in 8GB). chunk_size=None loads it all.
    Only the small float arrays are accumulated across chunks; each chunk's
    Data objects are freed before the next.
    """
    tokenizer = load_tokenizer(dataset)
    model, device = load_model(dataset)

    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    step = len(df) if chunk_size is None else chunk_size
    all_preds, all_trues = [], []
    for start in range(0, len(df), step):
        chunk_df = df.iloc[start:start + step]
        data_list = _build_from_df(chunk_df, tokenizer)
        p, t = _predict_data_list(data_list, model, device, batch_size)
        all_preds.append(p)
        all_trues.append(t)
        del data_list
    return np.concatenate(all_preds), np.concatenate(all_trues)


def cindex_bounded(Y, P, tile=1024):
    """Memory-bounded concordance index, identical value to utils.get_cindex.

    The official version builds N x N matrices (3.1 GB at kiba's N=19653 -> OOM).
    We tile the strictly-lower-triangle pairwise comparison, accumulating only
    scalar sums. Semantics preserved: over pairs (i>j) with Y[i]>Y[j], add
    1 if P[i]>P[j], 0.5 if equal, 0 otherwise; divide by that pair count.
    """
    Y = np.asarray(Y, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    n = len(Y)
    p_sum = 0.0
    y_sum = 0.0
    for i0 in range(0, n, tile):
        i1 = min(i0 + tile, n)
        Yi = Y[i0:i1][:, None]
        Pi = P[i0:i1][:, None]
        ii = np.arange(i0, i1)[:, None]
        for j0 in range(0, i1, tile):   # j blocks only up to i1 (need i>j)
            j1 = min(j0 + tile, n)
            dY = Yi - Y[j0:j1][None, :]
            m = dY > 0
            jj = np.arange(j0, j1)[None, :]
            m &= ii > jj                # strict lower triangle
            if not m.any():
                continue
            dP = Pi - P[j0:j1][None, :]
            pterm = (dP > 0).astype(np.float64) + 0.5 * (dP == 0)
            p_sum += float(np.sum(pterm[m]))
            y_sum += float(np.count_nonzero(m))
    if y_sum == 0:
        return 0
    return p_sum / y_sum


# challenge AUPR binarisation thresholds (from DeepDTAGen test.py)
_AUPR_THRESHOLD = {"kiba": 12.1, "davis": 7.0}


def evaluate(dataset, csv_path, limit=None, batch_size=128, chunk_size=2000):
    """Full metric set + per-sample predictions for the MPC reference."""
    predicted, ground_truth = predict(dataset, csv_path, limit=limit,
                                      batch_size=batch_size, chunk_size=chunk_size)
    thr = _AUPR_THRESHOLD.get(dataset, 7.0)
    return {
        "dataset": dataset,
        "csv": csv_path,
        "n": int(len(predicted)),
        "mse": float(ddg_utils.mse(ground_truth, predicted)),
        "rmse": float(ddg_utils.rmse(ground_truth, predicted)),
        "pearson": float(ddg_utils.pearson(ground_truth, predicted)),
        "spearman": float(ddg_utils.spearman(ground_truth, predicted)),
        "cindex": float(cindex_bounded(ground_truth, predicted)),
        "rm2": float(ddg_utils.get_rm2(ground_truth, predicted)),
        "aupr": float(ddg_utils.get_aupr(predicted, ground_truth, thr)),
        "aupr_threshold": thr,
        "predictions": [float(x) for x in predicted],
        "ground_truth": [float(x) for x in ground_truth],
    }


_DEFAULT_CSV = {
    "davis": "/home/jiang/master/idash/project/test/davis_test.csv",
    "kiba": "/home/jiang/master/idash/project/test/kiba_test.csv",
}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Official DeepDTAGen plaintext baseline")
    ap.add_argument("--dataset", choices=["davis", "kiba"], required=True)
    ap.add_argument("--csv", default=None, help="test CSV (defaults to challenge test set)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--chunk-size", type=int, default=2000,
                    help="rows built into RAM at once (bounds memory)")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    csv_path = args.csv or _DEFAULT_CSV[args.dataset]
    res = evaluate(args.dataset, csv_path, limit=args.limit,
                   batch_size=args.batch_size, chunk_size=args.chunk_size)

    out = args.out or os.path.join(os.path.dirname(__file__),
                                   f"official_baseline_{args.dataset}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"[{res['dataset']}] n={res['n']}  MSE={res['mse']:.4f}  "
          f"RMSE={res['rmse']:.4f}  CI={res['cindex']:.4f}  rm2={res['rm2']:.4f}  "
          f"Pearson={res['pearson']:.4f}  Spearman={res['spearman']:.4f}  "
          f"AUPR={res['aupr']:.4f}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
