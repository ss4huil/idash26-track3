"""Plaintext GatedCNN protein encoder. Protein is PUBLIC; its 128-d embedding is
computed here (outside MPC) and enters the online graph as a public constant
(party 1 holds the value, party 0 holds zeros)."""
import os, numpy as np, torch
from torch_geometric.data import Batch
from reference import mpc_config
from baseline import official_baseline_data as ob

def protein_embedding(dataset: str, sample) -> np.ndarray:
    """Return the (128,) GatedCNN FC output for one PyG data sample."""
    model, device = ob.load_model(dataset)
    model.eval()
    with torch.no_grad():
        # Batch the sample: model.cnn expects data.target with batch dimension
        batch = Batch.from_data_list([sample]).to(device)
        xt, _ = model.cnn(batch)            # (1,128) protein FC output
    return xt.view(-1).cpu().numpy().astype(np.float64)

def export_protein_emb(pvec_float, out_dir: str,
                       scale: int = mpc_config.SCALE, bw: int = mpc_config.BW) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fixed = np.rint(np.asarray(pvec_float, np.float64) * (1 << scale)).astype(np.int64)
    # np.int64(1) << 64 overflows (shift >= width); at bw=64 int64 bit
    # patterns already ARE the ring residues, so cast directly.
    if bw >= 64:
        ring = fixed.astype(f"<u{bw // 8}")
    else:
        ring = np.mod(fixed, np.int64(1) << bw).astype(f"<u{bw // 8}")
    path = os.path.join(out_dir, mpc_config.PROTEIN_EMB_FILE)
    ring.tofile(path)
    return path
