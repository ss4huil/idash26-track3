#!/usr/bin/env python3
"""
Prepare a batch of samples for MPC inference.

Creates a batch directory with:
  - {x,adj,mask}_share{0,1}.dat — batched secret shares (B, ...)
  - protein_emb.dat — batched protein embeddings (B, 128)
  - batch_manifest.json — metadata (row_indices, shapes, SMILES, sequences)
  - golden_affinities.json — ground truth affinities for validation

Usage:
    from prepare_batch_samples import prepare_batch_samples

    prepare_batch_samples(
        dataset="davis",
        csv_path="/path/to/davis_test.csv",
        row_indices=[0, 3, 5, 9],
        out_dir="gpu_mpc",
        batch_name="test_batch_4",
        scale=12,
        bw=32
    )
"""
import os
import sys
import json
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from reference import mpc_config, share_data, protein_plaintext, export_weights
from reference.affinity_model import AffinityModel
from reference.dense_graph import smile_to_dense_graph, smile_to_dense_raw_graph
from baseline import official_baseline_data as ob


def prepare_batch_samples(dataset: str, csv_path: str, row_indices: list,
                         out_dir: str, batch_name: str,
                         scale: int = 12, bw: int = 32,
                         raw_adj: bool = False) -> dict:
    """
    Prepare a batch of samples for MPC inference.

    Args:
        dataset: "davis", "kiba", or "bindingdb"
        csv_path: Path to test CSV file
        row_indices: List of row indices to include in batch
        out_dir: Output directory (batch will be created as out_dir/batch_name/)
        batch_name: Name of the batch subdirectory
        scale: Fixed-point scale (fractional bits)
        bw: Bit width (32 or 64)
        raw_adj: If True, write the RAW 0/1 adjacency (self-loops, no degree
                 normalization) as adj shares at scale=0, for the
                 DDG_SECURE_ADJ_NORM online-normalization path. mask shares
                 stay at `scale` (masked-maxpool encoding unchanged).

    Returns:
        dict: Manifest containing batch metadata
    """
    B = len(row_indices)
    NMAX = mpc_config.NMAX
    FEAT_DIM = 94  # SMILES feature dimension
    POOL_DIM = mpc_config.POOL_DIM

    print(f"[prepare_batch] Preparing batch: {batch_name}")
    print(f"[prepare_batch] Dataset: {dataset}, rows: {row_indices}")
    print(f"[prepare_batch] Batch size: {B}, scale: {scale}, bw: {bw}")

    # Create batch directory
    batch_dir = os.path.join(out_dir, batch_name)
    os.makedirs(batch_dir, exist_ok=True)

    # Initialize batched arrays (float — split_shares quantises internally)
    X_batch = np.zeros((B, NMAX, FEAT_DIM), dtype=np.float64)
    A_batch = np.zeros((B, NMAX, NMAX), dtype=np.float64)
    mask_batch = np.zeros((B, NMAX, POOL_DIM), dtype=np.float64)
    protein_emb_batch = np.zeros((B, 128), dtype=np.float64)

    smiles_list = []
    protein_seqs = []
    golden_affinities = {}

    # Load dataset for protein embeddings
    ds = ob.build_dataset(dataset, csv_path, limit=max(row_indices) + 1)

    # Process each sample
    for batch_idx, row_idx in enumerate(row_indices):
        print(f"[prepare_batch]   Processing sample {batch_idx}/{B}: row {row_idx}")

        # Load row from CSV
        row = ob.dataset_row(csv_path, row_idx)
        smiles_list.append(row["smile"])
        protein_seqs.append(row["protein_seq"])

        # Convert SMILES to dense graph (float tensors)
        if raw_adj:
            # Compliant path: raw 0/1 adjacency + self-loops, no normalization
            X_fp, A_adj_fp, mask_fp = smile_to_dense_raw_graph(row["smile"], NMAX)
        else:
            X_fp, A_adj_fp, mask_fp = smile_to_dense_graph(row["smile"], NMAX)

        # Tile mask to (NMAX, POOL_DIM) for maxpool — column replication
        # (matches share_data.share_drug_graph pre-tiling contract)
        mask_tiled = np.broadcast_to(mask_fp.reshape(NMAX, 1), (NMAX, POOL_DIM))

        X_batch[batch_idx] = X_fp
        A_batch[batch_idx] = A_adj_fp
        mask_batch[batch_idx] = mask_tiled

        # Get protein embedding (public, stored in fixed point)
        sample = ds[row_idx]
        pvec = protein_plaintext.protein_embedding(dataset, sample)
        protein_emb_batch[batch_idx] = np.asarray(pvec).reshape(-1)

        # Store golden affinity
        y_value = float(np.asarray(sample.y).reshape(-1)[0])
        golden_affinities[str(row_idx)] = y_value

    # Generate secret shares using the reference implementation
    # (ensures compatibility with MPC binary's share format)
    seed = hash(batch_name) % (2**31)

    # Use share_data.split_shares for correct fixed-point + modular arithmetic
    X_s0, X_s1 = share_data.split_shares(X_batch, scale=scale, seed=seed + 0, bw=bw)
    # raw_adj: adjacency is a 0/1 integer tensor — share it at scale=0
    adj_scale = 0 if raw_adj else scale
    A_s0, A_s1 = share_data.split_shares(A_batch, scale=adj_scale, seed=seed + 1, bw=bw)
    mask_s0, mask_s1 = share_data.split_shares(mask_batch, scale=scale, seed=seed + 2, bw=bw)

    # Reshape back to tensor shapes and write
    nbytes = bw // 8
    dtype_str = f"<u{nbytes}"

    X_s0.reshape(X_batch.shape).astype(dtype_str).tofile(os.path.join(batch_dir, "x_share0.dat"))
    X_s1.reshape(X_batch.shape).astype(dtype_str).tofile(os.path.join(batch_dir, "x_share1.dat"))
    A_s0.reshape(A_batch.shape).astype(dtype_str).tofile(os.path.join(batch_dir, "adj_share0.dat"))
    A_s1.reshape(A_batch.shape).astype(dtype_str).tofile(os.path.join(batch_dir, "adj_share1.dat"))
    mask_s0.reshape(mask_batch.shape).astype(dtype_str).tofile(os.path.join(batch_dir, "mask_share0.dat"))
    mask_s1.reshape(mask_batch.shape).astype(dtype_str).tofile(os.path.join(batch_dir, "mask_share1.dat"))

    # Write protein embeddings (no sharing, party 1 loads this)
    # Quantize to fixed-point using the same scheme as export_protein_emb
    protein_fx = np.rint(protein_emb_batch * (1 << scale)).astype(np.int64)
    protein_ring = np.mod(protein_fx, np.int64(1) << bw).astype(dtype_str)
    protein_ring.tofile(os.path.join(batch_dir, "protein_emb.dat"))

    # Create manifest
    manifest = {
        "batch_size": B,
        "row_indices": row_indices,
        "dataset": dataset,
        "scale": scale,
        "bw": bw,
        "raw_adj": raw_adj,
        "adj_scale": 0 if raw_adj else scale,
        "nmax": NMAX,
        "feat_dim": FEAT_DIM,
        "pool_dim": POOL_DIM,
        "shapes": {
            "X": list(X_batch.shape),
            "A_hat": list(A_batch.shape),
            "mask": list(mask_batch.shape),
            "protein_emb": list(protein_emb_batch.shape)
        },
        "smiles": smiles_list,
        "protein_seqs": protein_seqs
    }

    # Write manifest
    manifest_path = os.path.join(batch_dir, "batch_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Write golden affinities
    golden_path = os.path.join(batch_dir, "golden_affinities.json")
    with open(golden_path, "w") as f:
        json.dump(golden_affinities, f, indent=2)

    # Export weights (shared across all batches)
    model_dir = os.path.join(os.path.dirname(__file__), "../..", "model")
    model_pth = os.path.join(model_dir, f"deepdtagen_model_{dataset}.pth")
    weights_path = os.path.join(batch_dir, "weights.bin")

    if not os.path.exists(weights_path):
        print(f"[prepare_batch] Exporting weights to {weights_path}")
        export_weights.dump_mpc_weights(
            AffinityModel.from_pth(model_pth),
            weights_path,
            scale=scale
        )
    else:
        print(f"[prepare_batch] Weights already exist: {weights_path}")

    print(f"[prepare_batch] ✓ Batch ready: {batch_dir}")
    print(f"[prepare_batch]   Files: x/adj/mask_share{{0,1}}.dat, protein_emb.dat")
    print(f"[prepare_batch]   Manifest: {manifest_path}")
    print(f"[prepare_batch]   Golden: {golden_path}")

    return manifest


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 5:
        print(__doc__)
        print("\nExample:")
        print("  python prepare_batch_samples.py davis /path/to/davis_test.csv 0,3,5,9 test_batch")
        sys.exit(1)

    dataset = sys.argv[1]
    csv_path = sys.argv[2]
    row_indices = [int(x) for x in sys.argv[3].split(",")]
    batch_name = sys.argv[4]
    out_dir = sys.argv[5] if len(sys.argv) > 5 else "gpu_mpc"

    prepare_batch_samples(
        dataset=dataset,
        csv_path=csv_path,
        row_indices=row_indices,
        out_dir=out_dir,
        batch_name=batch_name,
        scale=12,
        bw=32
    )
