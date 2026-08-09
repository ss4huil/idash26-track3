#!/usr/bin/env python3
"""Real GPU 2PC benchmark on a small sample subset.

Runs actual cryptographic 2PC via deepdtagen_inference binary and compares
against plaintext float predictions for regression metrics and timing.
"""
import os
import sys
import time
import json
import subprocess
import tempfile
import shutil
import numpy as np
from scipy import stats

# Ensure we can import from reference/baseline
sys.path.insert(0, os.path.dirname(__file__))

# The released davis/kiba checkpoints have a decoder encoder-attn kdim/vdim
# mismatch (376x376 saved vs 376x512 in current model.py). We only need
# model.cnn for the public protein embedding, so install a lenient loader that
# skips the mismatched decoder weights (mirrors prepare_sample.py).
import torch as _torch
import baseline.official_baseline_data as _ob
from model import DeepDTAGen as _DeepDTAGen

def _load_model_lenient(dataset, device=None):
    if device is None:
        device = _ob._resolve_device()
    tokenizer = _ob.load_tokenizer(dataset)
    model = _DeepDTAGen(tokenizer)
    state = _torch.load(os.path.join(_ob.MODEL_DIR, f"deepdtagen_model_{dataset}.pth"),
                        map_location=device)
    model_state = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v
        elif k.startswith("cnn"):
            raise RuntimeError(f"CNN weight shape mismatch: {k} {v.shape} vs {model_state[k].shape}")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    bad = [k for k in missing if k.startswith("cnn")]
    if bad:
        raise RuntimeError(f"CNN weights missing from checkpoint: {bad}")
    model.to(device)
    model.eval()
    return model, device

_ob.load_model = _load_model_lenient

from reference.affinity_model import AffinityModel
from reference.offline_prepare import prepare_sample, model_pth
from reference.dense_graph import smile_to_dense_graph
from baseline import official_baseline_data as ob

# Paths
MPC_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MPC_ROOT, "data")
GPU_MPC_DIR = os.path.join(MPC_ROOT, "gpu_mpc")
RUN_2PC_SCRIPT = os.path.join(GPU_MPC_DIR, "run_local_2pc.sh")

DATASETS = {
    "davis": os.path.join(DATA_DIR, "davis_test.csv"),
    "kiba": os.path.join(DATA_DIR, "kiba_train.csv"),
}

# Sample subset size per dataset
N_SAMPLES = 10

def get_plaintext_prediction(model, smile, protein_seq, nmax=138):
    """Get float plaintext prediction from AffinityModel."""
    X, A_hat, mask = smile_to_dense_graph(smile, nmax)
    return float(model.predict(X, A_hat, mask, protein_seq))

def run_gpu_2pc_sample(sample_dir, weights_bin):
    """Run GPU 2PC for one sample and return (affinity, elapsed_seconds)."""
    with tempfile.TemporaryDirectory() as key_dir:
        start = time.perf_counter()
        result = subprocess.run(
            [RUN_2PC_SCRIPT, sample_dir, key_dir, weights_bin],
            capture_output=True,
            text=True,
            check=True
        )
        elapsed = time.perf_counter() - start

        # Extract AFFINITY= from stdout
        for line in result.stdout.strip().split('\n'):
            if line.startswith("AFFINITY="):
                affinity = float(line.split("=")[1])
                return affinity, elapsed

        raise RuntimeError(f"AFFINITY= not found in 2PC output:\n{result.stdout}")

def compute_regression_metrics(y_true, y_pred):
    """Compute MSE, RMSE, Pearson, Spearman correlation."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    spearman_r, _ = stats.spearmanr(y_true, y_pred)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "pearson": float(pearson_r),
        "spearman": float(spearman_r),
    }

def benchmark_dataset(dataset_name, csv_path, n_samples):
    """Run benchmark on n_samples from the given dataset."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {dataset_name.upper()} (n={n_samples})")
    print(f"{'='*60}\n")

    # Load plaintext float model
    print(f"Loading plaintext model from {model_pth(dataset_name)}...")
    float_model = AffinityModel.from_pth(model_pth(dataset_name))

    # Create temp dir for sample preparation
    work_dir = tempfile.mkdtemp(prefix=f"bench_{dataset_name}_")
    print(f"Work directory: {work_dir}\n")

    results = {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "samples": [],
        "plaintext_times": [],
        "mpc_times": [],
        "y_true": [],
        "y_plaintext": [],
        "y_mpc": [],
    }

    try:
        for i in range(n_samples):
            print(f"[{i+1}/{n_samples}] Processing sample {i}...")

            # Prepare sample
            print(f"  Preparing offline artifacts...")
            manifest = prepare_sample(dataset_name, csv_path, i, work_dir, scale=12, bw=32)
            sample_dir = manifest["sample_dir"]
            weights_bin = manifest["weights_path"]
            smile = manifest["smile"]
            protein_seq = manifest["protein_seq"]
            y_true = manifest["y"]

            results["y_true"].append(y_true)

            # Plaintext float prediction
            print(f"  Running plaintext float inference...")
            t0 = time.perf_counter()
            y_plaintext = get_plaintext_prediction(float_model, smile, protein_seq)
            plaintext_time = time.perf_counter() - t0
            results["y_plaintext"].append(y_plaintext)
            results["plaintext_times"].append(plaintext_time)
            print(f"    Plaintext: {y_plaintext:.6f} ({plaintext_time:.3f}s)")

            # GPU 2PC prediction
            print(f"  Running GPU 2PC (dealer + online phases)...")
            y_mpc, mpc_time = run_gpu_2pc_sample(sample_dir, weights_bin)
            results["y_mpc"].append(y_mpc)
            results["mpc_times"].append(mpc_time)
            print(f"    MPC (2PC): {y_mpc:.6f} ({mpc_time:.1f}s)")

            results["samples"].append({
                "idx": i,
                "y_true": y_true,
                "y_plaintext": y_plaintext,
                "y_mpc": y_mpc,
                "plaintext_time": plaintext_time,
                "mpc_time": mpc_time,
            })
            print()

        # Compute metrics
        print(f"\n{'='*60}")
        print(f"Results for {dataset_name.upper()}")
        print(f"{'='*60}\n")

        plaintext_metrics = compute_regression_metrics(results["y_true"], results["y_plaintext"])
        mpc_metrics = compute_regression_metrics(results["y_true"], results["y_mpc"])

        avg_plaintext_time = np.mean(results["plaintext_times"])
        avg_mpc_time = np.mean(results["mpc_times"])
        total_mpc_time = np.sum(results["mpc_times"])

        print(f"Plaintext Float:")
        print(f"  MSE:      {plaintext_metrics['mse']:.6f}")
        print(f"  RMSE:     {plaintext_metrics['rmse']:.6f}")
        print(f"  Pearson:  {plaintext_metrics['pearson']:.6f}")
        print(f"  Spearman: {plaintext_metrics['spearman']:.6f}")
        print(f"  Avg time: {avg_plaintext_time:.3f}s/sample")
        print()

        print(f"GPU 2PC (Cryptographic):")
        print(f"  MSE:      {mpc_metrics['mse']:.6f}")
        print(f"  RMSE:     {mpc_metrics['rmse']:.6f}")
        print(f"  Pearson:  {mpc_metrics['pearson']:.6f}")
        print(f"  Spearman: {mpc_metrics['spearman']:.6f}")
        print(f"  Avg time: {avg_mpc_time:.1f}s/sample")
        print(f"  Total:    {total_mpc_time:.1f}s for {n_samples} samples")
        print()

        print(f"Accuracy Degradation:")
        print(f"  ΔMSE:     {mpc_metrics['mse'] - plaintext_metrics['mse']:.6f}")
        print(f"  ΔPearson: {mpc_metrics['pearson'] - plaintext_metrics['pearson']:.6f}")
        print()

        print(f"Slowdown: {avg_mpc_time / avg_plaintext_time:.1f}x")
        print()

        results["plaintext_metrics"] = plaintext_metrics
        results["mpc_metrics"] = mpc_metrics
        results["avg_plaintext_time"] = avg_plaintext_time
        results["avg_mpc_time"] = avg_mpc_time
        results["total_mpc_time"] = total_mpc_time

        return results

    finally:
        # Cleanup
        print(f"Cleaning up {work_dir}...")
        shutil.rmtree(work_dir, ignore_errors=True)

def check_prerequisites():
    """Verify the compiled binary and model weights exist before benchmarking.

    Secret shares + weights.bin are generated per-sample by prepare_sample()
    during the run, so they do NOT need to exist beforehand — but the 2PC
    binary must be compiled and the .pth checkpoints must be present.
    """
    problems = []

    binary = os.path.join(GPU_MPC_DIR, "deepdtagen_inference")
    if not os.path.exists(binary):
        problems.append(
            f"  ✗ 2PC binary not found: {binary}\n"
            f"    Build it:  cd gpu_mpc && make GPU_MPC_ROOT=$GPU_MPC_ROOT "
            f"BW=32 GPU_ARCH=89 deepdtagen_inference"
        )

    for ds in DATASETS:
        pth = os.path.join(_ob.MODEL_DIR, f"deepdtagen_model_{ds}.pth")
        if not os.path.exists(pth):
            problems.append(f"  ✗ Model weights missing: {pth}")

    for ds, csv in DATASETS.items():
        if not os.path.exists(csv):
            problems.append(f"  ✗ Dataset CSV missing: {csv}")

    if problems:
        print("Prerequisites not met (offline artifacts are auto-generated, "
              "but these are required):")
        print("\n".join(problems))
        sys.exit(1)


def main():
    check_prerequisites()
    all_results = {}

    for dataset_name, csv_path in DATASETS.items():
        results = benchmark_dataset(dataset_name, csv_path, N_SAMPLES)
        all_results[dataset_name] = results

    # Save results
    output_path = os.path.join(MPC_ROOT, "real_2pc_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Benchmark complete. Results saved to:")
    print(f"  {output_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
