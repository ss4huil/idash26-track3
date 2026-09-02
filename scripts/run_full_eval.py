#!/usr/bin/env python3
"""Full-dataset sequential 2PC evaluation with resume support.

Iterates the test CSV in row order, runs batched 2PC (B=8) for each slice,
records per-sample MPC affinities incrementally, and reports final regression
metrics vs CSV labels and vs the official plaintext baseline JSON.

Unlike run_local_2pc.sh, this driver invokes the binary directly because the
script only echoes the last AFFINITY= line and deletes the per-party logs on
exit; batched output uses AFFINITY[i]= lines that must all be captured.

Usage:
  python3 scripts/run_full_eval.py davis [--batch-size 8] [--num-batches N]
  python3 scripts/run_full_eval.py kiba  --csv data/kiba_test.csv

Results: results_full_<dataset>.jsonl (one line per batch; re-running skips
batches already recorded -> safe resume after interruption).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "dev_tools"))

GPU_DIR = os.path.join(ROOT, "gpu_mpc")
BINARY = os.path.join(GPU_DIR, "deepdtagen_inference")
BW, SCALE = 32, 12
IP = "127.0.0.1"

ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + ENV.get("LD_LIBRARY_PATH", "")


def run_2pc(sample_dir, key_dir, batch, log_prefix):
    """dealer x2 + online 2PC; returns party-0 log text."""
    os.makedirs(key_dir, exist_ok=True)
    kd = key_dir.rstrip("/") + "/"
    env = dict(ENV)
    env["DDG_WEIGHTS_BIN"] = os.path.join(os.path.abspath(sample_dir), "weights.bin")
    for party in (0, 1):  # dealer keygen
        subprocess.run([BINARY, str(BW), str(SCALE), "0", str(party), kd,
                        sample_dir, str(batch)], check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p1 = subprocess.Popen([BINARY, str(BW), str(SCALE), "1", "1", kd,
                           sample_dir, str(batch), IP],
                          stdout=open(log_prefix + ".p1", "w"),
                          stderr=subprocess.STDOUT, env=env)
    time.sleep(1)
    with open(log_prefix + ".p0", "w") as f0:
        p0 = subprocess.run([BINARY, str(BW), str(SCALE), "1", "0", kd,
                             sample_dir, str(batch), IP],
                            stdout=f0, stderr=subprocess.STDOUT, env=env)
    p1.wait()
    if p0.returncode != 0 or p1.returncode != 0:
        raise RuntimeError(f"online failed rc p0={p0.returncode} p1={p1.returncode}")
    with open(log_prefix + ".p0") as f:
        return f.read()


def parse_affinities(p0_log):
    out = {}
    for line in p0_log.splitlines():
        line = line.strip()
        if line.startswith("AFFINITY[") and "=" in line:
            i = int(line[len("AFFINITY["):line.index("]")])
            out[i] = float(line.split("=", 1)[1])
        elif line.startswith("AFFINITY="):
            out[0] = float(line.split("=", 1)[1])
    return [out[i] for i in sorted(out)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["davis", "kiba"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-batches", type=int, default=None, help="limit batches (for testing)")
    ap.add_argument("--key-dir", default="/dev/shm/keys_full_eval")
    ap.add_argument("--work-root", default=os.path.join(GPU_DIR, "full_eval"))
    args = ap.parse_args()

    csv_path = args.csv or os.path.join(ROOT, "data", f"{args.dataset}_test.csv")
    results_path = os.path.join(ROOT, f"results_full_{args.dataset}.jsonl")

    from prepare_batch_samples import prepare_batch_samples

    df = pd.read_csv(csv_path)
    n = len(df)
    B = args.batch_size
    batches = [(i, list(range(i, min(i + B, n)))) for i in range(0, n, B)]
    if args.num_batches:
        batches = batches[:args.num_batches]

    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                done.add(json.loads(line)["batch"])

    os.makedirs(args.work_root, exist_ok=True)
    print(f"[full_eval] {args.dataset}: {n} rows, {len(batches)} batches of B={B}, "
          f"{len(done)} already done", flush=True)

    for b_idx, rows in batches:
        if b_idx in done:
            continue
        name = f"batch_{b_idx:05d}"
        bdir = os.path.join(args.work_root, name)
        t0 = time.time()
        prepare_batch_samples(dataset=args.dataset, csv_path=csv_path,
                              row_indices=rows, out_dir=args.work_root,
                              batch_name=name, scale=SCALE, bw=BW)
        t1 = time.time()
        p0_log = run_2pc(bdir, args.key_dir, len(rows), os.path.join("/tmp", name))
        affs = parse_affinities(p0_log)
        t2 = time.time()
        if len(affs) != len(rows):
            raise RuntimeError(f"batch {b_idx}: got {len(affs)} affinities for {len(rows)} rows")
        rec = {"batch": b_idx, "row_indices": rows, "affinities": affs,
               "prep_s": round(t1 - t0, 2), "mpc_s": round(t2 - t1, 2)}
        with open(results_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        shutil.rmtree(bdir, ignore_errors=True)
        shutil.rmtree(args.key_dir, ignore_errors=True)
        done.add(b_idx)
        eta = (len(batches) - len(done)) * (t2 - t0)
        print(f"[full_eval] batch {b_idx} ({len(done)}/{len(batches)}) "
              f"prep={rec['prep_s']}s mpc={rec['mpc_s']}s ETA={eta/3600:.1f}h", flush=True)

    # ---- final metrics ----
    recs = [json.loads(l) for l in open(results_path)]
    recs.sort(key=lambda r: r["batch"])
    pred = np.array([a for r in recs for a in r["affinities"]])
    rows = [i for r in recs for i in r["row_indices"]]
    y = df["affinity"].to_numpy()[rows]
    from scipy import stats
    mse = float(np.mean((pred - y) ** 2))
    summary = {
        "dataset": args.dataset, "csv": csv_path, "n": int(len(pred)),
        "mae": float(np.mean(np.abs(pred - y))), "mse": mse, "rmse": float(np.sqrt(mse)),
        "pearson": float(stats.pearsonr(pred, y)[0]),
        "spearman": float(stats.spearmanr(pred, y)[0]),
    }
    base_json = os.path.join(ROOT, "baseline", f"official_baseline_{args.dataset}.json")
    if os.path.exists(base_json):
        base = np.array(json.load(open(base_json))["predictions"])[rows]
        summary["vs_official_baseline"] = {
            "mae": float(np.mean(np.abs(pred - base))),
            "max_abs": float(np.max(np.abs(pred - base))),
            "mse": float(np.mean((pred - base) ** 2)),
        }
    out_json = os.path.join(ROOT, f"results_full_{args.dataset}_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
