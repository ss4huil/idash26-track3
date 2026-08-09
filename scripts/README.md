# Development Scripts

This directory contains utilities for benchmarking and dataset testing.

## Timing & Benchmarking

**`benchmark_plaintext_time.py`** — Measure plaintext DeepDTAGen inference time
- Tests on multiple davis samples (0, 3, 5, 9)
- Reports mean/min/max inference time with GPU synchronization
- Compares predictions vs ground truth

**`benchmark_plaintext_single.py`** — Benchmark plaintext affinity-only path
- Focuses on the exact computation path MPC executes (GCN + fusion FC)
- 3 warmup + 20 measured runs per sample
- Outputs per-sample and aggregate timing statistics

## Davis Dataset Testing

**`prepare_batch_samples.py`** — Prepare a batch of MPC samples
- Core utility for batch sample generation
- Creates batched secret shares: `{x,adj,mask}_share{0,1}.dat`
- Generates `batch_manifest.json` and `golden_affinities.json`
- Usage:
  ```python
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
  ```

**`prepare_davis_multibatch_slice.py`** — Stratified davis batch sampling
- Generates one batch with stratified affinity distribution (5 bins)
- Uses batch_idx as seed offset for non-overlapping batches
- Usage: `python prepare_davis_multibatch_slice.py <batch_idx> <batch_size> <batch_name>`

**`validate_davis_batch.py`** — Validate MPC outputs vs golden labels
- Compares MPC inference results with CSV ground truth
- Computes MAE, RMSE, and max error
- Shows per-sample breakdown and error percentiles
- Usage: `python validate_davis_batch.py <batch_name> <mpc_output.log> <golden_csv>`

**`aggregate_davis_validation.py`** — Aggregate multi-batch validation
- Combines results across multiple batches
- Reports aggregate MAE/RMSE for full test runs
- Usage: `python aggregate_davis_validation.py <num_batches>`

## Workflow Example

Prepare stratified davis batches → Run 2PC → Validate results:

```bash
# 1. Prepare 5 batches of 4 samples each
for i in {0..4}; do
    python prepare_davis_multibatch_slice.py $i 4 davis_batch_$i
done

# 2. Run 2PC on each batch
for i in {0..4}; do
    cd gpu_mpc
    BATCH=4 ./run_local_2pc.sh davis_batch_$i /tmp/keys weights.bin > batch_${i}.log
    cd ..
done

# 3. Validate each batch
for i in {0..4}; do
    python validate_davis_batch.py davis_batch_$i \
        gpu_mpc/batch_${i}.log \
        /path/to/davis_test.csv
done

# 4. Aggregate results
python aggregate_davis_validation.py 5
```

## Notes

- These scripts were preserved from development for performance analysis and dataset testing
- For regular usage, use the main project scripts:
  - [real_gpu_2pc_benchmark.py](../../real_gpu_2pc_benchmark.py) — Real 2PC benchmark (timing + accuracy)
  - [run_davis_multibatch.sh](../../run_davis_multibatch.sh) — Batched Davis MPC evaluation
