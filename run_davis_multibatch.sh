#!/usr/bin/env bash
# run_davis_multibatch.sh <num_batches> <batch_size>
#
# Runs multiple B=<batch_size> MPC inferences over stratified davis samples,
# accumulating results for aggregate MAE/RMSE against CSV golden labels.
#
# GPU has only 8GB → batch_size=4 is the safe ceiling (B=16 OOMs at keygen).
set -euo pipefail

NUM_BATCHES="${1:-5}"
BATCH_SIZE="${2:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_MPC="$SCRIPT_DIR/gpu_mpc"
CSV="$SCRIPT_DIR/data/davis_test.csv"
RESULTS="$SCRIPT_DIR/davis_multibatch_results.log"

: > "$RESULTS"   # truncate

echo "[multibatch] Running $NUM_BATCHES batches of B=$BATCH_SIZE = $((NUM_BATCHES * BATCH_SIZE)) samples"

for ((batch=0; batch<NUM_BATCHES; batch++)); do
    BATCH_NAME="davis_mb_${batch}"
    BATCH_DIR="$GPU_MPC/$BATCH_NAME"  # absolute path
    KEY_DIR="/tmp/keys_mb_${batch}"

    echo ""
    echo "=== Batch $batch / $NUM_BATCHES (dir: $BATCH_NAME) ==="

    # Prepare this batch's samples (stratified offset by batch index)
    # NOTE: Generates weights.bin + secret shares (.dat) into $BATCH_DIR
    python3 "$SCRIPT_DIR/scripts/dev_tools/prepare_davis_multibatch_slice.py" \
        "$batch" "$BATCH_SIZE" "$BATCH_NAME" 2>&1 | grep -E "(row|Selected|error)" || true

    # Run MPC inference with ABSOLUTE SAMPLE_DIR path
    BATCH="$BATCH_SIZE" "$GPU_MPC/run_local_2pc.sh" \
        "$BATCH_DIR" "$KEY_DIR" "$BATCH_DIR/weights.bin" \
        > "/tmp/mb_${batch}.log" 2>&1 || {
            echo "[multibatch] Batch $batch FAILED, see /tmp/mb_${batch}.log"
            tail -5 "/tmp/mb_${batch}.log"
            continue
        }

    # Append AFFINITY results tagged with batch name
    echo "### BATCH=$batch NAME=$BATCH_NAME" >> "$RESULTS"
    grep -E "^AFFINITY" "/tmp/mb_${batch}.log" >> "$RESULTS" || true

    # Clean up keys to save disk (each ~679MB/party)
    rm -rf "$KEY_DIR"

    echo "[multibatch] Batch $batch done"
    grep -E "^AFFINITY" "/tmp/mb_${batch}.log" | head
done

echo ""
echo "[multibatch] All batches complete. Results in $RESULTS"
echo "[multibatch] Run aggregate validation:"
echo "  python3 $SCRIPT_DIR/scripts/dev_tools/aggregate_davis_validation.py $NUM_BATCHES"
