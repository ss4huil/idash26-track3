#!/usr/bin/env bash
# p0_regression.sh <dataset> — dealer + 2-party eval for timing_b8_raw_<ds>_bw64
# with the v2 defaults (LocalARS, SECURE_ADJ_NORM=1, KEY_ARENA=0), NC=1, B=8.
# Extra env passed to BOTH dealer and eval via EXTRA_ENV (e.g. "DDG_EXACT_TRUNC=1").
set -euo pipefail
DS=${1:?usage: p0_regression.sh <davis|kiba|bindingdb> [keydir] [extra_env]}
KEYDIR=${2:-/dev/shm/p0_keys_$DS}
EXTRA=${3:-}
ROOT=/home/ecs-user/idash26/gpu-mpc-track
BIN=$ROOT/gpu_mpc/deepdtagen_inference_bw64
SAMPLE=$ROOT/gpu_mpc/timing_b8_raw_${DS}_bw64
BW=64; SCALE=12; B=8
rm -rf "$KEYDIR"; mkdir -p "$KEYDIR"

for p in 0 1; do
  env DDG_SECURE_ADJ_NORM=1 DDG_KEYBUF_CAP_GB=${DDG_KEYBUF_CAP_GB:-8} \
      DDG_WEIGHTS_BIN="$SAMPLE/weights.bin" $EXTRA \
      "$BIN" $BW $SCALE 0 $p "$KEYDIR/" "$SAMPLE" $B >"$KEYDIR/dealer_p$p.log" 2>&1
done

env DDG_SECURE_ADJ_NORM=1 DDG_KEY_ARENA=0 DDG_WEIGHTS_BIN="$SAMPLE/weights.bin" $EXTRA \
    "$BIN" $BW $SCALE 1 1 "$KEYDIR/" "$SAMPLE" $B 127.0.0.1 >"$KEYDIR/eval_p1.log" 2>&1 &
P1=$!
sleep 2
env DDG_SECURE_ADJ_NORM=1 DDG_KEY_ARENA=0 DDG_WEIGHTS_BIN="$SAMPLE/weights.bin" $EXTRA \
    "$BIN" $BW $SCALE 1 0 "$KEYDIR/" "$SAMPLE" $B 127.0.0.1 >"$KEYDIR/eval_p0.log" 2>&1 &
P0=$!
RC1=0; RC0=0
wait $P1 || RC1=$?
wait $P0 || RC0=$?
echo "[p0_regression] eval rc: p0=$RC0 p1=$RC1"
grep -c 'AFFINITY\[' "$KEYDIR/eval_p0.log" || true
exit $((RC0 || RC1))
