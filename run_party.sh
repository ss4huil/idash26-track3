#!/usr/bin/env bash
# run_party.sh — iDASH 2026 Track 3 submission entry point.
#
# One command per role; all performance parameters (micro-batch size B, chunk
# count NC, key arena budget, prefetch) are auto-detected from the machine and
# the key directory. Expert overrides remain available via env vars (see below).
#
# Usage:
#   ./run_party.sh dealer <party 0|1> <sampledir> <keydir>
#   ./run_party.sh eval   <party 0|1> <sampledir> <keydir> <peer_ip>
#
# The dealer additionally writes <keydir>/meta.json (B, NC); the eval side
# reads it, so eval needs NO parameter knowledge at all.
#
# Expert overrides (optional): DDG_FORCE_B, DDG_PREFETCH=0, DDG_KEY_ARENA_PCT,
# DDG_KEYBUF_CAP_GB.  DDG_SECURE_ADJ_NORM is always on (compliance path).
set -euo pipefail

ROOT=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
# Two binaries are shipped: deepdtagen_inference_bw32 (davis/bindingdb) and
# deepdtagen_inference_bw64 (kiba). BW is resolved below before the binary check.
BIN=""

# CUDA runtime libs (in the evaluation image these are already on the path;
# the guard keeps the script harmless both inside and outside docker).
for d in /usr/local/cuda-12.8/lib64 /usr/local/cuda/lib64; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}" && break
done

MODE=${1:-}; PARTY=${2:-}; SAMPLEDIR=${3:-}; KEYDIR=${4:-}; PEER_IP=${5:-127.0.0.1}
case "$MODE" in dealer|eval) ;; *) echo "usage: $0 dealer|eval <party> <sampledir> <keydir> [peer_ip]"; exit 1;; esac
[ -d "$SAMPLEDIR" ] || { echo "ERROR: sampledir not found: $SAMPLEDIR"; exit 1; }
mkdir -p "$KEYDIR"

SCALE=12
# Ring size: kiba's fusion-layer activations overflow the 32-bit ring — use
# bw=64 for kiba (allowed by the rules). Dealer picks it (env DDG_BW), records
# it in meta.json; eval reads it back so both sides always agree.
if [ "$MODE" = dealer ]; then
    BW=${DDG_BW:-32}
else
    BW=$(grep -o '"bw": *[0-9]*' "$KEYDIR/meta.json" 2>/dev/null | grep -o '[0-9]*' || true)
    [ -n "$BW" ] || BW=32
fi
KEYS_MB_PER_SAMPLE=$(( 180 * BW / 32 ))  # ~165 MB/sample at bw=32, ~330 at bw=64, +margin
HEADROOM_MB=8192                # VRAM headroom for activations/weights

BIN=$ROOT/gpu_mpc/deepdtagen_inference_bw$BW
[ -x "$BIN" ] || { echo "ERROR: binary not found: $BIN (build BW=$BW first, see README §4)"; exit 1; }

# ---- hardware auto-detection ---------------------------------------------
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
RAM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
[ -n "$VRAM_MB" ] && [ "$VRAM_MB" -gt 0 ] || { echo "ERROR: no NVIDIA GPU detected"; exit 1; }

pick_B() {  # largest B in {128,64,32,16,8} s.t. 2 VRAM slots + headroom fit
    local maxB=$(( (VRAM_MB - HEADROOM_MB) / (2 * KEYS_MB_PER_SAMPLE) ))
    for b in 128 64 32 16 8; do [ "$maxB" -ge "$b" ] && { echo "$b"; return; }; done
    echo 8
}

# ---- dealer: generate FSS keys chunk-by-chunk (offline, untimed) ----------
if [ "$MODE" = dealer ]; then
    N=$(grep -o '"batch_size": *[0-9]*' "$SAMPLEDIR/batch_manifest.json" | grep -o '[0-9]*')
    [ -n "$N" ] || { echo "ERROR: cannot read batch_size from $SAMPLEDIR/batch_manifest.json"; exit 1; }
    B=${DDG_FORCE_B:-$(pick_B)}
    [ "$B" -gt "$N" ] && B=$N   # never batch beyond the actual sample count
    NC=$(( (N + B - 1) / B ))
    echo "[run_party] dealer party$PARTY: N=$N, auto B=$B (VRAM ${VRAM_MB}MiB), NC=$NC, RAM ${RAM_MB}MiB"
    DDG_SECURE_ADJ_NORM=1 DDG_NUM_CHUNKS=$NC DDG_KEYBUF_CAP_GB=${DDG_KEYBUF_CAP_GB:-8} \
        DDG_WEIGHTS_BIN="$SAMPLEDIR/weights.bin" \
        "$BIN" $BW $SCALE 0 $PARTY "$KEYDIR/" "$SAMPLEDIR" $B
    printf '{\n  "B": %d,\n  "NC": %d,\n  "N": %d,\n  "bw": %d,\n  "scale": %d\n}\n' \
        "$B" "$NC" "$N" "$BW" "$SCALE" > "$KEYDIR/meta.json"
    echo "[run_party] dealer done; wrote $KEYDIR/meta.json"
    exit 0
fi

# ---- eval: online 2PC inference (timed) -----------------------------------
[ -f "$KEYDIR/meta.json" ] || { echo "ERROR: $KEYDIR/meta.json missing — run dealer first"; exit 1; }
B=${DDG_FORCE_B:-$(grep -o '"B": *[0-9]*' "$KEYDIR/meta.json" | grep -o '[0-9]*')}
NC=$(grep -o '"NC": *[0-9]*' "$KEYDIR/meta.json" | grep -o '[0-9]*')
[ -n "$B" ] && [ -n "$NC" ] || { echo "ERROR: malformed meta.json"; exit 1; }

# bw=64 key streams contain 8-byte elements at 4-byte-aligned file offsets;
# the zero-copy arena preserves those offsets on the device side and CUDA
# requires 8B alignment for u64 access. Until the key layout is padded for
# bw=64, fall back to the plain (slightly slower) copy path at bw=64.
ARENA_ENV="DDG_KEY_ARENA_PCT=${DDG_KEY_ARENA_PCT:-95}"
if [ "$BW" -eq 64 ]; then ARENA_ENV="DDG_KEY_ARENA=0"; fi

if [ "$NC" -gt 1 ]; then
    # Streaming mode: keys exceed memory, pipeline SSD -> pinned RAM -> VRAM.
    PREFETCH=${DDG_PREFETCH:-1}
    echo "[run_party] eval party$PARTY: STREAMING mode, B=$B NC=$NC prefetch=$PREFETCH (VRAM ${VRAM_MB}MiB)"
    env DDG_SECURE_ADJ_NORM=1 DDG_NUM_CHUNKS=$NC DDG_PREFETCH=$PREFETCH \
        $ARENA_ENV DDG_WEIGHTS_BIN="$SAMPLEDIR/weights.bin" \
        "$BIN" $BW $SCALE 1 $PARTY "$KEYDIR/" "$SAMPLEDIR" $B "$PEER_IP"
else
    # Resident mode: whole key file fits — bulk GPU-resident arena.
    echo "[run_party] eval party$PARTY: RESIDENT mode, B=$B (VRAM ${VRAM_MB}MiB)"
    env DDG_SECURE_ADJ_NORM=1 $ARENA_ENV \
        DDG_WEIGHTS_BIN="$SAMPLEDIR/weights.bin" \
        "$BIN" $BW $SCALE 1 $PARTY "$KEYDIR/" "$SAMPLEDIR" $B "$PEER_IP"
fi
