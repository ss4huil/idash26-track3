#!/usr/bin/env bash
# =====================================================================
# iDASH 2026 Track 3 — turnkey runner.
#
# HOW TO USE (only 4 lines to edit):
#   1. Edit the CONFIG block below on BOTH machines.
#   2. Offline (untimed, once, on either machine or both):
#         ./competition_run.sh dealer
#   3. Online (timed): on machine 1 (party 1) run FIRST:
#         ./competition_run.sh eval
#      then on machine 0 (party 0):
#         ./competition_run.sh eval
#   4. Read predictions on party 0's console: AFFINITY[i]=<value>
#
# Everything else (batch size B, chunk count NC, resident vs SSD
# streaming mode, GPU memory budget, prefetch pipeline) is auto-detected.
# =====================================================================

# ----------------------- CONFIG (edit me) ----------------------------
PARTY=0                       # 0 or 1 — which party THIS machine plays
PEER_IP=127.0.0.1             # IP address of the PARTY-0 machine
SAMPLEDIR=/data/samples       # secret-shared inputs (see README §5)
KEYDIR=/data/fss_keys         # writable dir for generated FSS keys
export DDG_BW=32              # ring size: 32 for davis/bindingdb, 64 for kiba
# ---------------------------------------------------------------------

set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)

ROLE=${1:-}
case "$ROLE" in
    dealer) exec "$HERE/run_party.sh" dealer "$PARTY" "$SAMPLEDIR" "$KEYDIR" ;;
    eval)   exec "$HERE/run_party.sh" eval   "$PARTY" "$SAMPLEDIR" "$KEYDIR" "$PEER_IP" ;;
    *) echo "usage: $0 dealer|eval   (edit the CONFIG block at the top of this file first)"; exit 1 ;;
esac
