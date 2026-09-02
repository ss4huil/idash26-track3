# Secure GPU-Accelerated 2PC Inference for DeepDTAGen — iDASH 2026 Track 3

**Team**: nameforgotten — NUDT
**Contact**: songshang19@nudt.edu.cn

---

## 1. Quick start — how to run the evaluation

The docker image ships everything pre-built (H100 / sm_90a binary, scripts, runtime).
The organizer only needs to:

```bash
# 1. On BOTH machines, edit the 4-line CONFIG block at the top of competition_run.sh:
#       PARTY=0|1            (which party this machine plays)
#       PEER_IP=...          (IP of the party-0 machine)
#       SAMPLEDIR=...        (secret-shared inputs, format in Section 5)
#       KEYDIR=...           (writable directory for FSS keys)
#       DDG_BW=32|64         (ring size: 32 for davis/bindingdb, 64 for kiba)

# 2. Offline phase (untimed; key generation, once per party):
./competition_run.sh dealer

# 3. Online phase (timed): start party 1 FIRST, then party 0:
./competition_run.sh eval

# 4. Party 0 prints one prediction per sample:  AFFINITY[i]=<value>
```

No other configuration is needed: the runner auto-detects GPU VRAM / system RAM,
chooses the micro-batch size B and chunk count NC, and switches between
resident mode (keys fit in memory) and SSD-streaming mode (large N, prefetch
pipeline) automatically. Details and expert overrides are in Section 6.

## 2. Overview

This submission implements a GPU-accelerated two-party secure inference system for the
**affinity prediction branch** of DeepDTAGen (GCN over the molecular graph + masked global
max-pooling + FC layers), using 2-party additive secret sharing over the ring Z_2^32 /
Z_2^64 in the preprocessing (trusted-dealer) model. The drug molecule (SMILES graph) is
the private input; the protein sequence and the model weights are public.

Two interchangeable backends for the non-linear (ReLU/compare) layer are provided:

- **FSS backend** (default, `master`): Orca/SIGMA-style DCF keys, evaluated on GPU.
- **LSS backend** (opt-in, `dev-v2`): dealer-generated secret-shared correlations
  (Matchmaker-style) with PRF seed compression, evaluated on CPU — shrinks the ReLU key
  material ~20× and end-to-end online time 3.85× in the SSD-streaming regime
  (Section 3.4).

Security: 128-bit computational security (AES-based DPF/DCF key generation), semi-honest
adversary, one corruption.

## 3. Method

### 3.1 Compliant online adjacency normalization (A_hat)

Per the competition rules, the degree matrix D and D^{-1/2} are derived **online** from the
secret shares of the raw adjacency matrix A — no precomputed normalized adjacency is given
to the online phase.

Key observation: molecular adjacency matrices are **binary** (bond / no-bond, plus
self-loops), so every row degree is a small integer in {0, ..., 138}. We therefore replace
iterative inverse-square-root computation with an **exact table lookup** performed on secret
shares:

1. **Degree vector** `d = (A+I)·1` — local row sums on shares (zero communication).
2. **D^{-1/2} via DPF-LUT**: a public constant table of 139 entries
   `table[k] = round(2^s / sqrt(k))` (s = 12 fractional bits; `table[0] = 0` handles padded
   atoms, matching the plaintext reference semantics) is evaluated on the secret degree
   vector using a DPF-based lookup over an 8-bit input domain. The table is a universal
   mathematical constant (input- and model-independent); the selection on secret data
   happens online.
3. **Two masked diagonal scalings** `A_hat = diag(r) · A · diag(r)` via Beaver
   multiplication — the first multiplication needs no truncation (A is 0/1), only the
   second uses one deterministic truncation. A_hat is computed once and reused by all
   three GCN layers.

This is **exact** (the table entries are the same fixed-point quantization the plaintext
pipeline would produce), costs ~2 communication rounds and <5% extra key material, and is
fully compliant with the rule that D and D^{-1/2} be derived online from shares.

### 3.2 GPU performance optimizations — Mode A: keys resident in memory

When the FSS keys for the batch fit in RAM + VRAM, end-to-end online latency is minimized
by eliminating host-side data movement and CPU/GPU serialization:

- **Key arena (bulk GPU-resident keys)**: instead of per-operator H2D copies of key
  material (which dominated runtime — ~78% at B=32), the entire key file is uploaded to
  the GPU **once**; every per-operator key access becomes a zero-copy pointer translation.
  Key buffers that the protocol writes in place are transparently cloned device-to-device.
- **Synchronization removal**: all kernels and copies run on the default stream, so
  ordering is already guaranteed by the runtime. We replaced 46 redundant
  `cudaDeviceSynchronize` calls (the median call blocked the CPU for 168 µs; ~542 per
  forward pass) with non-blocking error checks, recovering 23% of online time.
- **D2D input refresh**: secret input tensors are uploaded once (H2D) and refreshed per
  forward pass with device-to-device copies.

Measured on a single NVIDIA A10 (weaker than the evaluation H100), bw=32, scale=12,
batch=16: **136 ms per batch (8.5 ms/sample)**, down from 485 ms for the unoptimized
implementation (3.6×). Communication: 8.4 MB/sample.

### 3.3 GPU performance optimizations — Mode B: keys streamed from SSD

For evaluation-scale batches (up to hundreds of thousands of samples), FSS keys
(~165 MB/sample/party at bw=32) far exceed RAM/VRAM capacity, so keys must be streamed
from storage during the online phase. We built a **chunked streaming pipeline**:

- **Chunked key files**: the dealer generates keys chunk-by-chunk (bounded dealer memory)
  and appends them to one file per party; chunk k holds the keys for one micro-batch of
  B samples.
- **Three-stage double-buffered prefetch pipeline**:
  `SSD → pinned RAM slot → (async PCIe copy on a dedicated CUDA stream) → VRAM slot`.
  While the GPU computes chunk k, an I/O thread reads chunk k+1 from disk into a pinned
  staging buffer and copies it asynchronously into the alternate VRAM slot; CUDA events
  and atomic handshakes coordinate slot reuse. The host staging buffer doubles as the
  key-cursor base, so no extra copy exists anywhere on the path.
- **Arena slot remapping**: the zero-copy key translation layer is re-pointed to the
  active slot at each chunk boundary; the compute path is unchanged.
- **Tail handling**: when N is not a multiple of B, the final chunk is zero-padded
  (a zero adjacency has degree 0, which the LUT maps to 0 — protocol-safe), and padded
  outputs are discarded.

Steady-state per-chunk time is `max(compute, PCIe copy, SSD read)` instead of their sum;
prefetch fully hides key loading whenever storage is not the bottleneck (measured: stall
per chunk reduced from 443 ms to 3 µs when keys are on a fast device). Correctness was
verified bit-exact against single-chunk execution for NC=2/3/4 chunk counts and for
zero-padded tail chunks.

### 3.4 LSS comparison backend (v2, opt-in via `DDG_LSS_RELU=1`)

The DCF comparison keys of the FSS backend account for ~85% of total key material
(~167 of 198 MB/sample/party at bw=64), making large-batch runs SSD-bound. The LSS
backend replaces DCF-based ReLU/DReLU with **secret-shared correlations generated
directly by the dealer** (Matchmaker-style, ePrint 2025/424 — no two-party OT needed in
the dealer model), evaluated on CPU:

- **Protocol**: radix-16 Millionaires comparison (16 digits, 26 triples per 64-bit
  comparison, 6 rounds) built from four dealer-supplied primitives — AND triples,
  1-of-16 OT, MUX triples, and bit-to-arithmetic (B2A) correlations. DReLU/ReLU reduces
  to one comparison + one MUX. Bridge to the GPU pipeline via masked-public ↔ share
  conversion on an independent socket channel; OpenMP-parallel evaluation
  (`DDG_LSS_THREADS`).
- **LSS2 seed compression**: all pure-randomness components of the correlations are
  regenerated online from per-party, per-record-type **AES-128-CTR PRF streams** (14
  independent stream keys per party, counter mode with random access for the parallel
  evaluator); only the correlation *corrections* are stored explicitly
  (e.g. for an AND triple only `c0 = (a∧b)⊕c1` is stored — 1 bit). Record sizes
  (bits, incl. 3-bit tag, party0/party1): AND triple 6→4/3; OT16-send 35→3;
  OT16-recv 9→5; MUX triple 196→67/5; B2A 68→3/67; masks 67→3/3 or 3/67.
- **Result**: ReLU key material 35.8 → **9.8 MB/sample** (both parties combined;
  1046→225 bit/element for party 0); total per-sample key footprint
  **198 → 36 MB/party** (5.5×). AES-NI regeneration is essentially free — the ReLU
  stage got *faster* (1.04 → 0.76 s/forward at B=8) because memory traffic shrank.

Security: 128-bit computational (AES-128 in CTR mode as PRF; FIPS-197 known-answer
self-test at keygen time), semi-honest dealer + semi-honest parties. Full format and
proof sketch: `gpu_mpc/lss/lss_protocol.md`.

**N=128 head-to-head** (davis, bw=64, B=8×16 chunks, streaming pipeline, loopback,
single A10 shared by both parties):

| Metric | FSS backend | LSS2 backend | |
|---|---|---|---|
| Online wall time | 97.3 s | **25.3 s** | **3.85× faster** |
| per sample | 760 ms | 197 ms | |
| key total / party | 25.4 GB (198 MB/sample) | 4.6 GB (36 MB/sample) | 5.5× smaller |
| online comm / party | 10.2 MB/sample | 27.5 MB/sample | 2.7× more |
| keygen time / party | 52.3 s | 48.3 s | comparable |

The bottleneck analysis flips: FSS is key-I/O-bound (81 of 97 s waiting on SSD reads;
GPU 93% idle), while LSS2 fully hides I/O and is bound by the CPU ReLU evaluation
(OT16 leaf construction + AND-tree opens, 86% of compute). Outputs of the two backends
agree to max |Δ| = 0.0034 (fixed-point truncation noise), same MAE vs golden labels.
See `docs/RESEARCH_LOG.md` for the complete engineering log, per-stage breakdown,
pitfalls, and the prioritized list of remaining optimizations.

## 4. Repository layout and build

```
gpu_mpc/                    # MPC inference driver and model definition
  deepdtagen_inference.cu   # dealer + evaluator entry point (single binary, role switch)
  secure_adj_norm.h         # online adjacency normalization (Section 3.1)
  ddg_orca*.h               # FSS backend wrappers (DDGOrca eval / keygen classes)
  lss/                      # LSS backend (Section 3.4): keygen, online primitives,
                            #   compare/ReLU, GPU bridge, lss_protocol.md, unit tests
reference/                  # plaintext reference model, weight export, share preparation
scripts/dev_tools/          # dataset/share preparation and validation utilities
docs/RESEARCH_LOG.md        # full research log: measurements, pitfalls, future work
```

External dependency: the GPU-MPC framework (EzPC/GPU-MPC), included/patched alongside.

Build (requires CUDA; use `GPU_ARCH=90a` for H100). **Two binaries are needed** — one
per ring size (`run_party.sh` selects `deepdtagen_inference_bw$B` automatically):

```bash
cd gpu_mpc
# davis / bindingdb: 32-bit ring
make GPU_MPC_ROOT=<path-to-GPU-MPC> BW=32 GPU_ARCH=90a CUDA_VERSION=12.8 deepdtagen_inference
mv deepdtagen_inference deepdtagen_inference_bw32
# kiba: 64-bit ring (its fusion-layer activations overflow 32 bits; rules allow both)
rm -f pool_override.o
make GPU_MPC_ROOT=<path-to-GPU-MPC> BW=64 GPU_ARCH=90a CUDA_VERSION=12.8 deepdtagen_inference
mv deepdtagen_inference deepdtagen_inference_bw64
```

Pre-built H100 (sm_90a) binaries are included as
`gpu_mpc/deepdtagen_inference_h100_bw{32,64}` — on the evaluation machines simply copy
them over the `deepdtagen_inference_bw{32,64}` names.

## 5. Input data format

All inputs live in one sample directory (per dataset). Fixed-point encoding: ring Z_2^32,
scale s=12 (value v stored as `round(v · 2^12) mod 2^32`, little-endian u32, row-major,
sample-major for batched files).

| File | Content | Shape | Encoding |
|---|---|---|---|
| `x_share{0,1}.dat` | additive shares of atom features X | (N, 138, 94) | fixed-point Q12 |
| `adj_share{0,1}.dat` | additive shares of **raw 0/1 adjacency A+I** | (N, 138, 138) | ring values {0,1}, scale 0 |
| `mask_share{0,1}.dat` | additive shares of tiled node mask | (N, 138, 376) | fixed-point Q12 |
| `protein_emb.dat` | public protein embedding (party 1 only) | (N, 128) | fixed-point Q12 |
| `weights.bin` | exported model weights (public) | flat int64 blob | little-endian **int64** fixed-point Q12; per-layer shapes/offsets in the `weights.bin.json` sidecar; reduced to the 32-bit ring at load time |
| `batch_manifest.json` | metadata (shapes, scale, row indices) | — | JSON (the adjacency entry is keyed `A_hat` for legacy reasons; with `raw_adj=true` it holds raw A) |

`N` is the total number of samples; molecules are zero-padded to `nmax=138` atoms.
Share files contain all N samples; chunked execution reads per-chunk slices by offset.
The secret splitting itself (generation of the `*_share{0,1}.dat` files) is offline
preprocessing and is produced by `scripts/dev_tools/prepare_batch_samples.py`
(`raw_adj=True`).

## 6. Running the system (details and expert overrides)

The same binary acts as dealer (`role=0`) or online evaluator (`role=1`):

```
deepdtagen_inference <bw> <scale> <role> <party> <keydir> <sampledir> <B> [peer_ip]
```

**Offline (dealer, untimed)** — run once per party per chunk set:

```bash
DDG_SECURE_ADJ_NORM=1 DDG_KEYBUF_CAP_GB=6 DDG_NUM_CHUNKS=<NC> \
DDG_WEIGHTS_BIN=<sampledir>/weights.bin \
./deepdtagen_inference 32 12 0 <party> <keydir>/ <sampledir> <B>
```

**Online (evaluator, timed)** — two processes, one per party (party 1 starts first):

```bash
# Mode A: keys resident in RAM/VRAM (small N)
DDG_SECURE_ADJ_NORM=1 DDG_KEY_ARENA_PCT=95 DDG_WEIGHTS_BIN=<sampledir>/weights.bin \
./deepdtagen_inference 32 12 1 <party> <keydir>/ <sampledir> <B> <peer_ip>

# Mode B: streaming from SSD (large N); NC = ceil(N / B)
DDG_SECURE_ADJ_NORM=1 DDG_KEY_ARENA_PCT=95 DDG_NUM_CHUNKS=<NC> DDG_PREFETCH=1 \
DDG_WEIGHTS_BIN=<sampledir>/weights.bin \
./deepdtagen_inference 32 12 1 <party> <keydir>/ <sampledir> <B> <peer_ip>
```

Environment variables:

| Variable | Meaning |
|---|---|
| `DDG_SECURE_ADJ_NORM=1` | enable compliant online adjacency normalization (Section 3.1) |
| `DDG_NUM_CHUNKS` | number of key chunks (>1 enables streaming Mode B) |
| `DDG_PREFETCH=0/1` | prefetch pipeline on/off (default on in streaming mode) |
| `DDG_KEY_ARENA_PCT` | VRAM budget (%) for the resident key arena |
| `DDG_KEYBUF_CAP_GB` | dealer-side key buffer cap (GB) |
| `DDG_WEIGHTS_BIN` | path to exported weights |
| `DDG_BW` | ring size at dealer time (32 default; **64 required for kiba** — its fusion-layer activations overflow the 32-bit ring; eval reads `bw` from `meta.json` automatically). Note: at bw=64 the zero-copy key arena is auto-disabled (key-layout alignment), falling back to the plain copy path |
| `DDG_LSS_RELU=1` | route ReLU/DReLU through the LSS backend (Section 3.4) instead of FSS; requires `DDG_KEY_ARENA=0` at bw=64 |
| `DDG_LSS_THREADS` | OpenMP threads for the LSS evaluator (set to the number of physical cores, e.g. 8) |
| `DDG_EXACT_TRUNC=1` | fall back from LocalARS probabilistic truncation to exact truncation (default is LocalARS, which eliminates truncation keys: −47% key size) |

**Output**: party 0 prints one line per sample: `AFFINITY[i]=<float>` — the predicted
binding affinity (regression value, Q12 fixed-point decoded).

## 7. Measured performance (NVIDIA A10, scale=12, loopback)

**FSS backend, bw=32:**

| Setting | Time |
|---|---|
| Batch 16, keys resident, optimized | 136 ms/batch (8.5 ms/sample) |
| Batch 16, keys resident, unoptimized baseline | 485 ms/batch |
| Batch 16, simulated 1 Gbps link (shared) | 1071 ms/batch |
| Batch 16, simulated 1 Gbps full-duplex | 610 ms/batch |
| Streaming mode, prefetch on | per-chunk key-load stall fully hidden behind compute |

Key size: ~165 MB/sample/party (bw=32); online communication: ~8.4 MB/sample.

**N=128 streaming benchmark, bw=64, FSS vs LSS2** (see Section 2.4 for the full table):
FSS 97.3 s vs LSS2 **25.3 s** online (3.85×); key footprint 25.4 GB vs 4.6 GB per party.

## 8. Development status and roadmap

Completed and verified: compliant online adjacency normalization; both backends
(FSS and LSS2) end-to-end on all three datasets (davis / kiba / bindingdb) with
fixed-point outputs matching the official floating-point reference to ≤0.005;
multi-chunk SSD streaming with prefetch; AES-CTR seed compression; parallel LSS
evaluation.

Known gaps / next steps (tracked in `docs/RESEARCH_LOG.md`):

- **LSS key SSD streaming**: the `_lss.bin` key file is currently loaded into RAM once
  at chunk 0 — fine up to ~50K samples/party, but needs chunked streaming beyond that.
- **Turnkey integration**: `run_party.sh`'s automatic batch-size picker still uses
  FSS-era per-sample key estimates; the LSS mode switch is not yet exposed in
  `competition_run.sh`.
- **Competition accuracy gate**: outputs were validated as regression values against
  the floating-point reference; the official metric (mean of sensitivity and
  specificity at several affinity thresholds) has not yet been evaluated on the full
  test sets.
- **Network-throttled benchmark**: the 3.85× figure is loopback; LSS trades 2.7× more
  communication and more rounds, so a re-measurement under a rate-limited link
  (>1 Gbps, RTT <1 ms) is needed before choosing which backend(s) to submit.
- **Remaining key compression**: the non-ReLU FSS keys (~31 MB/sample/party) can
  undergo the same seed-compression treatment (Beaver triples: regenerate a,b, store
  only c), and the conv/GEMM key material has not yet been audited for redundancy
  (public-weight × secret-input products need no triples at all).
- H100 (sm_90a) binaries are compile-verified but have not been run on real H100
  hardware.
