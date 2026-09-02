> **iDASH 2026 Track 3 submission version**: see **[SUBMISSION_README.md](SUBMISSION_README.md)** for
> how to run the evaluation (`competition_run.sh`, zero-config), the method description, and the
> input data format. Framework patches: `patches/EzPC-GPU-MPC.track3.patch`.
> Pre-built H100 binaries: `gpu_mpc/deepdtagen_inference_h100_bw{32,64}`. Tag: `track3-v1-submission`.

# GPU-Accelerated 2PC for DeepDTAGen

A GPU-accelerated two-party computation (2PC) implementation of DeepDTAGen, a graph neural network for drug-target affinity prediction. This project uses the EzPC/GPU-MPC framework to perform secure inference over encrypted drug-protein data.

## Overview

**DeepDTAGen** predicts binding affinity between small molecule drugs (SMILES) and protein targets (amino acid sequences) using:
- Graph Convolutional Networks (GCN) for drug molecular graphs
- Gated CNN for protein sequence embeddings  
- Multi-layer fusion network for affinity prediction

This implementation enables **privacy-preserving inference** where:
- Party 0 holds the drug structure (secret-shared)
- Party 1 holds the protein sequence (secret-shared)
- Neither party learns the other's input
- Both parties learn only the predicted affinity score

### Key Features

- **GPU-accelerated MPC primitives**: ReLU, truncation, sign extension via Distributed Point Functions (DPF)
- **Fixed-point arithmetic**: Q20.12 format (32-bit ring, 12 fractional bits)
- **Deterministic execution**: Sigma-native truncation (TrFloor) for reproducible results
- **Single-machine 2PC testing**: Automated dealer + online protocol workflow
- **Reference implementations**: Python plaintext and fixed-point baselines for validation

## Project Structure

```
idash/mpc/
├── README.md                   # This file
├── gpu_mpc/                    # GPU MPC core implementation
│   ├── deepdtagen_inference.cu # Main binary (dealer + 2PC evaluator)
│   ├── deepdtagen.h            # Model architecture definition
│   ├── ddg_orca*.h             # MPC backend (Sigma-native primitives)
│   ├── gcn_layer.h             # Graph convolution layer
│   ├── masked_maxpool.h        # Masked global max pooling
│   ├── Makefile                # Build system
│   ├── run_local_2pc.sh        # Single-machine 2PC test script
│   ├── sytorch/                # Sytorch framework (Concat fix)
│   ├── utils/                  # Communication utilities
│   └── *.dat, weights.bin      # Generated offline artifacts (gitignored)
├── reference/                  # Python reference implementations
│   ├── affinity_model.py       # Plaintext PyTorch model
│   ├── fixed_forward.py        # Fixed-point forward pass
│   ├── dense_graph.py          # SMILES → dense graph conversion
│   ├── export_weights.py       # PyTorch → fixed-point weights
│   ├── offline_prepare.py      # Secret sharing generation
│   └── share_data.py           # Share serialization utilities
├── baseline/                   # Official plaintext baseline results
│   ├── official_baseline_davis.json
│   └── official_baseline_kiba.json
├── tests/                      # Davis/Kiba MPC accuracy-gate tests (pytest)
├── offline/                    # Offline data-prep tests (secret sharing, weight export)
├── microbench/                 # Component unit tests (GCN, fixed-point, metrics, ...)
├── model/                      # Pretrained weights (*.pth, ~400MB, git-ignored)
├── data/                       # Test datasets (davis_test.csv, kiba_train.csv)
├── docs/                       # Documentation (Chinese guides)
├── scripts/                    # Development utilities
├── real_gpu_2pc_benchmark.py   # Real 2PC benchmark script (timing + accuracy)
└── run_davis_multibatch.sh     # Davis batched MPC evaluation
```

## Quick Start

### Prerequisites

- **Hardware**: NVIDIA GPU with Compute Capability ≥ 7.5, ≥4GB VRAM
- **Software**: 
  - CUDA Toolkit 12.1+
  - GCC 9.x - 11.x
  - Python 3.8+
  - [EzPC/GPU-MPC](https://github.com/mpc-msri/EzPC) framework

### Installation

1. **Clone EzPC/GPU-MPC**:
```bash
cd ~
git clone https://github.com/mpc-msri/EzPC.git
export GPU_MPC_ROOT=$HOME/EzPC/GPU-MPC
```

2. **Install Python dependencies**:
```bash
pip install numpy scipy pandas torch torch-geometric rdkit-pypi
```

3. **Build the 2PC binary**:
```bash
cd idash/mpc/gpu_mpc
export PATH=/usr/local/cuda-12.1/bin:$PATH

# BW=32 for Q20.12, GPU_ARCH=89 for RTX 4060 (adjust for your GPU)
make GPU_MPC_ROOT=$GPU_MPC_ROOT BW=32 GPU_ARCH=89 deepdtagen_inference
```

4. **Download pretrained weights**:
```bash
# Place DeepDTAGen model files in idash/mpc/model/
# - deepdtagen_model_davis.pth
# - deepdtagen_model_kiba.pth
```

5. **Generate offline artifacts** (required before first run):
```bash
cd idash/mpc

# Prepare a test sample (generates weights.bin + secret shares)
python3 -c "
from reference.offline_prepare import prepare_sample
result = prepare_sample(
    dataset='davis',
    csv_path='data/davis_test.csv',
    row_idx=0,
    out_dir='gpu_mpc/davis_sample',
    scale=12,
    bw=32
)
print(f'Sample prepared: {result[\"sample_dir\"]}')
print(f'Weights: {result[\"weights_path\"]}')
"
# Generates:
#   gpu_mpc/davis_sample/sample_0/*.dat (secret shares)
#   gpu_mpc/davis_sample/weights.bin (~13MB)
```

### Running 2PC Inference

**Single sample test**:
```bash
cd idash/mpc/gpu_mpc
# Use the sample prepared above
./run_local_2pc.sh davis_sample/sample_0 /tmp/keys_test davis_sample/weights.bin
```

Expected output:
```
[dealer] keys written to /tmp/keys_test/DeepDTAGen_32_12
[online] Average time taken (microseconds)=...
[online] Comm (B)=...
AFFINITY=5.001234  ← Predicted binding affinity
```

**Benchmark test** (10 samples per dataset with timing + regression metrics):
```bash
cd idash/mpc
python3 real_gpu_2pc_benchmark.py
```

**Batched evaluation** (multiple batches of B=4 samples):
```bash
cd idash/mpc
./run_davis_multibatch.sh 5 4  # 5 batches × 4 samples = 20 total
```

### Preparing Custom Samples

```python
from reference.offline_prepare import prepare_sample

# Example: Erlotinib + EGFR kinase domain
result = prepare_sample(
    dataset="davis",
    csv_path="/path/to/davis_test.csv",
    row_idx=0,
    out_dir="/tmp/my_sample",
    scale=12,
    bw=32
)

# Generates secret shares: x_share{0,1}.dat, adj_share{0,1}.dat, 
# mask_share{0,1}.dat, protein_emb.dat, weights.bin
```

Then run 2PC:
```bash
cd gpu_mpc
./run_local_2pc.sh /tmp/my_sample /tmp/keys weights.bin
```

## Architecture

### Model Pipeline

```
Drug Path:
  SMILES → Dense Graph (X, A_hat, mask)
  → GCN → GCN → GCN → MaskedMaxPool → FC(1024) → ReLU → FC(128)

Protein Path:
  Sequence → GatedCNN → FC(128)  [precomputed, loaded as embedding]

Fusion:
  Concat(drug_emb, protein_emb) → FC(1024) → ReLU 
  → FC(512) → ReLU → FC(256) → ReLU → FC(1)
```

### MPC Protocol

- **Offline phase** (Dealer): Generate Function Secret Sharing (FSS) keys for DPF operations
- **Online phase** (2PC): 
  - Party 0 & 1 load secret shares of inputs
  - Execute GCN → MaxPool → FC layers via additive secret sharing
  - Non-linear operations (ReLU, MaxPool) use DPF/DCF comparisons
  - Reveal final affinity score

### Fixed-Point Arithmetic

- **Format**: Q20.12 (20 integer bits, 12 fractional bits, 32-bit ring)
- **Truncation**: TrFloor (Sigma-native, deterministic)
- **Operations**: 
  - Linear: exact in secret sharing
  - ReLU: 2-round DPF protocol
  - Mul: Beaver triples + truncation

## Performance

Tested on RTX 4060 (8GB VRAM), localhost:

| Phase           | Time    | Communication |
|-----------------|---------|---------------|
| Dealer (offline)| ~15s    | 0             |
| Online 2PC      | ~30-60s | ~500 MB       |

**Accuracy**: MPC predictions match plaintext baseline within ±0.1 (MSE < 0.01)

## Testing

```bash
cd idash/mpc

# Davis/Kiba MPC accuracy-gate tests
pytest tests/

# Offline data-preparation tests (secret sharing, weight/NPZ export)
pytest offline/

# Component unit tests (GCN, fixed-point, pooling, metrics, ...)
pytest microbench/

# Real 2PC benchmark (timing + regression metrics)
python3 real_gpu_2pc_benchmark.py

# Batched Davis MPC evaluation
./run_davis_multibatch.sh 5 4
```

## Documentation

- [docs/SETUP_GUIDE_CN.md](docs/SETUP_GUIDE_CN.md) - Detailed setup guide (Chinese)
- [docs/FILE_GUIDE_CN.md](docs/FILE_GUIDE_CN.md) - File-by-file documentation (Chinese)
- [gpu_mpc/GPU_MPC_FRAMEWORK_GUIDE.md](gpu_mpc/GPU_MPC_FRAMEWORK_GUIDE.md) - Framework usage (Chinese)

## Technical Details

### Key Fixes & Optimizations

1. **Sigma-native primitives** (2024-08): Replaced custom random truncation with deterministic TrFloor
2. **View aliasing fix** (2024-08): Fixed MaxPool CUDA memory double-free crash
3. **Mask consistency** (2024-08): Both dealers load share0 for consistent FSS keys
4. **Pure ring subtraction** (2024-08): Masked MaxPool uses ring-only ops (no truncation)

### Known Limitations

- MaxPool uses DCF (not DPF-optimized)
- Single-sample only (no batching)
- Tested only on localhost (network 2PC not benchmarked)

## Citation

This implementation builds on:

```bibtex
@inproceedings{ezpc-gpu-mpc-2024,
  title={EzPC: Programmable and Efficient Secure Two-Party Computation},
  author={Jawalkar, Neha and others},
  booktitle={IEEE S\&P},
  year={2024}
}
```

## License

- **EzPC/GPU-MPC**: MIT License (Microsoft Research)
- **This implementation**: Academic/research use

## Contact

For issues or questions, please open an issue on GitHub.
