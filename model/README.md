# Pretrained Model Weights

This directory contains the pretrained DeepDTAGen models for drug-target affinity prediction.

## Model Files

The following PyTorch checkpoint files are required:

- `deepdtagen_model_davis.pth` (133 MB)
- `deepdtagen_model_kiba.pth` (133 MB)
- `deepdtagen_model_bindingdb.pth` (133 MB)

**Note**: These files are **not tracked in git** due to their size (~400MB total).

## Download Instructions

### Option 1: From Official Source

Download from the official iDASH Track 3 challenge or DeepDTAGen repository:

```bash
cd idash/mpc/model/

# Download from official source (replace URL with actual source)
# wget https://example.com/deepdtagen_model_davis.pth
# wget https://example.com/deepdtagen_model_kiba.pth
# wget https://example.com/deepdtagen_model_bindingdb.pth
```

### Option 2: Use Existing Models

If you already have the models from a previous installation:

```bash
cp /path/to/existing/deepdtagen_model_*.pth idash/mpc/model/
```

## Model Details

- **Architecture**: DeepDTAGen (GCN + GatedCNN + Fusion FC layers)
- **Format**: PyTorch state dict (OrderedDict)
- **Precision**: float32
- **Framework**: PyTorch 1.12+

## Datasets

- **Davis**: Kinase inhibitor dataset (~5,000 test samples)
- **KIBA**: Kinase inhibitor bioactivities (~20,000 test samples)  
- **BindingDB**: Broader binding database

## Converting to Fixed-Point Weights

To use these models in the MPC framework, they must be converted to fixed-point format:

```bash
cd idash/mpc

# Convert Davis model to Q20.12 fixed-point binary
python3 reference/export_weights.py \
    model/deepdtagen_model_davis.pth \
    gpu_mpc/weights.bin \
    --scale 12 --bw 32

# For KIBA or BindingDB, replace the input .pth file
```

The converted `weights.bin` file is used by the GPU-MPC inference binary.

## Verification

To verify the models are loaded correctly:

```bash
cd idash/mpc

# Test plaintext model loading
python3 -c "
from reference.affinity_model import AffinityModel
model = AffinityModel.from_pth('model/deepdtagen_model_davis.pth')
print('Model loaded successfully')
"
```

## License

These pretrained models are from the original DeepDTAGen work. Please cite the appropriate paper if used in research.
