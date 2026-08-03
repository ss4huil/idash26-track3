#!/usr/bin/env bash
#
# Install CUDA 12.1 toolkit on Ubuntu 20.04 WSL2 (toolkit ONLY — no driver).
#
# WSL2 gets its GPU driver from Windows via the passed-through libcuda.so in
# /usr/lib/wsl/lib. We must NOT install cuda-drivers here or it clobbers that.
# We use NVIDIA's dedicated wsl-ubuntu repo which excludes the driver package.
#
# Target GPU: RTX 4060 Laptop = compute capability sm_89 (needs CUDA >= 11.8).
# CUDA 12.1 matches the path baked into tests/test_nvcc.py.
#
set -euo pipefail

CUDA_VER_DASH=12-1
CUDA_VER_DOT=12.1
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
TMP_DEB=/tmp/cuda-keyring_1.1-1_all.deb

echo "[1/5] downloading cuda-keyring ..."
wget -q -O "$TMP_DEB" "$KEYRING_URL"

echo "[2/5] installing keyring (sudo — will prompt for password) ..."
sudo dpkg -i "$TMP_DEB"

echo "[3/5] apt update ..."
sudo apt-get update -y

echo "[4/5] installing cuda-toolkit-${CUDA_VER_DASH} (NO driver) ..."
sudo apt-get install -y "cuda-toolkit-${CUDA_VER_DASH}"

echo "[5/5] verifying nvcc ..."
export PATH="/usr/local/cuda-${CUDA_VER_DOT}/bin:$PATH"
nvcc --version

echo ""
echo "DONE. Add to your shell rc if not already present:"
echo "  export PATH=/usr/local/cuda-${CUDA_VER_DOT}/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VER_DOT}/lib64:\$LD_LIBRARY_PATH"
