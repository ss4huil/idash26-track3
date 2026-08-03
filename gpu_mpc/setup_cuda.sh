#!/usr/bin/env bash
# setup_cuda.sh — install the CUDA 12.1 toolkit to ~/.local/cuda-12.1 (no sudo).
#
# Run this once on the target machine (H100 SXM, SM 9.0a) before building.
# After running, nvcc will be at:
#   ~/.local/cuda-12.1/usr/local/cuda-12.1/bin/nvcc
# The Makefile resolves this path automatically (CUDA_VERSION=12.1 default).
#
# Usage:
#   bash setup_cuda.sh            # download + install
#   bash setup_cuda.sh --verify   # just verify an existing install
#
# Requirements: wget or curl, ~3 GB free in ~/.local
set -euo pipefail

CUDA_VER="12.1.0"
CUDA_BUILD="525.85.12"
CUDA_SHORT="12.1"
INSTALL_DIR="$HOME/.local/cuda-${CUDA_SHORT}"
RUNFILE="cuda_${CUDA_VER}_${CUDA_BUILD}_linux.run"
URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/local_installers/${RUNFILE}"

verify_install() {
    local nvcc="${INSTALL_DIR}/usr/local/cuda-${CUDA_SHORT}/bin/nvcc"
    if [[ -f "$nvcc" ]]; then
        echo "nvcc found: $nvcc"
        "$nvcc" --version
        echo "✓ CUDA ${CUDA_SHORT} toolkit is ready"
        return 0
    else
        echo "✗ nvcc not found at $nvcc"
        return 1
    fi
}

if [[ "${1:-}" == "--verify" ]]; then
    verify_install
    exit $?
fi

if verify_install 2>/dev/null; then
    echo "Already installed — nothing to do."
    exit 0
fi

echo "Downloading CUDA ${CUDA_VER} runfile (~3 GB)..."
mkdir -p "$INSTALL_DIR"
if command -v wget &>/dev/null; then
    wget -q --show-progress -O "${INSTALL_DIR}/${RUNFILE}" "$URL"
elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "${INSTALL_DIR}/${RUNFILE}" "$URL"
else
    echo "Error: wget or curl required" >&2; exit 1
fi

echo "Installing toolkit (no-root, toolkit only) to ${INSTALL_DIR}..."
sh "${INSTALL_DIR}/${RUNFILE}" \
    --silent \
    --toolkit \
    --installpath="${INSTALL_DIR}/usr/local/cuda-${CUDA_SHORT}" \
    --no-opengl-libs \
    --no-drm \
    --no-man-page \
    --override

rm -f "${INSTALL_DIR}/${RUNFILE}"
verify_install

echo ""
echo "Add to PATH for this session:"
echo "  export PATH=${INSTALL_DIR}/usr/local/cuda-${CUDA_SHORT}/bin:\$PATH"
echo ""
echo "Or build directly:"
echo "  make -C idash/mpc/gpu_mpc GPU_MPC_ROOT=\$HOME/EzPC/GPU-MPC \\"
echo "       GPU_ARCH=90a BW=32 deepdtagen_inference"
echo "  # For production 64-bit ring:"
echo "  make -C idash/mpc/gpu_mpc GPU_MPC_ROOT=\$HOME/EzPC/GPU-MPC \\"
echo "       GPU_ARCH=90a BW=64 deepdtagen_inference"
