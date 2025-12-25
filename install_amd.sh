#!/bin/bash
# TRELLIS-AMD Installation Script
# Tested on: AMD RX 7800 XT, ROCm 6.4.2, Ubuntu

set -e

echo "=============================================="
echo "  TRELLIS-AMD Installation Script"
echo "  For AMD GPUs with ROCm"
echo "=============================================="

# Check for ROCm
if ! command -v rocm-smi &> /dev/null; then
    echo "ERROR: ROCm not found. Please install ROCm 6.4+ first."
    exit 1
fi

echo "[1/6] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/6] Installing PyTorch for ROCm..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

echo "[3/6] Installing TRELLIS dependencies..."
pip install -r requirements.txt

echo "[4/6] Installing nvdiffrast-hip (AMD-optimized)..."
cd extensions/nvdiffrast-hip
pip install . --no-build-isolation
cd ../..

echo "[5/6] Installing diff-gaussian-rasterization (AMD-optimized)..."
cd extensions/diff-gaussian-rasterization
pip install . --no-build-isolation
cd ../..

echo "[6/6] Installing torchsparse..."
pip install torchsparse

echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
echo "To run TRELLIS:"
echo "  source .venv/bin/activate"
echo "  ATTN_BACKEND=sdpa XFORMERS_DISABLED=1 SPARSE_BACKEND=torchsparse python app.py"
echo ""
echo "Then open http://localhost:7860 in your browser"
