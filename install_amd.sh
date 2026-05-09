#!/bin/bash
# TRELLIS-AMD Installation Script
# One-click installer for AMD GPUs with ROCm
# Tested on: AMD RX 7800 XT, ROCm 6.4.2, Ubuntu

set -e

echo "=============================================="
echo "  TRELLIS-AMD Installation Script"
echo "  For AMD GPUs with ROCm"
echo "=============================================="

# Check for ROCm
if ! command -v rocminfo &> /dev/null; then
    echo "ERROR: ROCm not found. Please install ROCm 6.4+ first."
    echo "See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    exit 1
fi

# Detect GPU
GPU_ARCH=$(rocminfo | grep -o 'gfx[0-9a-z]*' | head -1)
if [ -z "$GPU_ARCH" ]; then
    echo "WARNING: Could not detect GPU architecture, defaulting to gfx1100"
    GPU_ARCH="gfx1100"
fi
echo "Detected GPU: $GPU_ARCH"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "[1/8] Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo ""
echo "[2/8] Upgrading pip..."
pip install --upgrade pip wheel setuptools

echo ""
echo "[3/8] Installing PyTorch for ROCm..."
# Check if torch is already installed with ROCm
if python3 -c "import torch; exit(0 if hasattr(torch.version, 'hip') else 1)" 2>/dev/null; then
    echo "PyTorch for ROCm already installed"
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
fi

echo ""
echo "[4/8] Installing TRELLIS Python dependencies..."
pip install -r requirements.txt

echo ""
echo "[5/8] Installing nvdiffrast-hip..."
cd extensions/nvdiffrast-hip
pip install . --no-build-isolation
cd ../..

echo ""
echo "[6/8] Building diff-gaussian-rasterization (manual HIP build)..."
cd extensions/diff-gaussian-rasterization
chmod +x build_hip.sh
./build_hip.sh
cd ../..

echo ""
echo "[7/8] Installing torchsparse with GPU support..."
cd extensions/torchsparse
rm -rf build *.egg-info 2>/dev/null || true
# FORCE_CUDA=1 is required to build the HIP/GPU backend
PYTORCH_ROCM_ARCH=gfx1100 ROCM_HOME=/opt/rocm FORCE_CUDA=1 pip install . --no-build-isolation
cd ../..

echo ""
echo "[8/8] Patching gradio_client for compatibility..."
# Fix gradio_litmodel3d schema compatibility issue
UTILS_FILE=".venv/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/gradio_client/utils.py"
if [ -f "$UTILS_FILE" ]; then
    # Patch get_type function to handle boolean schemas
    sed -i 's/def get_type(schema: dict):/def get_type(schema: dict):\n    # Handle non-dict schemas (e.g., boolean from additionalProperties: true)\n    if not isinstance(schema, dict):\n        return "Any"/' "$UTILS_FILE"
    # Patch _json_schema_to_python_type function
    sed -i 's/def _json_schema_to_python_type(schema: Any, defs) -> str:/def _json_schema_to_python_type(schema: Any, defs) -> str:\n    # Handle non-dict schemas (e.g., boolean from additionalProperties: true)\n    if not isinstance(schema, dict):\n        return "Any"/' "$UTILS_FILE"
    echo "Patched gradio_client for compatibility"
fi

echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
echo "To run TRELLIS:"
echo ""
echo "  source .venv/bin/activate"
echo "  ATTN_BACKEND=sdpa XFORMERS_DISABLED=1 SPARSE_BACKEND=torchsparse \\"
echo "    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python app.py"
echo ""
echo "Then open http://localhost:7860 in your browser"
echo ""
echo "NOTE: GLB export takes 5-10 minutes - this is normal!"
echo "      Gaussian export is much faster (~30 seconds)."
echo ""
