#!/bin/bash
# Manual HIP build script for diff-gaussian-rasterization
# This bypasses PyTorch's BuildExtension which adds -fno-gpu-rdc that breaks kernel execution

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="/home/hubcaps/.gemini/antigravity/scratch/TRELLIS/.venv"
ROCM_DIR="/opt/rocm-6.4.2"
BUILD_DIR="${SCRIPT_DIR}/build_manual"
HIP_RASTERIZER="${SCRIPT_DIR}/hip_rasterizer"

# Python and PyTorch paths
PYTHON_INCLUDE="${VENV_DIR}/include/python3.10"
TORCH_INCLUDE="${VENV_DIR}/lib/python3.10/site-packages/torch/include"
TORCH_LIB="${VENV_DIR}/lib/python3.10/site-packages/torch/lib"

# Create build directory
mkdir -p "${BUILD_DIR}"

echo "=== Building diff-gaussian-rasterization with manual hipcc (NO -fno-gpu-rdc) ==="

# Compile flags - NOTE: we do NOT use -fno-gpu-rdc!
HIPCC="${ROCM_DIR}/bin/hipcc"
COMMON_FLAGS="-fPIC -O3 -std=c++17 --offload-arch=gfx1101"
COMMON_FLAGS+=" -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2"
COMMON_FLAGS+=" -DCUDA_HAS_FP16=1 -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1"
COMMON_FLAGS+=" -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=1"
COMMON_FLAGS+=" -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\""

INCLUDES="-I${TORCH_INCLUDE}"
INCLUDES+=" -I${TORCH_INCLUDE}/torch/csrc/api/include"
INCLUDES+=" -I${TORCH_INCLUDE}/TH -I${TORCH_INCLUDE}/THC -I${TORCH_INCLUDE}/THH"
INCLUDES+=" -I${ROCM_DIR}/include"
INCLUDES+=" -I/usr/include/python3.10"
INCLUDES+=" -I${VENV_DIR}/include"
INCLUDES+=" -I${SCRIPT_DIR}/third_party/glm/"
INCLUDES+=" -I${HIP_RASTERIZER}"

echo "[1/5] Compiling rasterizer_impl.hip..."
${HIPCC} ${COMMON_FLAGS} ${INCLUDES} -c "${HIP_RASTERIZER}/rasterizer_impl.hip" -o "${BUILD_DIR}/rasterizer_impl.o"

echo "[2/5] Compiling forward.hip..."
${HIPCC} ${COMMON_FLAGS} ${INCLUDES} -c "${HIP_RASTERIZER}/forward.hip" -o "${BUILD_DIR}/forward.o"

echo "[3/5] Compiling backward.hip..."
${HIPCC} ${COMMON_FLAGS} ${INCLUDES} -c "${HIP_RASTERIZER}/backward.hip" -o "${BUILD_DIR}/backward.o"

echo "[4/5] Compiling rasterize_points.hip..."
${HIPCC} ${COMMON_FLAGS} ${INCLUDES} -c "${SCRIPT_DIR}/rasterize_points.hip" -o "${BUILD_DIR}/rasterize_points.o"

echo "[5/5] Compiling ext.cpp..."
g++ -fPIC -O3 -std=c++17 \
    -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 \
    -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=1 \
    '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' \
    ${INCLUDES} -c "${SCRIPT_DIR}/ext.cpp" -o "${BUILD_DIR}/ext.o"

echo "[LINK] Creating shared library..."
g++ -shared -Wl,-O1 -Wl,-Bsymbolic-functions \
    -Wl,-rpath,${TORCH_LIB} -Wl,-rpath,${ROCM_DIR}/lib \
    "${BUILD_DIR}/rasterizer_impl.o" \
    "${BUILD_DIR}/forward.o" \
    "${BUILD_DIR}/backward.o" \
    "${BUILD_DIR}/rasterize_points.o" \
    "${BUILD_DIR}/ext.o" \
    -L${TORCH_LIB} -L${ROCM_DIR}/lib -L${ROCM_DIR}/hip/lib \
    -lc10 -ltorch -ltorch_cpu -ltorch_python -lamdhip64 -lc10_hip -ltorch_hip \
    -o "${BUILD_DIR}/_C.cpython-310-x86_64-linux-gnu.so"

# Install to venv
echo "[INSTALL] Installing to venv..."
INSTALL_DIR="${VENV_DIR}/lib/python3.10/site-packages/diff_gaussian_rasterization"
cp "${BUILD_DIR}/_C.cpython-310-x86_64-linux-gnu.so" "${INSTALL_DIR}/"

echo "=== Build complete! ==="
echo "Installed to: ${INSTALL_DIR}/_C.cpython-310-x86_64-linux-gnu.so"
