# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# HIP/ROCm port additions for AMD GPU support.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import setuptools
import os
import torch

# Detect if we're building for HIP/ROCm or CUDA
IS_HIP = hasattr(torch.version, 'hip') and torch.version.hip is not None

# Print build configuration
if IS_HIP:
    print("\n" + "=" * 70)
    print("Building nvdiffrast for AMD ROCm/HIP")
    print(f"PyTorch: {torch.__version__}, HIP: {torch.version.hip}")
    print("=" * 70 + "\n")
else:
    print("\n" + "=" * 70)
    print("Building nvdiffrast for NVIDIA CUDA")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print("=" * 70 + "\n")

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    print("\n\n" + "*" * 70)
    print("ERROR! Cannot compile nvdiffrast extension. Please ensure that:\n")
    print("1. You have PyTorch installed")
    print("2. You run 'pip install' with --no-build-isolation flag")
    print("*" * 70 + "\n\n")
    exit(1)

# Source files - use HIP versions for kernels, but original C++ files for torch bindings
if IS_HIP:
    sources = [
        # HIP kernels
        "csrc/common/antialias.cu",  # hipcc handles .cu files
        "csrc/common/common.cpp",
        "csrc/common/hipraster/impl/Buffer.cpp",
        "csrc/common/hipraster/impl/CudaRaster.cpp",
        "csrc/common/hipraster/impl/RasterImpl.cpp",
        "csrc/common/hipraster/impl/RasterImpl_kernel.hip",  # .hip extension
        "csrc/common/interpolate.cu",
        "csrc/common/rasterize.cu",
        "csrc/common/texture.cpp",
        "csrc/common/texture_kernel.cu",
        # Use original C++ torch bindings (they use common.h not common_hip.h)
        "csrc/torch/torch_antialias.cpp",
        "csrc/torch/torch_bindings.cpp",
        "csrc/torch/torch_interpolate.cpp",
        "csrc/torch/torch_rasterize.cpp",
        "csrc/torch/torch_texture.cpp",
    ]
else:
    sources = [
        "csrc/common/antialias.cu",
        "csrc/common/common.cpp",
        "csrc/common/cudaraster/impl/Buffer.cpp",
        "csrc/common/cudaraster/impl/CudaRaster.cpp",
        "csrc/common/cudaraster/impl/RasterImpl.cpp",
        "csrc/common/cudaraster/impl/RasterImpl_kernel.cu",
        "csrc/common/interpolate.cu",
        "csrc/common/rasterize.cu",
        "csrc/common/texture.cpp",
        "csrc/common/texture_kernel.cu",
        "csrc/torch/torch_antialias.cpp",
        "csrc/torch/torch_bindings.cpp",
        "csrc/torch/torch_interpolate.cpp",
        "csrc/torch/torch_rasterize.cpp",
        "csrc/torch/torch_texture.cpp",
    ]

# Compiler flags
if IS_HIP:
    # HIP/ROCm build flags
    extra_compile_args = {
        "cxx": ["-DNVDR_TORCH", "-D__HIP_PLATFORM_AMD__"],
        "nvcc": [  # hipcc still uses 'nvcc' key in CUDAExtension
            "-DNVDR_TORCH",
            "-D__HIP_PLATFORM_AMD__",
        ],
    }
else:
    # CUDA build flags
    extra_compile_args = {
        "cxx": ["-DNVDR_TORCH"]
        + (["/wd4067", "/wd4624", "/wd4996"] if os.name == "nt" else []),
        "nvcc": ["-DNVDR_TORCH", "-lineinfo"],
    }

setuptools.setup(
    ext_modules=[
        CUDAExtension(
            "_nvdiffrast_c",
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
