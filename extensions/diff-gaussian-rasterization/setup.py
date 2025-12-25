#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import re
import glob
import torch

# Check if we're running on ROCm/HIP
is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
base_dir = os.path.dirname(os.path.abspath(__file__))

def fix_hipify_issues(content):
    """Fix issues in hipified files for HIP/ROCm compatibility."""
    
    # Fix malformed kernel launch syntax: << <grid, block >> > -> hipLaunchKernelGGL
    # Actual format from hipify: '<< <grid, block >> > ('
    pattern = r'(\w+(?:<[^>]+>)?)\s*<<\s*<\s*([^,]+),\s*([^>]+)\s*>>\s*>\s*\('
    
    def convert_kernel_launch(match):
        kernel = match.group(1)
        grid = match.group(2).strip()
        block = match.group(3).strip()
        return f'hipLaunchKernelGGL({kernel}, {grid}, {block}, 0, 0, '
    
    # Apply the fix
    fixed = re.sub(pattern, convert_kernel_launch, content)
    
    # Also handle standard <<< >>> syntax
    pattern2 = r'(\w+(?:<[^>]+>)?)\s*<<<\s*([^,]+),\s*([^>]+)>>>\s*\('
    fixed = re.sub(pattern2, convert_kernel_launch, fixed)
    
    # Fix extra 0, 0 args
    fixed = re.sub(r',\s*0,\s*0,\s*0,\s*0,', ', 0, 0,', fixed)
    
    return fixed


if is_rocm:
    # For ROCm/HIP, we need to use our pre-fixed hip files as sources
    # This prevents PyTorch's CUDAExtension from re-running hipify
    hip_rasterizer_dir = os.path.join(base_dir, "hip_rasterizer")
    
    # Check if we have already hipified and fixed files
    hip_forward = os.path.join(hip_rasterizer_dir, "forward.hip")
    has_fixed_files = os.path.exists(hip_forward)
    
    if has_fixed_files:
        # Check if the files have already been fixed
        with open(hip_forward, 'r') as f:
            content = f.read()
        needs_fix = '<< <' in content  # malformed syntax indicator
        
        if needs_fix:
            print("[HIP FIX] Fixing kernel launch syntax in hipified files...")
            hip_files = glob.glob(os.path.join(hip_rasterizer_dir, "*.hip"))
            for hip_file in hip_files:
                with open(hip_file, 'r') as f:
                    content = f.read()
                fixed_content = fix_hipify_issues(content)
                if fixed_content != content:
                    print(f"[HIP FIX] Fixed: {os.path.basename(hip_file)}")
                    with open(hip_file, 'w') as f:
                        f.write(fixed_content)
        
        # Use the hipified .hip files directly as sources
        sources = [
            os.path.join(hip_rasterizer_dir, "rasterizer_impl.hip"),
            os.path.join(hip_rasterizer_dir, "forward.hip"),
            os.path.join(hip_rasterizer_dir, "backward.hip"),
            os.path.join(base_dir, "rasterize_points.hip") if os.path.exists(os.path.join(base_dir, "rasterize_points.hip")) else os.path.join(base_dir, "rasterize_points.cu"),
            os.path.join(base_dir, "ext.cpp"),
        ]
        
        # Force compile only for gfx1100 to avoid multi-arch issues
        extra_compile_args = {
            "nvcc": [
                "-I" + os.path.join(base_dir, "third_party/glm/"),
                "-I" + hip_rasterizer_dir,  # For headers
                "--offload-arch=gfx1100",  # Force single architecture
                "-fgpu-rdc",  # Enable GPU relocatable device code
            ]
        }
        print(f"[HIP] Using pre-fixed hipified sources from {hip_rasterizer_dir}")
    else:
        # First run - use original .cu files, hipify will generate them
        sources = [
            os.path.join(base_dir, "cuda_rasterizer/rasterizer_impl.cu"),
            os.path.join(base_dir, "cuda_rasterizer/forward.cu"),
            os.path.join(base_dir, "cuda_rasterizer/backward.cu"),
            os.path.join(base_dir, "rasterize_points.cu"),
            os.path.join(base_dir, "ext.cpp"),
        ]
        extra_compile_args = {
            "nvcc": [
                "-I" + os.path.join(base_dir, "third_party/glm/"),
                "-I" + os.path.join(base_dir, "cuda_rasterizer/"),
            ]
        }
        print("[HIP] First build - will use original .cu files. Run again after hipify to use fixed files.")
else:
    # CUDA path - use original files
    sources = [
        os.path.join(base_dir, "cuda_rasterizer/rasterizer_impl.cu"),
        os.path.join(base_dir, "cuda_rasterizer/forward.cu"),
        os.path.join(base_dir, "cuda_rasterizer/backward.cu"),
        os.path.join(base_dir, "rasterize_points.cu"),
        os.path.join(base_dir, "ext.cpp"),
    ]
    extra_compile_args = {
        "nvcc": ["-I" + os.path.join(base_dir, "third_party/glm/")]
    }


setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=sources,
            extra_compile_args=extra_compile_args)
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
