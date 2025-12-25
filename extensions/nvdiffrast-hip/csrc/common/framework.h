// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

// Framework-specific macros to enable code sharing.

//------------------------------------------------------------------------
// PyTorch.

#ifdef NVDR_TORCH
#ifndef __CUDACC__
#include <torch/extension.h>

// Use HIP headers for ROCm, CUDA headers for NVIDIA
#ifdef __HIP_PLATFORM_AMD__
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPStream.h>
#else
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include <pybind11/numpy.h>
#endif // __CUDACC__

#define NVDR_CHECK(COND, ERR)                                                  \
  do {                                                                         \
    TORCH_CHECK(COND, ERR)                                                     \
  } while (0)

#ifdef __HIP_PLATFORM_AMD__
#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL)                                       \
  do {                                                                         \
    hipError_t err = CUDA_CALL;                                                \
    TORCH_CHECK(!err, "Cuda error: ", hipGetLastError(), "[", #CUDA_CALL,      \
                ";]");                                                         \
  } while (0)
#else
#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL)                                       \
  do {                                                                         \
    cudaError_t err = CUDA_CALL;                                               \
    TORCH_CHECK(!err, "Cuda error: ", cudaGetLastError(), "[", #CUDA_CALL,     \
                ";]");                                                         \
  } while (0)
#endif

#endif // NVDR_TORCH

//------------------------------------------------------------------------
