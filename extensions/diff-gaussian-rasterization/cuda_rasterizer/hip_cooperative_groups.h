/*
 * HIP Cooperative Groups Compatibility Layer
 * Provides CUDA cooperative_groups equivalents for HIP/ROCm
 */

#ifndef HIP_COOPERATIVE_GROUPS_H
#define HIP_COOPERATIVE_GROUPS_H

#include <hip/hip_runtime.h>

#ifdef __HIP_PLATFORM_AMD__

namespace cooperative_groups {

// Forward declarations
class thread_block;
class grid_group;

// Thread block class - provides block-level cooperative operations
class thread_block {
public:
  __device__ __forceinline__ dim3 group_index() const { return blockIdx; }

  __device__ __forceinline__ dim3 thread_index() const { return threadIdx; }

  __device__ __forceinline__ unsigned int thread_rank() const {
    return threadIdx.x + threadIdx.y * blockDim.x +
           threadIdx.z * blockDim.x * blockDim.y;
  }

  __device__ __forceinline__ unsigned int size() const {
    return blockDim.x * blockDim.y * blockDim.z;
  }

  __device__ __forceinline__ void sync() const { __syncthreads(); }
};

// Grid group class - provides grid-level operations
class grid_group {
public:
  __device__ __forceinline__ unsigned long long thread_rank() const {
    return (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x +
           ((unsigned long long)blockIdx.y * blockDim.x * gridDim.x +
            threadIdx.y * blockDim.x) +
           ((unsigned long long)blockIdx.z * blockDim.x * gridDim.x *
                gridDim.y +
            threadIdx.z * blockDim.x * blockDim.y);
  }

  __device__ __forceinline__ unsigned long long size() const {
    return (unsigned long long)gridDim.x * gridDim.y * gridDim.z * blockDim.x *
           blockDim.y * blockDim.z;
  }
};

// Factory functions
__device__ __forceinline__ thread_block this_thread_block() {
  return thread_block();
}

__device__ __forceinline__ grid_group this_grid() { return grid_group(); }

} // namespace cooperative_groups

#endif // __HIP_PLATFORM_AMD__

#endif // HIP_COOPERATIVE_GROUPS_H
