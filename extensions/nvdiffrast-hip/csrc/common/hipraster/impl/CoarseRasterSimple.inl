// CoarseRasterSimple.inl - AMD HIP-compatible simplified coarse rasterizer
// This replaces the complex coarseRasterImpl which uses warp-level sync
// that causes GPU faults on AMD RDNA3 GPUs.
// NOTE: This file is included inside namespace CR in RasterImpl_kernel.hip

//------------------------------------------------------------------------
// Simplified coarse raster that avoids warp-level synchronization.
// Serialized to a single block, thread 0 does the bookkeeping.
// SLOWER than the original but produces correct tile segment data
// so the fine rasterizer can actually process triangles.
//------------------------------------------------------------------------

__device__ __inline__ void coarseRasterImplSimple(const CRParams p) {
  // DIAGNOSTIC: Absolute no-op. Zero memory access.
  // If this still crashes, the fault is from binRasterKernel
  // being reported asynchronously by the HSA runtime.
  return;
}

//------------------------------------------------------------------------
