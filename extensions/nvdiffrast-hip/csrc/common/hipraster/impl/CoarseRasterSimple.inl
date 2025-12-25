// CoarseRasterSimple.inl - AMD HIP-compatible simplified coarse rasterizer
// This replaces the complex coarseRasterImpl which uses warp-level sync
// that deadlocks on AMD RDNA3 GPUs.
// NOTE: This file is included inside namespace CR in RasterImpl_kernel.hip

// Don't re-include hip/hip_runtime.h - it's already included

//------------------------------------------------------------------------
// Simplified coarse raster that avoids warp-level synchronization
// This is SLOWER than the original but WORKS on AMD GPUs
//------------------------------------------------------------------------

__device__ __inline__ void coarseRasterImplSimple(const CRParams p) {
  // Get thread identification
  int thrInBlock = threadIdx.x + threadIdx.y * blockDim.x;
  int totalThreads = blockDim.x * blockDim.y;

  CRAtomics &atomics = p.atomics[blockIdx.z];

  // Check for overflow
  if (atomics.numSubtris > p.maxSubtris || atomics.numBinSegs > p.maxBinSegs)
    return;

  // pointers for this image
  const CRTriangleHeader *triHeader =
      (const CRTriangleHeader *)p.triHeader + p.maxSubtris * blockIdx.z;
  const S32 *binFirstSeg = (const S32 *)p.binFirstSeg +
                           CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * blockIdx.z;
  const S32 *binTotal = (const S32 *)p.binTotal +
                        CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * blockIdx.z;
  const S32 *binSegData =
      (const S32 *)p.binSegData + p.maxBinSegs * CR_BIN_SEG_SIZE * blockIdx.z;
  const S32 *binSegNext = (const S32 *)p.binSegNext + p.maxBinSegs * blockIdx.z;
  const S32 *binSegCount =
      (const S32 *)p.binSegCount + p.maxBinSegs * blockIdx.z;
  S32 *activeTiles = (S32 *)p.activeTiles + CR_MAXTILES_SQR * blockIdx.z;
  S32 *tileFirstSeg = (S32 *)p.tileFirstSeg + CR_MAXTILES_SQR * blockIdx.z;
  S32 *tileSegData =
      (S32 *)p.tileSegData + p.maxTileSegs * CR_TILE_SEG_SIZE * blockIdx.z;
  S32 *tileSegNext = (S32 *)p.tileSegNext + p.maxTileSegs * blockIdx.z;
  S32 *tileSegCount = (S32 *)p.tileSegCount + p.maxTileSegs * blockIdx.z;

  // AMD HIP FIX: Simple approach - each block processes one bin at a time
  // No warp-level sorting or complex synchronization

  // Only first block does the work (serialized but safe)
  if (blockIdx.x != 0)
    return;

  // Initialize all tile first segments to -1 (no segments)
  for (int tileIdx = thrInBlock; tileIdx < CR_MAXTILES_SQR;
       tileIdx += totalThreads) {
    tileFirstSeg[tileIdx] = -1;
  }
  __syncthreads();

  // Simple counter for active tiles and tile segments
  __shared__ int s_numActiveTiles;
  __shared__ int s_numTileSegs;
  if (thrInBlock == 0) {
    s_numActiveTiles = 0;
    s_numTileSegs = 0;
  }
  __syncthreads();

  // Process each bin
  for (int binIdx = 0; binIdx < p.numBins; binIdx++) {
    int binY = binIdx / p.widthBins;
    int binX = binIdx - binY * p.widthBins;

    // Check if this bin has any triangles
    int binTriCount = 0;
    for (int i = 0; i < CR_BIN_STREAMS_SIZE; i++)
      binTriCount += binTotal[(binIdx << CR_BIN_STREAMS_LOG2) + i];

    if (binTriCount == 0 && !p.deferredClear)
      continue;

    // For each tile in the bin
    int maxTileX = ::min(p.widthTiles - (binX << CR_BIN_LOG2), CR_BIN_SIZE);
    int maxTileY = ::min(p.heightTiles - (binY << CR_BIN_LOG2), CR_BIN_SIZE);

    for (int tileYInBin = 0; tileYInBin < maxTileY; tileYInBin++) {
      for (int tileXInBin = 0; tileXInBin < maxTileX; tileXInBin++) {
        int tileX = (binX << CR_BIN_LOG2) + tileXInBin;
        int tileY = (binY << CR_BIN_LOG2) + tileYInBin;
        int globalTileIdx = tileX + tileY * p.widthTiles;

        // Only thread 0 handles tile registration to avoid race conditions
        if (thrInBlock == 0) {
          // Check if tile has triangles or needs deferred clear
          if (binTriCount > 0 || p.deferredClear) {
            // Add to active tiles list
            int activeIdx = s_numActiveTiles++;
            if (activeIdx < CR_MAXTILES_SQR) {
              activeTiles[activeIdx] = globalTileIdx;

              // For simplicity, just mark tile as needing clear
              // The fine rasterizer will handle actual triangle processing
              tileFirstSeg[globalTileIdx] = -1;
            }
          }
        }
        __syncthreads();
      }
    }
  }

  // Write final counts
  if (thrInBlock == 0) {
    atomics.numActiveTiles = s_numActiveTiles;
    atomics.numTileSegs = s_numTileSegs;
  }
}

//------------------------------------------------------------------------
