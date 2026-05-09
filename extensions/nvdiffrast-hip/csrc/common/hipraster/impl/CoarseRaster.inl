#include "hip/hip_runtime.h"
// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

//------------------------------------------------------------------------

__device__ __inline__ int globalTileIdx(int tileInBin, int widthTiles) {
  int tileX = tileInBin & (CR_BIN_SIZE - 1);
  int tileY = tileInBin >> CR_BIN_LOG2;
  return tileX + tileY * widthTiles;
}

//------------------------------------------------------------------------

__device__ __inline__ void coarseRasterImpl(const CRParams& p, char* s_smem) {
  // SAFE-MODE temporarily disabled for Test 116 diagnostic. The real
  // impl returns at line ~363 after recording currPtr diagnostics, no
  // offending write performed. Restore SAFE-MODE after the run.
  // Diagnostic checkpoint markers + Test 42 OOB clamps remain below for
  // future debugging.
  S32* dbg = (S32*)p.debugTrace;
  if (dbg) atomicMax(dbg, 1);
  CoarseSmem& smem = *(CoarseSmem*)s_smem;

  // AMD RDNA3 LDS-relief: warpEmitMask + warpEmitPrefixSum live in per-block
  // global memory. One CoarseGlobalScratch slot per (image, block).
  // Pointer-arithmetic binary search at "Find warp in tile" requires these two
  // arrays be contiguous in memory (no padding) -- preserved by struct layout.
  CoarseGlobalScratch* gscratch =
      ((CoarseGlobalScratch*)p.warpEmitGlobal) + blockIdx.x +
      gridDim.x * blockIdx.z;

  // Alias struct members to original variable names so code below is unchanged.
  volatile S32& s_oobCount = smem.oobCount;
  volatile U32& s_workCounter = smem.workCounter;
  volatile U32 (&s_scanTemp)[CR_COARSE_WARPS][48] = smem.scanTemp;
  volatile U32 (&s_binOrder)[CR_MAXBINS_SQR] = smem.binOrder;
  volatile S32 (&s_binStreamCurrSeg)[CR_BIN_STREAMS_SIZE] = smem.binStreamCurrSeg;
  volatile S32 (&s_binStreamFirstTri)[CR_BIN_STREAMS_SIZE] = smem.binStreamFirstTri;
  volatile S32 (&s_triQueue)[CR_COARSE_QUEUE_SIZE] = smem.triQueue;
  volatile S32& s_triQueueWritePos = smem.triQueueWritePos;
  volatile U32& s_binStreamSelectedOfs = smem.binStreamSelectedOfs;
  volatile U32& s_binStreamSelectedSize = smem.binStreamSelectedSize;
  volatile U32 (&s_warpEmitMask)[CR_COARSE_WARPS][CR_BIN_SQR + 1] = gscratch->warpEmitMask;
  volatile U32 (&s_warpEmitPrefixSum)[CR_COARSE_WARPS][CR_BIN_SQR + 1] = gscratch->warpEmitPrefixSum;
  volatile U32 (&s_tileEmitPrefixSum)[CR_BIN_SQR + 1] = smem.tileEmitPrefixSum;
  volatile U32 (&s_tileAllocPrefixSum)[CR_BIN_SQR + 1] = smem.tileAllocPrefixSum;
  volatile S32 (&s_tileStreamCurrOfs)[CR_BIN_SQR] = smem.tileStreamCurrOfs;
  volatile U32& s_firstAllocSeg = smem.firstAllocSeg;
  volatile U32& s_firstActiveIdx = smem.firstActiveIdx;

  // Pointers and constants.

  CRAtomics &atomics = p.atomics[blockIdx.z];
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

  int tileLog = CR_TILE_LOG2 + CR_SUBPIXEL_LOG2;
  int thrInBlock = threadIdx.x + threadIdx.y * 32;
  int emitShift =
      CR_BIN_LOG2 * 2 +
      5; // We scan ((numEmits << emitShift) | numAllocs) over tiles.

  // CHECKPOINT 2: pointer & gscratch setup complete.
  if (dbg) atomicMax(dbg, 2);

  if (atomics.numSubtris > p.maxSubtris || atomics.numBinSegs > p.maxBinSegs)
    return;

  // Test 42: bounds limit for tile segment writes.
  int maxTileSegOfs = p.maxTileSegs * CR_TILE_SEG_SIZE;

  // CHECKPOINT 3: passed early-out check.
  if (dbg) atomicMax(dbg, 3);

  // Initialize sharedmem arrays.

  if (thrInBlock == 0) {
    s_tileEmitPrefixSum[0] = 0;
    s_tileAllocPrefixSum[0] = 0;
#if CR_DEBUG_OOB
    s_oobCount = 0;
#endif
  }
  s_scanTemp[threadIdx.y][threadIdx.x] = 0;

  // CHECKPOINT 4: smem init done.
  if (dbg) atomicMax(dbg, 4);

  // Sort bins in descending order of triangle count.
  // AMD HIP FIX: Skip sorting - just use natural bin order
  // sortShared has warp-level sync issues on AMD RDNA3

  for (int binIdx = thrInBlock; binIdx < p.numBins;
       binIdx += CR_COARSE_WARPS * 32) {
    // AMD FIX: Just use binIdx directly instead of sorted order
    // Original code computed triangle counts and sorted, but that causes hangs
    s_binOrder[binIdx] = binIdx; // Simple identity mapping
  }

  __syncthreads();
  // AMD HIP FIX: sortShared commented out - causes deadlock on AMD
  // sortShared(s_binOrder, p.numBins);

  // CHECKPOINT 5: binOrder init done, entering main bin loop.
  if (dbg) atomicMax(dbg, 5);

  // Process each bin by one block.

  for (;;) {
    // CHECKPOINT 6: bin loop iter start.
    if (dbg) atomicMax(dbg, 6);
    // Pick a bin for the block.

    if (thrInBlock == 0)
      s_workCounter = atomicAdd(&atomics.coarseCounter, 1);
    __syncthreads();

    int workCounter = s_workCounter;
    if (workCounter >= p.numBins)
      break;

    U32 binOrder = s_binOrder[workCounter];
    // AMD HIP FIX: Since we use identity mapping (binOrder = binIdx),
    // we can't detect empty bins from the encoding. Check binTotal directly.
    int binIdx = binOrder; // Now binOrder IS the binIdx directly

    // Check if bin has any triangles
    int triCount = 0;
    for (int i = 0; i < CR_BIN_STREAMS_SIZE; i++)
      triCount += binTotal[(binIdx << CR_BIN_STREAMS_LOG2) + i];
    bool binEmpty = (triCount == 0);
    if (binEmpty && !p.deferredClear)
      continue; // Skip empty bins instead of break

    // CHECKPOINT 7: passed binEmpty check (non-empty bin found).
    if (dbg) atomicMax(dbg, 7);

    // Initialize input/output streams.

    int triQueueWritePos = 0;
    int triQueueReadPos = 0;

    if (thrInBlock < CR_BIN_STREAMS_SIZE) {
      int segIdx = binFirstSeg[(binIdx << CR_BIN_STREAMS_LOG2) + thrInBlock];
      s_binStreamCurrSeg[thrInBlock] = segIdx;
      s_binStreamFirstTri[thrInBlock] =
          (segIdx == -1) ? ~0u : binSegData[segIdx << CR_BIN_SEG_LOG2];
    }

    for (int tileInBin = CR_COARSE_WARPS * 32 - 1 - thrInBlock;
         tileInBin < CR_BIN_SQR; tileInBin += CR_COARSE_WARPS * 32)
      s_tileStreamCurrOfs[tileInBin] = -CR_TILE_SEG_SIZE;

    // Initialize per-bin state.

    int binY = idiv_fast(binIdx, p.widthBins);
    int binX = binIdx - binY * p.widthBins;
    int originX = (binX << (CR_BIN_LOG2 + tileLog)) -
                  (p.widthPixelsVp << (CR_SUBPIXEL_LOG2 - 1));
    int originY = (binY << (CR_BIN_LOG2 + tileLog)) -
                  (p.heightPixelsVp << (CR_SUBPIXEL_LOG2 - 1));
    int maxTileXInBin =
        ::min(p.widthTiles - (binX << CR_BIN_LOG2), CR_BIN_SIZE) - 1;
    int maxTileYInBin =
        ::min(p.heightTiles - (binY << CR_BIN_LOG2), CR_BIN_SIZE) - 1;
    int binTileIdx = (binX + binY * p.widthTiles) << CR_BIN_LOG2;

    // CHECKPOINT 8: stream init complete, entering merge do-while.
    if (dbg) atomicMax(dbg, 8);

    // Entire block: Merge input streams and process triangles.

    if (!binEmpty)
      do {
        // CHECKPOINT 9: merge do-while iter start.
        if (dbg) atomicMax(dbg, 9);
        //------------------------------------------------------------------------
        // Merge.
        //------------------------------------------------------------------------

        // Entire block: Not enough triangles => merge and queue segments.
        // NOTE: The bin exit criterion assumes that we queue more triangles
        // than we actually need.

        while (triQueueWritePos - triQueueReadPos <= CR_COARSE_WARPS * 32) {
          // First warp: Choose the segment with the lowest initial triangle
          // index.

          bool hasStream = (thrInBlock < CR_BIN_STREAMS_SIZE);
          U32 hasStreamMask = __ballot_sync(~0u, hasStream);
          if (hasStream) {
            // Find the stream with the lowest triangle index.

            U32 firstTri = s_binStreamFirstTri[thrInBlock];
            U32 t = firstTri;
            volatile U32 *v = &s_scanTemp[0][thrInBlock + 16];

#if (CR_BIN_STREAMS_SIZE > 1)
            v[0] = t;
            __syncwarp(hasStreamMask);
            t = ::min(t, v[-1]);
            __syncwarp(hasStreamMask);
#endif
#if (CR_BIN_STREAMS_SIZE > 2)
            v[0] = t;
            __syncwarp(hasStreamMask);
            t = ::min(t, v[-2]);
            __syncwarp(hasStreamMask);
#endif
#if (CR_BIN_STREAMS_SIZE > 4)
            v[0] = t;
            __syncwarp(hasStreamMask);
            t = ::min(t, v[-4]);
            __syncwarp(hasStreamMask);
#endif
#if (CR_BIN_STREAMS_SIZE > 8)
            v[0] = t;
            __syncwarp(hasStreamMask);
            t = ::min(t, v[-8]);
            __syncwarp(hasStreamMask);
#endif
#if (CR_BIN_STREAMS_SIZE > 16)
            v[0] = t;
            __syncwarp(hasStreamMask);
            t = ::min(t, v[-16]);
            __syncwarp(hasStreamMask);
#endif
            v[0] = t;
            __syncwarp(hasStreamMask);

            // Consume and broadcast.

            bool first =
                (s_scanTemp[0][CR_BIN_STREAMS_SIZE - 1 + 16] == firstTri);
            U32 firstMask = __ballot_sync(hasStreamMask, first);
            if (first && (firstMask >> threadIdx.x) == 1u) {
              int segIdx = s_binStreamCurrSeg[thrInBlock];
              s_binStreamSelectedOfs = segIdx << CR_BIN_SEG_LOG2;
              if (segIdx != -1) {
                int segSize = binSegCount[segIdx];
                int segNext = binSegNext[segIdx];
                s_binStreamSelectedSize = segSize;
                s_triQueueWritePos = triQueueWritePos + segSize;
                s_binStreamCurrSeg[thrInBlock] = segNext;
                s_binStreamFirstTri[thrInBlock] =
                    (segNext == -1) ? ~0u
                                    : binSegData[segNext << CR_BIN_SEG_LOG2];
              }
            }
          }

          // No more segments => break.

          __syncthreads();
          triQueueWritePos = s_triQueueWritePos;
          int segOfs = s_binStreamSelectedOfs;
          if (segOfs < 0)
            break;

          int segSize = s_binStreamSelectedSize;
          __syncthreads();

          // Fetch triangles into the queue.

          for (int idxInSeg = CR_COARSE_WARPS * 32 - 1 - thrInBlock;
               idxInSeg < segSize; idxInSeg += CR_COARSE_WARPS * 32) {
            S32 triIdx = binSegData[segOfs + idxInSeg];
            s_triQueue[(triQueueWritePos - segSize + idxInSeg) &
                       (CR_COARSE_QUEUE_SIZE - 1)] = triIdx;
          }
        }

        // All threads: Clear emit masks.

        for (int maskIdx = thrInBlock; maskIdx < CR_COARSE_WARPS * CR_BIN_SQR;
             maskIdx += CR_COARSE_WARPS * 32)
          s_warpEmitMask[maskIdx >> (CR_BIN_LOG2 * 2)]
                        [maskIdx & (CR_BIN_SQR - 1)] = 0;

        __syncthreads();

        //------------------------------------------------------------------------
        // Raster.
        //------------------------------------------------------------------------

        // Triangle per thread: Read from the queue.

        int triIdx = -1;
        if (triQueueReadPos + thrInBlock < triQueueWritePos)
          triIdx = s_triQueue[(triQueueReadPos + thrInBlock) &
                              (CR_COARSE_QUEUE_SIZE - 1)];

        // Test 125: count number of threads where bounds check would
        // fire. atomicAdd to dbg[16] every time pre-.misc dataIdx is
        // OOB; atomicAdd to dbg[17] every time post-.misc dataIdx is
        // OOB. atomicCAS for one OOB sample to dbg[18..21].
        uint4 triData = make_uint4(0, 0, 0, 0);
        if (triIdx != -1) {
          int dataIdx = triIdx >> 3;
          int subtriIdx = triIdx & 7;
          int origDataIdx = dataIdx;
          bool preOOB = (dataIdx >= (int)p.maxSubtris);
          if (subtriIdx != 7) {
            if (!preOOB)
              dataIdx = triHeader[dataIdx].misc + subtriIdx;
            else
              dataIdx = -1;
          }
          bool postOOB = (dataIdx < 0 || dataIdx >= (int)p.maxSubtris);
          if (preOOB && dbg) {
            U32* dbgU = (U32*)dbg;
            atomicAdd(&dbgU[16], 1u);
            // Capture first OOB sample
            if (atomicCAS(&dbgU[18], 0u, 1u) == 0u) {
              dbgU[19] = (U32)triIdx;
              dbgU[20] = (U32)origDataIdx;
              dbgU[21] = (U32)((threadIdx.y << 16) | threadIdx.x);
            }
          }
          if (!preOOB && postOOB && dbg) {
            U32* dbgU = (U32*)dbg;
            atomicAdd(&dbgU[17], 1u);
            // Capture first post-OOB sample
            if (atomicCAS(&dbgU[22], 0u, 1u) == 0u) {
              dbgU[23] = (U32)triIdx;
              dbgU[24] = (U32)origDataIdx;
              dbgU[25] = (U32)subtriIdx;
              dbgU[26] = (U32)dataIdx;       // post-.misc dataIdx (could be huge or -1)
              // Also capture .misc itself
              dbgU[27] = (U32)triHeader[origDataIdx].misc;
            }
          }
          if (dataIdx >= 0 && dataIdx < (int)p.maxSubtris)
            triData = *((uint4 *)triHeader + dataIdx);
        }

        // 32 triangles per warp: Record emits (= tile intersections).

        if (__any_sync(~0u, triIdx != -1)) {
          S32 v0x = sub_s16lo_s16lo(triData.x, originX);
          S32 v0y = sub_s16hi_s16lo(triData.x, originY);
          S32 d01x = sub_s16lo_s16lo(triData.y, triData.x);
          S32 d01y = sub_s16hi_s16hi(triData.y, triData.x);
          S32 d02x = sub_s16lo_s16lo(triData.z, triData.x);
          S32 d02y = sub_s16hi_s16hi(triData.z, triData.x);

          // Compute tile-based AABB.

          int lox = add_clamp_0_x((v0x + min_min(d01x, 0, d02x)) >> tileLog, 0,
                                  maxTileXInBin);
          int loy = add_clamp_0_x((v0y + min_min(d01y, 0, d02y)) >> tileLog, 0,
                                  maxTileYInBin);
          int hix = add_clamp_0_x((v0x + max_max(d01x, 0, d02x)) >> tileLog, 0,
                                  maxTileXInBin);
          int hiy = add_clamp_0_x((v0y + max_max(d01y, 0, d02y)) >> tileLog, 0,
                                  maxTileYInBin);
          int sizex = add_sub(hix, 1, lox);
          int sizey = add_sub(hiy, 1, loy);
          int area = sizex * sizey;

          // Miscellaneous init.

          U8 *currPtr =
              (U8 *)&s_warpEmitMask[threadIdx.y][lox + (loy << CR_BIN_LOG2)];
          int ptrYInc = CR_BIN_SIZE * 4 - (sizex << 2);
          U32 maskBit = 1 << threadIdx.x;

          // Test 121: ONLY thread (0,0,0,0) reads from currPtr — every
          // other thread does NOTHING. If kernel passes, currPtr OOB
          // is the issue for OTHER threads. If still crashes, fault
          // is elsewhere.
          if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 && triIdx != -1 && dbg) {
            U32 sample = *(volatile U32*)currPtr;
            U32* dbgU = (U32*)dbg;
            uintptr_t base = (uintptr_t)p.warpEmitGlobal;
            uintptr_t cp = (uintptr_t)currPtr;
            uintptr_t offset = cp - base;
            dbgU[1] = 1u;
            dbgU[2] = (U32)(offset & 0xFFFFFFFFu);
            dbgU[3] = (U32)(offset >> 32);
            dbgU[4] = (U32)((threadIdx.y << 16) | threadIdx.x);
            dbgU[5] = (U32)((lox << 16) | (loy & 0xFFFFu));
            dbgU[6] = (U32)((blockIdx.x << 16) | (blockIdx.z & 0xFFFFu));
            dbgU[7] = (U32)maxTileXInBin;
            dbgU[11] = sample;
          }
          return;
        }

        __syncthreads();

        //------------------------------------------------------------------------
        // Count.
        //------------------------------------------------------------------------

        // Tile per thread: Initialize prefix sums.

        for (int tileInBin_base = 0; tileInBin_base < CR_BIN_SQR;
             tileInBin_base += CR_COARSE_WARPS * 32) {
          int tileInBin = tileInBin_base + thrInBlock;
          bool act = (tileInBin < CR_BIN_SQR);
          U32 actMask = __ballot_sync(~0u, act);
          if (act) {
            // Compute prefix sum of emits over warps.

            U8 *srcPtr = (U8 *)&s_warpEmitMask[0][tileInBin];
            U8 *dstPtr = (U8 *)&s_warpEmitPrefixSum[0][tileInBin];
            int tileEmits = 0;
            for (int i = 0; i < CR_COARSE_WARPS; i++) {
              tileEmits += __popc(*(U32 *)srcPtr);
              *(U32 *)dstPtr = tileEmits;
              srcPtr += (CR_BIN_SQR + 1) * 4;
              dstPtr += (CR_BIN_SQR + 1) * 4;
            }

            // Determine the number of segments to allocate.

            int spaceLeft =
                -s_tileStreamCurrOfs[tileInBin] & (CR_TILE_SEG_SIZE - 1);
            int tileAllocs = (tileEmits - spaceLeft + CR_TILE_SEG_SIZE - 1) >>
                             CR_TILE_SEG_LOG2;
            volatile U32 *v = &s_tileEmitPrefixSum[tileInBin + 1];

            // All counters within the warp are small => compute prefix sum
            // using ballot.

            if (!__any_sync(actMask, tileEmits >= 2)) {
              U32 m = getLaneMaskLe();
              *v = (__popc(__ballot_sync(actMask, tileEmits & 1) & m)
                    << emitShift) |
                   __popc(__ballot_sync(actMask, tileAllocs & 1) & m);
            }

            // Otherwise => scan-32 within the warp.

            else {
              U32 sum = (tileEmits << emitShift) | tileAllocs;
              *v = sum;
              __syncwarp(actMask);
              if (threadIdx.x >= 1)
                sum += v[-1];
              __syncwarp(actMask);
              *v = sum;
              __syncwarp(actMask);
              if (threadIdx.x >= 2)
                sum += v[-2];
              __syncwarp(actMask);
              *v = sum;
              __syncwarp(actMask);
              if (threadIdx.x >= 4)
                sum += v[-4];
              __syncwarp(actMask);
              *v = sum;
              __syncwarp(actMask);
              if (threadIdx.x >= 8)
                sum += v[-8];
              __syncwarp(actMask);
              *v = sum;
              __syncwarp(actMask);
              if (threadIdx.x >= 16)
                sum += v[-16];
              __syncwarp(actMask);
              *v = sum;
              __syncwarp(actMask);
            }
          }
        }

        // First warp: Scan-8.

        __syncthreads();

        bool scan8 = (thrInBlock < CR_BIN_SQR / 32);
        U32 scan8Mask = __ballot_sync(~0u, scan8);
        if (scan8) {
          int sum = s_tileEmitPrefixSum[(thrInBlock << 5) + 32];
          volatile U32 *v = &s_scanTemp[0][thrInBlock + 16];
          v[0] = sum;
          __syncwarp(scan8Mask);
#if (CR_BIN_SQR > 1 * 32)
          sum += v[-1];
          __syncwarp(scan8Mask);
          v[0] = sum;
          __syncwarp(scan8Mask);
#endif
#if (CR_BIN_SQR > 2 * 32)
          sum += v[-2];
          __syncwarp(scan8Mask);
          v[0] = sum;
          __syncwarp(scan8Mask);
#endif
#if (CR_BIN_SQR > 4 * 32)
          sum += v[-4];
          __syncwarp(scan8Mask);
          v[0] = sum;
          __syncwarp(scan8Mask);
#endif
        }

        __syncthreads();

        // Tile per thread: Finalize prefix sums.
        // Single thread: Allocate segments.

        for (int tileInBin = thrInBlock; tileInBin < CR_BIN_SQR;
             tileInBin += CR_COARSE_WARPS * 32) {
          int sum = s_tileEmitPrefixSum[tileInBin + 1] +
                    s_scanTemp[0][(tileInBin >> 5) + 15];
          int numEmits = sum >> emitShift;
          int numAllocs = sum & ((1 << emitShift) - 1);
          s_tileEmitPrefixSum[tileInBin + 1] = numEmits;
          s_tileAllocPrefixSum[tileInBin + 1] = numAllocs;

          if (tileInBin == CR_BIN_SQR - 1 && numAllocs != 0) {
            int t = atomicAdd(&atomics.numTileSegs, numAllocs);
            s_firstAllocSeg = (t + numAllocs <= p.maxTileSegs) ? t : 0;
          }
        }

        __syncthreads();
        int firstAllocSeg = s_firstAllocSeg;
        int totalEmits = s_tileEmitPrefixSum[CR_BIN_SQR];
        int totalAllocs = s_tileAllocPrefixSum[CR_BIN_SQR];

        //------------------------------------------------------------------------
        // Emit.
        //------------------------------------------------------------------------

        // Emit per thread: Write triangle index to globalmem.

        for (int emitInBin = thrInBlock; emitInBin < totalEmits;
             emitInBin += CR_COARSE_WARPS * 32) {
          // Find tile in bin.

          U8 *tileBase = (U8 *)&s_tileEmitPrefixSum[0];
          U8 *tilePtr = tileBase;
          U8 *ptr;

#if (CR_BIN_SQR > 128)
          ptr = tilePtr + 0x80 * 4;
          if (emitInBin >= *(U32 *)ptr)
            tilePtr = ptr;
#endif
#if (CR_BIN_SQR > 64)
          ptr = tilePtr + 0x40 * 4;
          if (emitInBin >= *(U32 *)ptr)
            tilePtr = ptr;
#endif
#if (CR_BIN_SQR > 32)
          ptr = tilePtr + 0x20 * 4;
          if (emitInBin >= *(U32 *)ptr)
            tilePtr = ptr;
#endif
#if (CR_BIN_SQR > 16)
          ptr = tilePtr + 0x10 * 4;
          if (emitInBin >= *(U32 *)ptr)
            tilePtr = ptr;
#endif
#if (CR_BIN_SQR > 8)
          ptr = tilePtr + 0x08 * 4;
          if (emitInBin >= *(U32 *)ptr)
            tilePtr = ptr;
#endif
#if (CR_BIN_SQR > 4)
          ptr = tilePtr + 0x04 * 4;
          if (emitInBin >= *(U32 *)ptr)
            tilePtr = ptr;
#endif
#if (CR_BIN_SQR > 2)
          ptr = tilePtr + 0x02 * 4;
          if (emitInBin >= *(U32 *)ptr)
            tilePtr = ptr;
#endif
#if (CR_BIN_SQR > 1)
          ptr = tilePtr + 0x01 * 4;
          if (emitInBin >= *(U32 *)ptr)
            tilePtr = ptr;
#endif

          int tileInBin = (tilePtr - tileBase) >> 2;
          int emitInTile = emitInBin - *(U32 *)tilePtr;

          // Find warp in tile.

          int warpStep = (CR_BIN_SQR + 1) * 4;
          U8 *warpBase = (U8 *)&s_warpEmitPrefixSum[0][tileInBin] - warpStep;
          U8 *warpPtr = warpBase;

#if (CR_COARSE_WARPS > 8)
          ptr = warpPtr + 0x08 * warpStep;
          if (emitInTile >= *(U32 *)ptr)
            warpPtr = ptr;
#endif
#if (CR_COARSE_WARPS > 4)
          ptr = warpPtr + 0x04 * warpStep;
          if (emitInTile >= *(U32 *)ptr)
            warpPtr = ptr;
#endif
#if (CR_COARSE_WARPS > 2)
          ptr = warpPtr + 0x02 * warpStep;
          if (emitInTile >= *(U32 *)ptr)
            warpPtr = ptr;
#endif
#if (CR_COARSE_WARPS > 1)
          ptr = warpPtr + 0x01 * warpStep;
          if (emitInTile >= *(U32 *)ptr)
            warpPtr = ptr;
#endif

          int warpInTile = (warpPtr - warpBase) >> (CR_BIN_LOG2 * 2 + 2);
          U32 emitMask =
              *(U32 *)(warpPtr + warpStep +
                       ((U8 *)s_warpEmitMask - (U8 *)s_warpEmitPrefixSum));
          int emitInWarp =
              emitInTile - *(U32 *)(warpPtr + warpStep) + __popc(emitMask);

          // Find thread in warp.

          int threadInWarp = 0;
          int pop = __popc(emitMask & 0xFFFF);
          bool pred = (emitInWarp >= pop);
          if (pred)
            emitInWarp -= pop;
          if (pred)
            emitMask >>= 0x10;
          if (pred)
            threadInWarp += 0x10;

          pop = __popc(emitMask & 0xFF);
          pred = (emitInWarp >= pop);
          if (pred)
            emitInWarp -= pop;
          if (pred)
            emitMask >>= 0x08;
          if (pred)
            threadInWarp += 0x08;

          pop = __popc(emitMask & 0xF);
          pred = (emitInWarp >= pop);
          if (pred)
            emitInWarp -= pop;
          if (pred)
            emitMask >>= 0x04;
          if (pred)
            threadInWarp += 0x04;

          pop = __popc(emitMask & 0x3);
          pred = (emitInWarp >= pop);
          if (pred)
            emitInWarp -= pop;
          if (pred)
            emitMask >>= 0x02;
          if (pred)
            threadInWarp += 0x02;

          if (emitInWarp >= (emitMask & 1))
            threadInWarp++;

          // Figure out where to write.

          int currOfs = s_tileStreamCurrOfs[tileInBin];
          int spaceLeft = -currOfs & (CR_TILE_SEG_SIZE - 1);
          int outOfs = emitInTile;

          if (outOfs < spaceLeft)
            outOfs += currOfs;
          else {
            int allocLo = firstAllocSeg + s_tileAllocPrefixSum[tileInBin];
            outOfs += (allocLo << CR_TILE_SEG_LOG2) - spaceLeft;
          }

          // Write.

          int queueIdx = warpInTile * 32 + threadInWarp;
          int triIdx = s_triQueue[(triQueueReadPos + queueIdx) &
                                  (CR_COARSE_QUEUE_SIZE - 1)];

          // Test 42: bounds check tileSegData write.
          if (outOfs >= 0 && outOfs < maxTileSegOfs)
            tileSegData[outOfs] = triIdx;
#if CR_DEBUG_OOB
          else if (atomicAdd((S32*)&s_oobCount, 1) < 4)
            printf("[OOB-A] outOfs=%d max=%d tile=%d emit=%d currOfs=%d spaceLeft=%d allocLo=%d firstAlloc=%d\n",
                   outOfs, maxTileSegOfs, tileInBin, emitInTile, currOfs, spaceLeft,
                   firstAllocSeg + (int)s_tileAllocPrefixSum[tileInBin], firstAllocSeg);
#endif
        }

        //------------------------------------------------------------------------
        // Patch.
        //------------------------------------------------------------------------

        // Allocated segment per thread: Initialize next-pointer and count.

        for (int i = CR_COARSE_WARPS * 32 - 1 - thrInBlock; i < totalAllocs;
             i += CR_COARSE_WARPS * 32) {
          int segIdx = firstAllocSeg + i;
          // Test 42: bounds check tileSegNext/Count write.
          if (segIdx >= 0 && segIdx < p.maxTileSegs) {
            tileSegNext[segIdx] = segIdx + 1;
            tileSegCount[segIdx] = CR_TILE_SEG_SIZE;
          }
#if CR_DEBUG_OOB
          else if (atomicAdd((S32*)&s_oobCount, 1) < 4)
            printf("[OOB-B] segIdx=%d max=%d i=%d firstAlloc=%d totalAllocs=%d\n",
                   segIdx, p.maxTileSegs, i, firstAllocSeg, totalAllocs);
#endif
        }

        // Tile per thread: Fix previous segment's next-pointer and update
        // s_tileStreamCurrOfs.

        __syncthreads();
        for (int tileInBin = CR_COARSE_WARPS * 32 - 1 - thrInBlock;
             tileInBin < CR_BIN_SQR; tileInBin += CR_COARSE_WARPS * 32) {
          int oldOfs = s_tileStreamCurrOfs[tileInBin];
          int newOfs =
              oldOfs + s_warpEmitPrefixSum[CR_COARSE_WARPS - 1][tileInBin];
          int allocLo = s_tileAllocPrefixSum[tileInBin];
          int allocHi = s_tileAllocPrefixSum[tileInBin + 1];

          if (allocLo != allocHi) {
            // Test 42: bounds check nextPtr write.
            if (oldOfs <= 0) {
              tileFirstSeg[binTileIdx + globalTileIdx(tileInBin, p.widthTiles)] = firstAllocSeg + allocLo;
            } else {
              int nextSegIdx = (oldOfs - 1) >> CR_TILE_SEG_LOG2;
              if (nextSegIdx >= 0 && nextSegIdx < p.maxTileSegs)
                tileSegNext[nextSegIdx] = firstAllocSeg + allocLo;
#if CR_DEBUG_OOB
              else if (atomicAdd((S32*)&s_oobCount, 1) < 4)
                printf("[OOB-C] nextSegIdx=%d max=%d oldOfs=%d tileInBin=%d allocLo=%d\n",
                       nextSegIdx, p.maxTileSegs, oldOfs, tileInBin, allocLo);
#endif
            }

            newOfs--;
            newOfs &= CR_TILE_SEG_SIZE - 1;
            newOfs |= (firstAllocSeg + allocHi - 1) << CR_TILE_SEG_LOG2;
            newOfs++;
          }
          s_tileStreamCurrOfs[tileInBin] = newOfs;
        }

        // Advance queue read pointer.
        // Queue became empty => bin done.

        triQueueReadPos += CR_COARSE_WARPS * 32;
      } while (triQueueReadPos < triQueueWritePos);

    // Tile per thread: Fix next-pointer and count of the last segment.
    // 32 tiles per warp: Count active tiles.

    __syncthreads();

    for (int tileInBin_base = 0; tileInBin_base < CR_BIN_SQR;
         tileInBin_base += CR_COARSE_WARPS * 32) {
      int tileInBin = tileInBin_base + thrInBlock;
      bool act = (tileInBin < CR_BIN_SQR);
      U32 actMask = __ballot_sync(~0u, act);
      if (act) {
        int tileX = tileInBin & (CR_BIN_SIZE - 1);
        int tileY = tileInBin >> CR_BIN_LOG2;
        bool force =
            (p.deferredClear & tileX <= maxTileXInBin & tileY <= maxTileYInBin);

        int ofs = s_tileStreamCurrOfs[tileInBin];
        int segIdx = (ofs - 1) >> CR_TILE_SEG_LOG2;
        int segCount = ofs & (CR_TILE_SEG_SIZE - 1);

        if (ofs >= 0) {
          // Test 42: bounds check tileSegNext finalize write.
          if (segIdx >= 0 && segIdx < p.maxTileSegs)
            tileSegNext[segIdx] = -1;
#if CR_DEBUG_OOB
          else if (atomicAdd((S32*)&s_oobCount, 1) < 4)
            printf("[OOB-D] segIdx=%d max=%d ofs=%d tileInBin=%d\n",
                   segIdx, p.maxTileSegs, ofs, tileInBin);
#endif
        } else if (force) {
          s_tileStreamCurrOfs[tileInBin] = 0;
          tileFirstSeg[binTileIdx + tileX + tileY * p.widthTiles] = -1;
        }

        if (segCount != 0) {
          // Test 42: bounds check tileSegCount finalize write.
          if (segIdx >= 0 && segIdx < p.maxTileSegs)
            tileSegCount[segIdx] = segCount;
#if CR_DEBUG_OOB
          else if (atomicAdd((S32*)&s_oobCount, 1) < 4)
            printf("[OOB-E] segIdx=%d max=%d ofs=%d segCount=%d tileInBin=%d\n",
                   segIdx, p.maxTileSegs, ofs, segCount, tileInBin);
#endif
        }

        U32 res = __ballot_sync(actMask, ofs >= 0 | force);
        if (threadIdx.x == 0)
          s_scanTemp[0][(tileInBin >> 5) + 16] = __popc(res);
      }
    }

    // First warp: Scan-8.
    // One thread: Allocate space for active tiles.

    __syncthreads();

    bool scan8 = (thrInBlock < CR_BIN_SQR / 32);
    U32 scan8Mask = __ballot_sync(~0u, scan8);
    if (scan8) {
      volatile U32 *v = &s_scanTemp[0][thrInBlock + 16];
      U32 sum = v[0];
#if (CR_BIN_SQR > 1 * 32)
      sum += v[-1];
      __syncwarp(scan8Mask);
      v[0] = sum;
      __syncwarp(scan8Mask);
#endif
#if (CR_BIN_SQR > 2 * 32)
      sum += v[-2];
      __syncwarp(scan8Mask);
      v[0] = sum;
      __syncwarp(scan8Mask);
#endif
#if (CR_BIN_SQR > 4 * 32)
      sum += v[-4];
      __syncwarp(scan8Mask);
      v[0] = sum;
      __syncwarp(scan8Mask);
#endif

      if (thrInBlock == CR_BIN_SQR / 32 - 1)
        s_firstActiveIdx = atomicAdd(&atomics.numActiveTiles, sum);
    }

    // Tile per thread: Output active tiles.

    __syncthreads();

    for (int tileInBin_base = 0; tileInBin_base < CR_BIN_SQR;
         tileInBin_base += CR_COARSE_WARPS * 32) {
      int tileInBin = tileInBin_base + thrInBlock;
      bool act =
          (tileInBin < CR_BIN_SQR) && (s_tileStreamCurrOfs[tileInBin] >= 0);
      U32 actMask = __ballot_sync(~0u, act);
      if (act) {
        int activeIdx = s_firstActiveIdx;
        activeIdx += s_scanTemp[0][(tileInBin >> 5) + 15];
        activeIdx += __popc(actMask & getLaneMaskLt());
        activeTiles[activeIdx] =
            binTileIdx + globalTileIdx(tileInBin, p.widthTiles);
      }
    }
  }
}

//------------------------------------------------------------------------
