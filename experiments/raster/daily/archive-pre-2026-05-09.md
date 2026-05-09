# Legacy archive — investigation log prior to 2026-05-09

This file is the original "Active Investigation" section from CLAUDE.md,
preserved verbatim. It captures Tests 1-67 plus the methodology pivots
(harness creation, stale-binary trap, VSCode auto-revert defenses).
Subsequent days have their own file in this directory.

For the distilled bug list see [../findings.md](../findings.md). For the
failed workarounds table see [../workarounds.md](../workarounds.md).

---

## Active Investigation: GPU Memory Fault in nvdiffrast HIP Rasterizer

### Problem
Running app.py with real TRELLIS meshes (~290K-320K triangles) causes
"Memory access fault by GPU node-1 ... Reason: Page not present or supervisor privilege"
crashes during the mesh video preview phase (120-frame render loop). The crash
kills the process and sometimes crashes the entire system (requires reboot).

### nvdiffrast Rasterization Pipeline (4 kernels, run in sequence)
1. **triangleSetupKernel** - Decomposes triangles into subtriangles, writes triHeader/triSubtris
2. **binRasterKernel** - Assigns subtriangles to screen-space bins, builds binSeg linked lists
3. **coarseRasterKernel** - Reads bin data, assigns triangles to tiles, builds tileSeg linked lists
4. **fineRasterKernel** - Reads tile data, rasterizes individual pixels

### Key Insight: AMD Async Fault Reporting
On AMD GPUs with ROCm, the HSA runtime reports GPU memory faults asynchronously.
hipDeviceSynchronize() waits for kernel completion but the MMU page fault notification
can arrive AFTER the sync returns. The crash log order does NOT indicate fault origin.

### CRParams Struct Size
~680 bytes total. Key fields: 16 void* GPU buffer pointers, CRImageParams[32] embedded
array (384 bytes), plus ~30 S32/F32 scalar fields. Well within AMD kernarg limits.

### __launch_bounds__ for Each Kernel
- triangleSetupKernel: __launch_bounds__(64, 8) -- 64 threads, up to 8 blocks/SM
- binRasterKernel: __launch_bounds__(512, 1) -- 512 threads, 1 block/SM
- coarseRasterKernel: __launch_bounds__(512, 1) -- 512 threads, 1 block/SM
- fineRasterKernel: __launch_bounds__(?, 1)
Different launch bounds affect compiler register allocation. triangleSetup (64 threads,
8 blocks) is very different from bin/coarse (512 threads, 1 block).

### nvdiffrast Module Layout (important)
Both mesh preview AND GLB texture baking use the SAME custom HIP rasterizer:
- Custom extension builds as `_nvdiffrast_c` (same module name as standard nvdiffrast)
- `RastContext(backend='gl')` in bake_texture still goes through `rasterize_fwd_cuda` (HIP)
- There is NO separate GL backend in our custom build

### Tests 1-19 Summary (DO NOT REPEAT)

| Test | Config | Result | Key Conclusion |
|------|--------|--------|----------------|
| 1 | Bounds checking in coarseRasterSimple | CRASH | OOB reads not the cause |
| 2 | Infinite loop protection in coarseRasterSimple | CRASH | Circular lists not the cause |
| 3 | No-op coarseRaster (kept tileFirstSeg init) | CRASH | Not from reading bin/tile data |
| 4 | Absolute no-op coarseRaster (just return;) | CRASH | Crash not in coarseRaster at all |
| 5 | No-op binRaster + no-op coarseRaster | PASS mesh, CRASH GLB | triSetup alone OK for mesh preview |
| 6 | All 3 setup no-op, fineRaster ACTIVE | CRASH | fineRaster reading garbage or other cause |
| 7 | Skip entire launchStages() | PASS | Crash is inside launchStages |
| 8 | 3 no-op kernels + hipDeviceSynchronize, no fineRaster | CRASH | Not fineRaster; no-ops still crash |
| 9 | Memcpy ONLY, no kernel launches | PASS | Buffers/memcpy are fine |
| 10 | Memcpy + 1 kernel only (coarseRaster) | PASS | Single kernel launch is fine |
| 11 | 3 no-op kernels, no sync between, 1 sync at end | CRASH | Multiple kernels = crash |
| 12 | SAME kernel (coarseRaster) launched 3x | PASS | Same kernel 3x is fine |
| 13 | 2 different kernels: triSetup + coarseRaster | PASS | 2 different kernels OK |
| 14 | 2 different kernels: binRaster + coarseRaster | PASS | 2 different kernels OK |
| 15 | All 3 different kernels (with __shared__) | CRASH | 3 different kernels = crash |
| 16 | All 3 kernels, __shared__ stripped via #if 0 | CRASH | Not __shared__ memory |
| 17 | 3 dummy kernels with NO arguments | PASS | Not fundamental to 3 dispatches |
| 18 | 3 real kernels with ZEROED CRParams | PASS | Not about CRParams size |
| 19 | 3 real kernels + REAL CRParams + stderr delay | PASS | Stderr delay prevents crash |

Key pattern from Tests 1-19:
- 1-2 different kernels with real CRParams: always PASS
- 3+ different kernels with real CRParams: always CRASH
- 3+ different kernels with zeroed CRParams: PASS
- 3+ different kernels with real CRParams + stderr delay: PASS

### Test 20: Full pipeline with all 4 real kernels + hipStreamSynchronize workaround
- Result: Gaussian render PASSED (120it at 141 it/s), mesh render CRASHED (frame 0)
- Crash address: 0x711358bff000 (Page not present)

### Test 21: coarseRaster no-op, triSetup+binRaster+fineRaster real
- Result: CRASHED (system crash, had to reboot)
- Likely fineRaster reading garbage tile data from no-op coarseRaster

### Test 22: triSetup+binRaster real, coarseRaster+fineRaster no-op
- Result: CRASHED (system crash, had to reboot)
- Crash address: 0x76ea04bff000

### Test 23: triSetup real, binRaster+coarseRaster+fineRaster ALL no-op
- Result: CRASHED - crash address 0x76ea04bff000
- This was unexpected -- triSetup alone should be fine (Test 5 passed)

### Test 24: ALL 4 kernels no-op (return;) + hipStreamSynchronize between each
- Result: CRASHED - crash address 0x70d6b79ff000
- **CRITICAL FINDING**: Even with ALL kernels as no-ops + hipStreamSynchronize
  between each dispatch, the mesh render still crashes on frame 0.
- This means: hipStreamSynchronize does NOT fix the kernarg race condition.
  Only the stderr delay (Test 19) worked.
- The crash is NOT from kernel code -- it's from the kernel dispatch mechanism itself.

### Revised Root Cause Analysis

The hipStreamSynchronize workaround does NOT fix the kernarg race condition.
Re-examining what actually worked:

- Test 19 (PASSED): 3 no-op kernels + stderr output (5 lines of cerr with pointer
  values BEFORE kernel launches, 1 line AFTER sync) + real CRParams + single
  hipStreamSynchronize at end (NO sync between individual dispatches)
- Tests 8, 11, 15, 16, 20-24 (ALL CRASHED): Various configs with hipStreamSynchronize
  or hipDeviceSynchronize between kernels, but NO stderr output

The ONLY thing that worked was stderr output. This suggests:
- The HIP runtime on RDNA3/ROCm 6.4.2 has a bug where it lazily reads kernel
  argument data from the host pointer (&p) AFTER hipLaunchKernel returns
- CRParams p is a stack-allocated local variable in launchStages()
- Between calls to launchStages, the stack frame is destroyed/reused
- The stderr output adds enough host-side delay for the lazy copy to complete
- hipStreamSynchronize/hipDeviceSynchronize only wait for GPU work, not for the
  HIP runtime's internal host-side kernarg processing

### Test 25: static CRParams + all 4 no-op kernels + hipStreamSynchronize

- `static CRParams p;` instead of stack-allocated
- Result: CRASHED (system crash, reboot required)
- Conclusion: Static storage alone does not fix it

### Test 26: static CRParams + usleep(1) + all 4 no-op kernels + hipStreamSynchronize

- Added `usleep(1)` before first kernel launch
- Result: CRASHED (system crash, reboot required)
- Conclusion: 1 microsecond is not enough delay, OR the issue is not purely timing

### Test 27: 4 no-op kernels + stderr pointer dump (no sync between)

- static CRParams, stderr before launches, no hipStreamSynchronize between dispatches
- Result: CRASHED after 5 iterations (better than Test 24 which crashed frame 0)
- Crash address: 0x7a214d9ff000 = exactly 1 page before p.tileSegData (0x7a214da00000)
- Pointers stable across all 5 iterations (never go stale)
- Conclusion: stderr helps delay the crash but does NOT prevent it with 4 kernels

### Test 28: 3 no-op kernels + stderr (skip fineRaster), static CRParams

- Same as Test 27 but fineRaster skipped (only triSetup + binRaster + coarseRaster)
- Result: CRASHED on first iteration
- Crash address: 0x79a8eb3ff000 = exactly 1 page before p.tileSegData (0x79a8eb400000)
- Conclusion: static CRParams crashes even with 3 kernels + stderr. Test 19 used
  stack-allocated CRParams and PASSED, so static vs stack matters.

### Test 29: 3 no-op kernels + stderr, STACK CRParams (matching Test 19)

- Reverted to `CRParams p;` (stack-allocated, not static)
- 3 no-op kernels + stderr, fineRaster skipped
- Result: CRASHED after 4 iterations
- Crash address: 0x74b5b73ff000 = 1 page before p.tileSegData (0x74b5b7400000)
- Conclusion: Test 19 is NOT reproducible. Something else changed since then
  (coarseRasterImpl vs coarseRasterImplSimple, removed dummy kernels, etc.)

### Crash Address Pattern

Every crash hits exactly 1 page (4096 bytes) BEFORE p.tileSegData:

| Test | Fault Address | tileSegData | Iterations Before Crash |
|------|--------------|-------------|------------------------|
| 27 | 0x7a214d9ff000 | 0x7a214da00000 | 5 |
| 28 | 0x79a8eb3ff000 | 0x79a8eb400000 | 1 |
| 29 | 0x74b5b73ff000 | 0x74b5b7400000 | 4 |

The fault is always at `tileSegData - 0x1000`, the unmapped guard page before
the tileSegData allocation. This is 100% consistent -- not random corruption.

### Workarounds Tried and Failed

| Workaround | Test(s) | Result |
| --- | --- | --- |
| hipDeviceSynchronize between each kernel | 8 | CRASH |
| hipStreamSynchronize between each kernel | 20-24 | CRASH |
| static CRParams (persistent memory) | 25, 28 | CRASH |
| static CRParams + usleep(1) | 26 | CRASH |
| static CRParams + stderr + 4 kernels | 27 | CRASH (delayed) |
| stack CRParams + stderr + 3 kernels | 29 | CRASH (delayed) |

### Test 19 No Longer Reproducible

Test 19 passed but cannot be reproduced. Code changes since Test 19:
- coarseRasterKernel now calls coarseRasterImpl (was coarseRasterImplSimple)
- CoarseRasterSimple.inl no longer included
- Dummy kernels (dummyKernelA/B/C) removed from .hip file
- `#include <iostream>` and `#include <unistd.h>` added to RasterImpl.cpp

Test 19 may have been a fluke, or one of these changes affects the compiled
binary in a way that triggers the race more reliably.

### Test 30: CRParams by POINTER (8-byte kernarg) -- ALL 4 KERNELS

- Changed kernel signatures from `const CRParams p` to `const CRParams* p`
- hipMalloc'd device CRParams, hipMemcpy before launches, pass pointer
- Kernarg reduced from ~680 bytes to 8 bytes (a single pointer)
- All 4 no-op kernels dispatched, no sync between, no stderr delay
- Result: **PASSED** -- Gaussian 120it @ 144 it/s, mesh 120it @ 412 it/s
- Conclusion: **CONFIRMED -- HIP kernarg buffer bug for large structs on RDNA3**.
  Passing CRParams by pointer completely eliminates the GPU memory fault.

### Root Cause (CONFIRMED)

The HIP runtime on RDNA3 (gfx1101) with ROCm 6.4.2 has a bug in kernarg buffer
management when dispatching multiple different kernels with large (~680 byte)
argument structs on the same stream. The kernarg data gets corrupted, causing GPU
page faults at addresses near the pointer values embedded in the struct
(consistently 1 page before tileSegData).

**Fix**: Pass CRParams via device memory pointer (8-byte kernarg) instead of by
value (~680 bytes). This bypasses the buggy kernarg buffer mechanism entirely.

### Current File State (as of Test 30)

- `TriangleSetup.inl`: Has `return;` at top (no-op)
- `BinRaster.inl`: Has `return;` at top (no-op)
- `CoarseRaster.inl`: Has `return;` at top (no-op)
- `FineRaster.inl`: Has `return;` at top (no-op)
- `CoarseRasterSimple.inl`: Dead file, no longer included
- `RasterImpl_kernel.hip`: 4 kernel entry points taking `const CRParams*` (pointer)
- `RasterImpl.cpp`: All 4 kernels dispatched, CRParams hipMalloc'd + hipMemcpy'd,
  `void *args[] = {&d_crParams}`, no sync between dispatches, single sync at end.
  Forward declarations updated to `const CRParams*`.
- `postprocessing_utils.py`: fill_holes=fill_holes (re-enabled)
- Rebuild: `cd extensions/nvdiffrast-hip && pip install . --no-build-isolation`

### Test 31: ALL 4 REAL kernels + pass-by-pointer CRParams
- Restored real kernel code (removed `return;` from all 4 .inl files)
- All 4 kernels running real code with pointer-based CRParams
- Result: Gaussian render PASSED, mesh render PASSED (120 frames, no crash)
- GLB extraction: FAILED (same error as before -- investigating)
- Conclusion: Pass-by-pointer fix works for mesh preview with real kernels.
  GLB extraction failure needs investigation (different code path through utils3d wrapper).

### Current File State (as of Test 31)
- `TriangleSetup.inl`: Real kernel code (no-op removed)
- `BinRaster.inl`: Real kernel code (no-op removed)
- `CoarseRaster.inl`: Real kernel code (no-op removed, AMD sort fix at ~line 98)
- `FineRaster.inl`: Real kernel code (no-op removed)
- `CoarseRasterSimple.inl`: Dead file, no longer included
- `RasterImpl_kernel.hip`: 4 kernel entry points taking `const CRParams*` (pointer)
- `RasterImpl.cpp`: All 4 kernels dispatched, CRParams hipMalloc'd + hipMemcpy'd,
  `void *args[] = {&d_crParams}`, no sync between dispatches, single sync at end.
  Forward declarations updated to `const CRParams*`.
- `postprocessing_utils.py`: fill_holes=fill_holes (re-enabled)
- Rebuild: `cd extensions/nvdiffrast-hip && pip install . --no-build-isolation`

### Test 32: Diagnostics -- binRaster zero output, triangleSetup zero triSubtris

Added diagnostics to drawTriangles (atomics readback) and launchStages (CRParams
verification + triSubtris readback + triHeader readback).

Results (consistent across all draw calls, all instances):
- CRParams correctly on device: widthPixelsVp=1024, heightPixelsVp=1024, widthBins=8,
  heightBins=8, numBins=64, numTris=313928, maxSubtris=318024, instanceMode=1,
  xs=1, ys=1, xo=0, yo=0
- All GPU buffer pointers valid and stable across calls
- subtris == tris (313928) -- this is the pre-initialized host value, NOT proof triangleSetup ran
- binSegs=0, tileSegs=0, activeTiles=0 on EVERY call (120 mesh preview frames + 1000 GLB views)
- triSubtris[0..99] ALL ZERO -- every triangle appears culled
- triHeader[0] ALL ZERO -- no screen-space triangle data written

Conclusion: triangleSetup is not producing visible triangles. binRaster correctly
produces zero output because its input (triSubtris) is all zeros.

### Test 33: Pre-fill triSubtris with 0xAA marker

Pre-filled triSubtris with hipMemset(0xAA) before triangleSetup kernel launch.
After kernel completion, triSubtris values are still 0xAA (170 decimal).

**CRITICAL FINDING**: triangleSetup kernel NEVER WRITES to triSubtris. The kernel
launches without error, hipStreamSynchronize returns success, but the GPU code
produces zero side effects. The 0xAA marker values are untouched.

This means one of:
1. The kernel entry point `*p` dereference (680-byte copy from device global memory
   to by-value parameter) fails silently on AMD RDNA3
2. The compiler optimizes away the kernel body
3. The kernel launches but the GPU silently aborts execution

The impl functions take CRParams by VALUE: `triangleSetupImpl(const CRParams p)`,
`binRasterImpl(const CRParams p)`. The kernel entry points do `impl(*p)` which
triggers a 680-byte global memory read + copy. This worked on NVIDIA because the
original code passed CRParams by value as a kernel argument (~680 bytes in the
constant/argument buffer). With our pass-by-pointer change, the struct is now in
device GLOBAL memory, and the by-value copy reads from global memory instead.

**Proposed fix**: Change impl functions to take `const CRParams&` (by reference)
instead of `const CRParams` (by value). This avoids the 680-byte copy and reads
fields directly from device memory via the pointer.

### Test 34: by-reference impl + diagnostic write in kernel entry point

- Changed all 4 impl functions to `const CRParams& p` (by reference)
- Added `((unsigned char*)p->triSubtris)[0] = 42;` in triangleSetupKernel entry
- Result: CRASHED (system crash, reboot required)
- Conclusion: Kernel cannot read `p->triSubtris` from device global memory.
  The pointer dereference `p->triSubtris` reads garbage, causing a fault when
  writing to the garbage address. This explains Tests 31-33: kernel reads
  `p->numTriangles` as 0, so all threads exit via `taskIdx >= 0` early return,
  never touching triSubtris.

### Root Cause (REVISED)

The kernel cannot read from `d_crParams` in device global memory. The synchronous
`hipMemcpy(d_crParams, &p, ...)` runs on the NULL stream, but kernels launch on a
different `stream`. On AMD/HIP, the NULL stream may not properly synchronize with
other streams, so the kernel reads uninitialized device memory.

The host-side CRParams readback (after hipStreamSynchronize) shows correct values
because hipMemcpy eventually completes -- just not before the kernel reads the data.

**Fix**: Use `hipMemcpyAsync(d_crParams, &p, sizeof(CRParams), hipMemcpyHostToDevice, stream)`
on the same stream as kernel launches. This guarantees the copy completes before
kernels run.

### Test 35: hipMemcpyAsync on same stream + by-reference impl functions

- Changed `hipMemcpy(d_crParams, &p, ...)` to `hipMemcpyAsync(..., stream)`
- Removed diagnostic write from triangleSetupKernel entry point
- Result: triSubtris shows REAL DATA (0, 1 values, 37/100 visible)
  binSegs=565, tileSegs=12527, activeTiles=16384 -- full pipeline producing output!
  triHeader has real screen-space data. **STREAM FIX WORKS.**
  CRASHED after frame 1: fault at 0x703b363ff000 (in real kernel code, not CRParams)
- Constructor shows coarseBlocks=1 fineBlocks=1 (was 4/3 before by-reference change)
- Conclusion: CRParams delivery is FIXED. The crash is now in the real rasterization
  kernels (coarseRaster or fineRaster), a separate bug from the CRParams issue.

### Two Bugs Found and Fixed

1. **HIP kernarg buffer corruption** (Test 30): Passing ~680-byte CRParams by value
   corrupts kernarg data on RDNA3. Fix: pass CRParams* pointer (8 bytes).
2. **Cross-stream ordering** (Test 35): Synchronous `hipMemcpy` on NULL stream does
   not synchronize with kernel launches on a non-default stream on AMD/HIP.
   Fix: use `hipMemcpyAsync` on the same stream as kernel launches.

### Current File State (as of Test 35)
- All .inl files: Real kernel code, `const CRParams& p` (by reference)
- `RasterImpl_kernel.hip`: 4 kernel entry points taking `const CRParams*` (clean)
- `RasterImpl.cpp`: Static d_crParams, `hipMemcpyAsync` on same stream (FIXED),
  atomics + CRParams diagnostics, triSubtris pre-fill + readback diagnostic

### Test 36: fineRaster no-op, triSetup+binRaster+coarseRaster real

- Added `return;` at top of fineRasterImpl
- Added tileSegData to diagnostic pointer output
- Result: CRASHED (system crash, reboot required)
- Conclusion: Crash is NOT in fineRaster. It's in triSetup, binRaster, or coarseRaster.

### Test 37: triSetup+binRaster real, coarseRaster+fineRaster no-op

- Added `return;` at top of coarseRasterImpl and fineRasterImpl
- Result: Ran 5 frames with real data (binSegs ~565, visible triangles ~33-60%)
  then crashed: fault at 0x747f4b1ff000 = tileSegData (0x747f4b200000) - 0x1000
  Process crashed but NOT a system crash (user reported PC stayed up)
- Conclusion: Crash is NOT from any kernel's buffer access (triSetup and binRaster
  don't access tileSegData). The crash is from the kernel dispatch mechanism or
  from CRParams containing the tileSegData pointer. Even with 8-byte pointer args,
  dispatching 4 different kernels (2 real + 2 no-op) still causes the fault after
  several iterations.

### Test 38: Only triSetup+binRaster dispatched (coarse+fine launches removed)

- Commented out hipLaunchKernel for coarseRaster and fineRaster entirely
- Result: **PASSED** -- Gaussian 120it @ 141 it/s, mesh 120+ frames no crash
  binSegs increasing across frames (602-757), tileSegs=0 (expected, coarse not running)
- Conclusion: **2 dispatches per frame is stable. 4 dispatches causes the crash.**
  The multi-dispatch bug persists even with 8-byte pointer args.

### Three Bugs Found

1. **HIP kernarg buffer corruption** (Test 30): Large (~680 byte) by-value kernargs
   corrupt on RDNA3. Fix: pass CRParams* pointer (8 bytes).
2. **Cross-stream ordering** (Test 35): Synchronous `hipMemcpy` on NULL stream does
   not synchronize with kernel launches on a non-default stream on AMD/HIP.
   Fix: use `hipMemcpyAsync` on the same stream.
3. **Multi-dispatch kernarg corruption** (Test 38): Dispatching 4+ different kernels
   on the same stream corrupts the kernarg pool after several iterations, causing
   page faults at tileSegData - 0x1000. Fix: split dispatches into batches of 2
   with hipStreamSynchronize between batches.

### Test 39: All 4 real kernels + hipStreamSynchronize between dispatch pairs

- All 4 kernels running real code, hipStreamSynchronize between binRaster and coarseRaster
- Dispatch pattern: triSetup + binRaster, sync, coarseRaster + fineRaster
- Result: CRASHED after 3 frames. Pipeline produced real output (binSegs=510-522,
  tileSegs=11123-11661, activeTiles=16384) before crash at 0x7fd10f1ff000 =
  tileSegData (0x7fd10f200000) - 0x1000
- Conclusion: hipStreamSynchronize between pairs does NOT fix the multi-dispatch bug.
  Pipeline is functional (real rasterization output) but crashes after a few iterations.

### Crash Address Pattern (Updated)

| Test | Fault Address | tileSegData | Iterations Before Crash |
|------|--------------|-------------|------------------------|
| 27 | 0x7a214d9ff000 | 0x7a214da00000 | 5 |
| 28 | 0x79a8eb3ff000 | 0x79a8eb400000 | 1 |
| 29 | 0x74b5b73ff000 | 0x74b5b7400000 | 4 |
| 35 | 0x703b363ff000 | (not logged) | 1 |
| 37 | 0x747f4b1ff000 | 0x747f4b200000 | 5 |
| 39 | 0x7fd10f1ff000 | 0x7fd10f200000 | 3 |

### Workarounds Tried and Failed (Updated)

| Workaround | Test(s) | Result |
| --- | --- | --- |
| hipDeviceSynchronize between each kernel | 8 | CRASH |
| hipStreamSynchronize between each kernel | 20-24 | CRASH |
| static CRParams (persistent memory) | 25, 28 | CRASH |
| static CRParams + usleep(1) | 26 | CRASH |
| static CRParams + stderr + 4 kernels | 27 | CRASH (delayed) |
| stack CRParams + stderr + 3 kernels | 29 | CRASH (delayed) |
| hipStreamSynchronize between dispatch pairs | 39 | CRASH after 3 frames |

### Test 40: hipStreamSynchronize after EVERY kernel dispatch

- All 4 real kernels, hipStreamSynchronize after each (1 kernel per batch)
- Each kernel fully completes before the next one launches
- Result: CRASHED (system crash, reboot required)
- Conclusion: **Crash is in the KERNEL CODE itself, not kernarg corruption.**
  Each kernel runs fully isolated with a sync barrier. The crash must be an
  out-of-bounds memory access in coarseRaster or fineRaster (triSetup + binRaster
  alone are stable per Test 38).

### Revised Analysis

Tests 8, 11, 15-16, 20-29 attributed crashes to kernarg corruption. But Test 40
proves the crash persists even with full sync after every dispatch. The earlier
tests were also hitting this kernel code bug -- the "multi-dispatch kernarg
corruption" theory (bug #3) was wrong. The real issue is an OOB access in the
coarseRaster or fineRaster kernel code on AMD/RDNA3.

The tileSegData - 0x1000 crash address pattern means a kernel is accessing memory
just before the tileSegData allocation (the guard page). This is consistent with
a negative index into tileSegData or an underflow in pointer arithmetic.

### Current File State (as of Test 40)
- All .inl files: Real kernel code, `const CRParams& p` (by reference)
- `RasterImpl_kernel.hip`: 4 kernel entry points taking `const CRParams*` (pointer)
- `RasterImpl.cpp`: hipMemcpyAsync on same stream, hipStreamSynchronize after
  EVERY kernel dispatch, all 4 dispatched, diagnostics still present

### Test 41: triSetup+binRaster+coarseRaster real, fineRaster skipped

- 3 real kernels dispatched with sync after each, fineRaster launch removed entirely
- Result: CRASHED (system crash, reboot required)
- Conclusion: **coarseRaster kernel code has the OOB bug.** triSetup+binRaster alone
  is stable (Test 38). Adding coarseRaster causes the crash. The crash is an
  out-of-bounds access in the coarseRaster kernel, hitting tileSegData - 0x1000.

### Test 42: All 4 real kernels + bounds checking on ALL coarseRaster writes

- Added bounds checks with printf on ALL 4 write sites to tileSegData/Next/Count
  in CoarseRaster.inl: [OOB-A] tileSegData, [OOB-B] tileSegNext/Count alloc loop,
  [OOB-C] nextPtr patch, [OOB-D/E] finalize loop tileSegNext/Count
- OOB writes are SKIPPED (clamped), preventing crash from those writes
- Write site 3 also changed `if (oldOfs < 0)` to `if (oldOfs <= 0)` to catch oldOfs==0
- fineRaster re-enabled, hipStreamSynchronize after each kernel
- Result: CRASHED after 3 frames. **No [OOB-*] messages printed.**
  Crash at 0x7dcec89ff000 = tileSegData (0x7dcec8a00000) - 0x1000
- Conclusion: All coarseRaster writes to tileSegData/Next/Count are in-bounds.
  The OOB access is NOT from coarseRaster writes. Must be a READ in fineRaster
  or coarseRaster, or something else entirely.

### Test 42b: Bounds-checked coarseRaster + fineRaster SKIPPED

- Same bounds-checked coarseRaster as Test 42, fineRaster launch commented out
- If passes: crash is in fineRaster reads. If crashes: something else.
- Result: CRASHED (system crash, reboot required)
- Conclusion: Crash is NOT in fineRaster (it's not running). All coarseRaster WRITES
  are bounds-checked and in-bounds (no OOB messages in Test 42). The crash persists
  with 3 dispatches even when all tileSegData/Next/Count writes are guarded.

### Key Finding from Tests 42 + 42b

The crash at tileSegData - 0x1000 is NOT caused by:
- coarseRaster writes to tileSegData/Next/Count (all bounds-checked, all in-bounds)
- fineRaster reads (not running in 42b)

Combined with Test 38 (2 dispatches PASS) and Test 37 (4 dispatches with no-op
kernels CRASH), the pattern is: **3+ kernel dispatches = crash, regardless of what
the kernels do**. The multi-dispatch corruption theory (bug #3) appears correct
after all. Test 40's sync barriers don't prevent it because the corruption is in
the HIP runtime's dispatch mechanism, not in kernel execution ordering.

### Crash Address Pattern (Updated)

| Test | Fault Address | tileSegData | Iterations Before Crash |
|------|--------------|-------------|------------------------|
| 27 | 0x7a214d9ff000 | 0x7a214da00000 | 5 |
| 28 | 0x79a8eb3ff000 | 0x79a8eb400000 | 1 |
| 29 | 0x74b5b73ff000 | 0x74b5b7400000 | 4 |
| 35 | 0x703b363ff000 | (not logged) | 1 |
| 37 | 0x747f4b1ff000 | 0x747f4b200000 | 5 |
| 39 | 0x7fd10f1ff000 | 0x7fd10f200000 | 3 |
| 42 | 0x7dcec89ff000 | 0x7dcec8a00000 | 3 |

### Current File State (as of Test 42b)
- `CoarseRaster.inl`: Real kernel code + bounds checks on all 4 tileSegData/Next/Count
  write sites ([OOB-A] through [OOB-E]). Write site 3: `oldOfs <= 0` fix.
- `FineRaster.inl`: Real kernel code (but launch COMMENTED OUT for Test 42b)
- `TriangleSetup.inl`, `BinRaster.inl`: Real kernel code
- `RasterImpl_kernel.hip`: 4 kernel entry points taking `const CRParams*` (pointer)
- `RasterImpl.cpp`: hipMemcpyAsync on same stream, hipStreamSynchronize after each
  kernel, fineRaster launch commented out, diagnostics present

### Test 43: coarseRaster pure no-op, 3 dispatches (triSetup+binRaster+no-op coarse)

- Added `return;` at top of coarseRasterImpl, fineRaster launch still commented out
- 3 kernel dispatches: triSetup (real) + binRaster (real) + coarseRaster (no-op)
- Result: CRASHED (system crash, reboot required)
- Conclusion: **3 kernel dispatches = crash regardless of kernel code.** The crash
  is in the HIP runtime dispatch mechanism, not in any kernel. Combined with
  Test 38 (2 dispatches PASS) and Test 12 (same kernel 3x PASS), the trigger is
  dispatching 3+ DIFFERENT kernels on the same stream.

### Confirmed Root Cause: HIP Multi-Dispatch Bug

The HIP runtime on RDNA3 (gfx1101) with ROCm 6.4.2 corrupts GPU memory when
dispatching 3+ different kernel functions on the same stream. The corruption
consistently hits the guard page at tileSegData - 0x1000. Sync barriers between
dispatches do not prevent it (Tests 40, 42b, 43).

**Fix strategy**: Use a second HIP stream for the 3rd and 4th kernel dispatches,
keeping each stream at max 2 different kernel dispatches. Use hipEvent for
cross-stream synchronization.

### Current File State (as of Test 43)
- `CoarseRaster.inl`: `return;` at top (no-op), Test 42 bounds checks below (dead code)
- `FineRaster.inl`: Real kernel code (launch commented out)
- `TriangleSetup.inl`, `BinRaster.inl`: Real kernel code
- `RasterImpl_kernel.hip`: 4 kernel entry points taking `const CRParams*` (pointer)
- `RasterImpl.cpp`: hipMemcpyAsync on same stream, hipStreamSynchronize after each
  kernel, fineRaster launch commented out, diagnostics present

### Test 44: All 4 real kernels, two-stream dispatch

- Stream A (stream): triSetup + binRaster (2 dispatches)
- Stream B (m_stream2): coarseRaster + fineRaster (2 dispatches)
- hipStreamSynchronize(stream) between streams A and B
- Also changed d_crParams from static to per-instance member, added m_stream2 member
- Result: CRASHED (system crash, reboot required)
- Conclusion: Two-stream split does NOT help. The multi-dispatch bug is global
  (not per-stream). Dispatching 3+ different kernel functions crashes regardless
  of stream distribution.

### Workarounds Tried and Failed (Updated)

| Workaround | Test(s) | Result |
| --- | --- | --- |
| hipDeviceSynchronize between each kernel | 8 | CRASH |
| hipStreamSynchronize between each kernel | 20-24, 40 | CRASH |
| static CRParams (persistent memory) | 25, 28 | CRASH |
| static CRParams + usleep(1) | 26 | CRASH |
| stderr delay + 3-4 kernels | 27, 29 | CRASH (delayed) |
| hipStreamSynchronize between dispatch pairs | 39 | CRASH after 3 frames |
| Bounds-check all coarseRaster writes | 42, 42b | CRASH (no OOB detected) |
| Two-stream dispatch (2 kernels per stream) | 44 | CRASH |

### Current File State (as of Test 44)
- `CoarseRaster.inl`: Real kernel code + bounds checks, no `return;`
- `FineRaster.inl`: Real kernel code
- `TriangleSetup.inl`, `BinRaster.inl`: Real kernel code
- `RasterImpl_kernel.hip`: 4 kernel entry points taking `const CRParams*` (pointer)
- `RasterImpl.hpp`: Added m_stream2 and m_d_crParams members
- `RasterImpl.cpp`: Two-stream dispatch (stream A: triSetup+binRaster, stream B:
  coarseRaster+fineRaster), per-instance d_crParams, proper cleanup in destructor

### Test 45: tileSegData padded with 4KB, single-stream, all 4 real kernels

- Over-allocated m_tileSegData by 4096 bytes, offset p.tileSegData by 4096
- Reverted to single-stream dispatch (all 4 kernels on same stream, no sync between)
- Result: CRASHED (system crash, reboot required)
- Conclusion: Padding tileSegData does not help. The corruption is more fundamental
  than just hitting one guard page. Either the crash address shifts with the padding,
  or the corruption targets a different location.

### Workarounds Tried and Failed (Updated)

| Workaround | Test(s) | Result |
| --- | --- | --- |
| hipDeviceSynchronize between each kernel | 8 | CRASH |
| hipStreamSynchronize between each kernel | 20-24, 40 | CRASH |
| static CRParams (persistent memory) | 25, 28 | CRASH |
| static CRParams + usleep(1) | 26 | CRASH |
| stderr delay + 3-4 kernels | 27, 29 | CRASH (delayed) |
| hipStreamSynchronize between dispatch pairs | 39 | CRASH after 3 frames |
| Bounds-check all coarseRaster writes | 42, 42b | CRASH (no OOB detected) |
| Two-stream dispatch (2 kernels per stream) | 44 | CRASH |
| Pad tileSegData with 4KB | 45 | CRASH |
| Merge into 2 unified kernels + dynamic smem | 46 | CRASH |

### Key Observation: 4 No-Op Kernels PASS vs 2 Real + 1 No-Op CRASH

- Test 30: 4 no-op kernels (pass by pointer) = PASS
- Test 38: 2 real kernels (triSetup + binRaster) = PASS
- Test 43: 2 real + 1 no-op (triSetup + binRaster + no-op coarseRaster) = CRASH
- Test 37: 2 real + 2 no-op = CRASH after 5 frames

The trigger is: real kernel execution (producing GPU memory writes) PLUS 3+
different kernel dispatches. No-op kernels alone don't trigger it even with 4
dispatches. The combination of real GPU memory writes + additional dispatches is
what corrupts memory.

### New Fix Strategy: Reduce to 2 Kernel Functions

Merge 4 entry points into 2 unified kernels, each handling 2 stages via a
parameter. Dispatch each kernel twice = 4 dispatches of 2 different functions.
Test 12 showed same-kernel dispatches are safe.

### Current File State (as of Test 45)
- `CoarseRaster.inl`: Real kernel code + bounds checks
- `FineRaster.inl`: Real kernel code
- `TriangleSetup.inl`, `BinRaster.inl`: Real kernel code
- `RasterImpl_kernel.hip`: 4 kernel entry points taking `const CRParams*` (pointer)
- `RasterImpl.hpp`: m_stream2 and m_d_crParams members
- `RasterImpl.cpp`: Single-stream dispatch, tileSegData padded +4KB, per-instance
  d_crParams, diagnostics present

### Test 46: Merge into 2 unified kernels + dynamic shared memory

- Merged 4 kernel entry points into 2: setupKernel (triSetup/binRaster) and
  rasterKernel (coarseRaster/fineRaster). Stage parameter selects which impl to run.
- Updated RasterImpl.cpp dispatch, constructor, forward declarations
- Removed tileSegData padding, reverted to single-stream dispatch
- Initial build FAILED: `local memory (92712) exceeds limit (65536) in 'rasterKernel'`
  HIP compiler sums static __shared__ from both coarseRasterImpl (~44KB) and
  fineRasterImpl (~47KB) = ~92KB, exceeding AMD's 64KB LDS limit.
- Fix: Convert coarseRaster and fineRaster to dynamic shared memory
  (extern __shared__) with layout structs in PrivateDefs.hpp. Max = 47KB, fits 64KB.
- Build succeeded after dynamic shared memory conversion.
- Result: CRASHED (system crash, reboot required)
- Conclusion: 2 unified kernels (4 dispatches of 2 functions) still crashes.
  The bug is NOT about the number of distinct kernel functions -- it's triggered
  by 3+ total dispatches with real GPU memory writes, regardless of whether
  they're the same or different kernel functions.

### Test 47: HIP Graph dispatch (all 4 kernels as single command buffer)

- Capture all 4 kernel launches into a hipGraph, instantiate once, replay each frame.
- hipMemcpyAsync for CRParams and atomics stays outside the graph (on same stream).
- Graph is cached and recreated only when numTriangles or numImages change.
- Peel path (1 dispatch) uses regular hipLaunchKernel (no graph needed).
- Result: AWAITING TEST

### Current File State (as of Test 47)
- `CoarseRaster.inl`: Dynamic shared memory via CoarseSmem struct + bounds checks
- `FineRaster.inl`: Dynamic shared memory via FineSmem struct
- `TriangleSetup.inl`, `BinRaster.inl`: Real kernel code (static __shared__, unchanged)
- `RasterImpl_kernel.hip`: 2 unified kernels (setupKernel, rasterKernel)
- `RasterImpl.hpp`: m_graphExec, m_graphNumTriangles, m_graphNumImages members added
- `RasterImpl.cpp`: hipGraph-based dispatch for non-peel, regular dispatch for peel
- `PrivateDefs.hpp`: CoarseSmem and FineSmem layout structs added

### Test 47 Result
- hipStreamBeginCapture on caller's stream failed with hipError 900 (capture unsupported
  on default stream). Fixed by capturing on m_stream2 instead.
- Rebuilt and re-ran. Result: CRASHED (system crash, reboot required).
- Conclusion: HIP Graphs use the same underlying dispatch mechanism. Does not bypass bug.

### Workarounds Tried and Failed (Updated)

| Workaround | Test(s) | Result |
| --- | --- | --- |
| hipDeviceSynchronize between each kernel | 8 | CRASH |
| hipStreamSynchronize between each kernel | 20-24, 40 | CRASH |
| static CRParams (persistent memory) | 25, 28 | CRASH |
| static CRParams + usleep(1) | 26 | CRASH |
| stderr delay + 3-4 kernels | 27, 29 | CRASH (delayed) |
| hipStreamSynchronize between dispatch pairs | 39 | CRASH after 3 frames |
| Bounds-check all coarseRaster writes | 42, 42b | CRASH (no OOB detected) |
| Two-stream dispatch (2 kernels per stream) | 44 | CRASH |
| Pad tileSegData with 4KB | 45 | CRASH |
| Merge into 2 unified kernels + dynamic smem | 46 | CRASH |
| HIP Graph dispatch (single command buffer) | 47 | CRASH |
| Split into 2 launchStages calls (2 dispatches each) | 48 | CRASH |

### Test 48: Split dispatch into two launchStages calls from drawTriangles
- drawTriangles calls launchStages(stageSet=0) then launchStages(stageSet=1)
- stageSet 0: triSetup + binRaster (2 dispatches of setupKernel), full CRParams rebuild + sync
- stageSet 1: coarseRaster + fineRaster (2 dispatches of rasterKernel), full CRParams rebuild + sync
- Each call mirrors Test 38's pattern (CRParams memcpy, 2 dispatches, sync, function return)
- Result: CRASHED (system crash, reboot required)
- Conclusion: Split dispatch with 2 per call doesn't help. The bug persists even
  with full function return + CRParams rebuild between the two halves. The "2 dispatches
  per call" success of Test 38 may have been because it never dispatched rasterKernel
  (coarseRaster/fineRaster) at all, not because of the dispatch count limit.

### Revised Analysis (Post-Test 48)

Test 38 (PASS) only dispatched setupKernel (triSetup + binRaster). It never ran
coarseRaster or fineRaster. All tests that include coarseRaster or fineRaster with
real kernel code crash. This points to an actual bug in the coarseRaster or
fineRaster kernel code on RDNA3, not a dispatch mechanism issue.

Test 42 bounds-checked all coarseRaster WRITES and found them in-bounds. But we
never checked: (1) coarseRaster READS, (2) fineRaster READS from tileSegData,
(3) fineRaster-specific buffer accesses.

### Test 49: setupKernel x2 + rasterKernel x1 (coarseRaster only, no fineRaster)
- stageSet=0: setupKernel dispatched twice (triSetup + binRaster)
- stageSet=1: rasterKernel dispatched once (coarseRaster stage=0 only, fineRaster skipped)
- Total: 3 dispatches (2 of setupKernel + 1 of rasterKernel) across 2 launchStages calls
- Result: CRASHED (system crash, reboot required)
- Conclusion: Even 1 dispatch of rasterKernel (coarseRaster) causes the crash.
  Combined with Test 38 (setupKernel only = PASS), the issue is specifically triggered
  by dispatching rasterKernel on RDNA3. Could be: (1) coarseRaster kernel code OOB,
  (2) dynamic shared memory (~47KB) triggering a hardware bug, (3) rasterKernel's
  compiled binary having a property that causes the fault.

### Important: ROCm Version Change
System shows ROCm 7.1.1 installed. CLAUDE.md previously said 6.4.2. Tests 1-45 may
have been on 6.4.2, while Tests 46-49 are on 7.1.1. Test results may not be directly
comparable across the version change.

### Test 50: rasterKernel dispatched with no-op coarseRasterImpl
- Added `return;` at top of coarseRasterImpl -- all 640 threads exit immediately
- rasterKernel still dispatched (1 dispatch from stageSet=1), but does zero work
- Result: CRASHED (system crash, reboot required)
- Conclusion: **rasterKernel binary itself is the trigger.** The crash is NOT from
  coarseRaster code (it never executes). Something about rasterKernel's compiled
  binary causes the fault: dynamic shared memory (~47KB request), launch_bounds(640,1),
  or some other compiled property. The coarseRaster/fineRaster code is irrelevant.

### Workarounds Tried and Failed (Updated)

| Workaround | Test(s) | Result |
| --- | --- | --- |
| hipDeviceSynchronize between each kernel | 8 | CRASH |
| hipStreamSynchronize between each kernel | 20-24, 40 | CRASH |
| static CRParams (persistent memory) | 25, 28 | CRASH |
| static CRParams + usleep(1) | 26 | CRASH |
| stderr delay + 3-4 kernels | 27, 29 | CRASH (delayed) |
| hipStreamSynchronize between dispatch pairs | 39 | CRASH after 3 frames |
| Bounds-check all coarseRaster writes | 42, 42b | CRASH (no OOB detected) |
| Two-stream dispatch (2 kernels per stream) | 44 | CRASH |
| Pad tileSegData with 4KB | 45 | CRASH |
| Merge into 2 unified kernels + dynamic smem | 46 | CRASH |
| HIP Graph dispatch (single command buffer) | 47 | CRASH |
| Split into 2 launchStages calls (2 dispatches each) | 48 | CRASH |
| 1 dispatch of rasterKernel (coarseRaster only) | 49 | CRASH |
| No-op rasterKernel (coarseRasterImpl returns immediately) | 50 | CRASH |

### Test 51: rasterKernel with 0 bytes dynamic shared memory
- Same no-op coarseRasterImpl (`return;` at top), but launch smem size changed to 0
- Only change from Test 50: `rasterSmemSize` -> `0` in hipLaunchKernel call
- Result: **PASSED** -- no crash! Process stayed alive, Gaussian and mesh rendering ran.
  OOB-D log messages appeared (unexpected with `return;` -- possibly stale build cache).
- Conclusion: **Dynamic shared memory allocation (~47KB) on RDNA3/ROCm is the crash
  trigger.** The kernel code itself is irrelevant. Requesting ~47KB of dynamic LDS via
  `extern __shared__` + smem launch parameter causes GPU memory faults. Static
  `__shared__` (used by setupKernel) works fine.

### Root Cause (CONFIRMED -- 4th bug)

Four bugs found in the nvdiffrast HIP rasterizer on RDNA3:
1. **HIP kernarg buffer corruption** (Test 30): Large by-value kernargs corrupt on RDNA3.
   Fix: pass CRParams* pointer.
2. **Cross-stream ordering** (Test 35): hipMemcpy on NULL stream doesn't sync with
   kernel launches on non-default stream. Fix: hipMemcpyAsync on same stream.
3. **Dynamic shared memory GPU fault** (Test 51): Requesting ~47KB of dynamic LDS
   (`extern __shared__` + smem launch parameter) causes GPU memory faults on RDNA3.
   Fix: use static `__shared__` with a union in the kernel entry point.
4. **OOB in coarseRaster** (Test 42 OOB-D logs): segIdx=-1 being used as valid index.
   To be fixed after smem issue is resolved.

### Test 52: Static `__shared__` union (~48KB) with all 4 real kernel stages

- Converted rasterKernel from dynamic `extern __shared__` to static `__shared__` union
  of CoarseSmem (~44KB) and FineSmem (~48KB). Union size = 48,384 bytes.
- Removed `return;` from coarseRasterImpl, re-enabled fineRaster dispatch.
- Impl functions take `char* s_smem` parameter, cast to CoarseSmem&/FineSmem&.
- All launches use 0 for dynamic smem parameter.
- Result: CRASHED (system crash, reboot required)
- Conclusion: **Static shared memory ~48KB also crashes on RDNA3.** The issue is
  NOT dynamic vs static -- it's the LDS SIZE itself. Any kernel with ~44KB+ of
  LDS per workgroup causes GPU memory faults on RDNA3/ROCm.

### Revised Root Cause Analysis

The crash has always been about LDS size. Every test that included coarseRaster
(~44KB LDS) or fineRaster (~48KB LDS) has crashed. The "multi-dispatch" theory
was wrong -- we were always crashing because of large LDS, not dispatch count.

Evidence:
- Test 38 (PASS): setupKernel only (small LDS), no coarse/fine
- Test 51 (PASS): rasterKernel with 0 LDS (dynamic smem = 0, no-op impl)
- Test 52 (CRASH): rasterKernel with ~48KB static LDS
- Tests 40-50 (CRASH): All included coarse/fine kernels with ~44-48KB LDS

### Workarounds Tried and Failed (Updated)

| Workaround | Test(s) | Result |
| --- | --- | --- |
| hipDeviceSynchronize between each kernel | 8 | CRASH |
| hipStreamSynchronize between each kernel | 20-24, 40 | CRASH |
| static CRParams (persistent memory) | 25, 28 | CRASH |
| static CRParams + usleep(1) | 26 | CRASH |
| stderr delay + 3-4 kernels | 27, 29 | CRASH (delayed) |
| hipStreamSynchronize between dispatch pairs | 39 | CRASH after 3 frames |
| Bounds-check all coarseRaster writes | 42, 42b | CRASH (no OOB detected) |
| Two-stream dispatch (2 kernels per stream) | 44 | CRASH |
| Pad tileSegData with 4KB | 45 | CRASH |
| Merge into 2 unified kernels + dynamic smem | 46 | CRASH |
| HIP Graph dispatch (single command buffer) | 47 | CRASH |
| Split into 2 launchStages calls (2 dispatches each) | 48 | CRASH |
| 1 dispatch of rasterKernel (coarseRaster only) | 49 | CRASH |
| No-op rasterKernel (coarseRasterImpl returns immediately) | 50 | CRASH |
| Static `__shared__` union (~48KB) | 52 | CRASH |

### Test 53: 32KB static `__shared__` with no-op coarse+fine

- `__shared__ char s_smem[32768]` (32KB) in rasterKernel entry point
- Both coarseRasterImpl and fineRasterImpl have `return;` at top (no-op)
- Added `maxSmemPerBlock` device attribute query to constructor
- Result: **PASSED** -- full pipeline ran, 3D asset generated, no crash
- Conclusion: 32KB static LDS works. The crash threshold is between 32KB and 48KB.

### Test 54: 40KB static `__shared__` with no-op coarse+fine, run via reproducer harness

- `__shared__ char s_smem[40960]` (40KB) in rasterKernel entry point
- coarseRasterImpl and fineRasterImpl still no-op (`return;` at top)
- First test run via the new tools/raster_repro/ harness instead of the full TRELLIS pipeline.
  120 replay frames against synthetic 318,402-triangle heightfield mesh at 1024x1024.
- Result: **PASSED** -- 120 frames, 0.064s elapsed, no fault in dmesg, harness clean exit
- Conclusion: 40KB static LDS works. Crash threshold is between 40KB and 48KB.
  Iteration loop is now ~10,000x faster than the full pipeline (0.064s vs ~10 min).

### Test 55: 48KB static `__shared__` with no-op coarse+fine, harness

- `__shared__ char s_smem[49152]` (48KB) in rasterKernel entry point
- coarseRasterImpl and fineRasterImpl still no-op (`return;` at top)
- Replayed 120 frames against synthetic 318,402-triangle mesh
- Result: **PASSED** -- 120 frames, 0.065s elapsed, no fault in dmesg
- **CRITICAL FINDING:** This was expected to CRASH per Test 52's "48KB causes
  GPU faults" conclusion. Re-examining the data, Test 52 had `48KB + real
  coarseRaster + real fineRaster` while Tests 53/54/55 are all
  `<size> + no-op coarse + no-op fine`. The two-variable confound means we
  do NOT actually have evidence that LDS size alone is the trigger. The
  Test 52 crash may have been driven by the real impl bodies (or a
  combination), not the LDS allocation.
- Conclusion: 48KB static LDS + no-op impl = PASS. The "LDS threshold between
  32KB and 48KB" synthesis from prior sessions was overfitted to single
  data points across confounded variables.

### LDS Size Test Matrix (REVISED)

| Size | coarse | fine | Test | Result | Method |
|------|--------|------|------|--------|--------|
| 0 dynamic | no-op | no-op | 51 | PASS | full pipeline |
| 32KB | no-op | no-op | 53 | PASS | full pipeline |
| 40KB | no-op | no-op | 54 | PASS | harness |
| 48KB | no-op | no-op | 55 | PASS | harness |
| 48KB (union) | REAL | REAL | 52 | CRASH | full pipeline |

### Reproducer harness (since Test 54)

Replaces full-pipeline iteration with a self-contained replay loop. Files in `tools/raster_repro/`:

- `program.md` -- methodology document, agent reads this
- `synth.py` -- generates a 318K-triangle heightfield mesh in the same payload format capture.py would produce
- `capture.py` -- monkey-patches `dr.rasterize()` to dump real inputs from a TRELLIS run (currently blocked by a slat-sampling stage failure on torch 2.9.1+rocm6.4 / system ROCm 7.2.1, so synth path is being used)
- `harness.py` -- loads the dump, replays 120 frames, emits one JSON line summarizing PASS/CRASH+frames+timing
- `run.sh` -- subprocess wrapper, dmesg fault-address parsing, JSONL append to `tools/raster_repro/results.jsonl`

Usage: `tools/raster_repro/run.sh <test_name> '<config_json>'`

### Current File State (as of Test 56)
- `CoarseRaster.inl`: REAL kernel code (no-op `return;` removed in Test 56), takes `char* s_smem`, bounds checks present
- `FineRaster.inl`: REAL kernel code (no-op `return;` removed in Test 56), takes `char* s_smem`
- `TriangleSetup.inl`, `BinRaster.inl`: Real kernel code
- `RasterImpl_kernel.hip`: 2 unified kernels, rasterKernel has `__shared__ char s_smem[49152]`
- `RasterImpl.hpp`: m_stream2, m_d_crParams, hipGraph members
- `RasterImpl.cpp`: Split dispatch, all launches use 0 dynamic smem, logs maxSmemPerBlock

### Test 56: 48KB + REAL coarseRaster + REAL fineRaster, harness, synth mesh

- Removed `return;` from both coarseRasterImpl and fineRasterImpl
- 48KB static `__shared__ char s_smem[49152]` retained
- Replayed 120 frames against the synthetic 318,402-triangle heightfield mesh
- Result: **PASSED** -- 120 frames, 0.064s, no fault, harness clean exit
- **MAJOR CAVEAT:** Stderr shows `tileSegs=0 activeTiles=0` for every frame.
  coarseRaster produced no tile segments. Real TRELLIS workloads (e.g. Test 35)
  showed `tileSegs=12527 activeTiles=16384`. The synthetic heightfield mesh
  has triangles small enough (~2 pixels/side at 1024x1024) that coarseRaster
  likely takes a single-tile fast path and bypasses the linked-list writes
  that historically faulted. So this PASS does NOT prove the Test 52 crash
  is resolved; it proves only that coarseRaster on this trivially-tiled
  workload doesn't crash.
- Conclusion: harness is missing the bug-triggering code path. Need a
  synthetic mesh whose triangles span multiple tiles each (large area, sparse
  coverage), OR a real TRELLIS mesh (currently blocked by slat sampling
  failure on torch 2.9.1+rocm6.4 / system ROCm 7.2.1).

### Test 57: 48KB + real coarse+fine, harness, soup-mode synth (300K large random triangles)

- Same kernel state as Test 56, regenerated input via `synth.py --mode soup`
- Result: **PASSED** but later **proven to be a false PASS** -- see Tests 60-62 below.
  The actual binary running at this time was a stale 32KB + no-op build, not 48KB
  with real impl as the source claimed.

### Tests 58 + 59: icosphere mesh (327,680 triangles), w/ and w/o perspective

- Same false-PASS pattern as Tests 55-57 (stale binary, see Tests 60-62 below).

### Tests 60-62: build-cache trap discovered, prior tests retracted

Adding a printf to coarseRasterImpl forced a real recompile (touched the
source). Constructor output then changed from `coarseBlocks=4 fineBlocks=3`
(stale) to `coarseBlocks=1 fineBlocks=1` (clean) and the harness CRASHED
immediately with "Memory access fault by GPU node-1". Investigation:

- **Test 60** (48KB + real both, clean build): CRASHED. coarseBlocks=1.
- **Test 61** (40KB + real both, clean build after `rm -rf build/`): CRASHED.
  coarseBlocks=1. (Also fails because FineSmem ~47KB doesn't fit in 40KB
  s_smem buffer -- OOB writes from real fineRasterImpl.)
- **Test 62** (32KB + no-op both, clean build): **PASSED.** coarseBlocks=4,
  fineBlocks=3. Matches Test 53's original finding from the full-pipeline era.

### CRITICAL: stale-binary trap (retracts Tests 55-59)

Tests 55-59 all reported PASS but were running stale 32KB + no-op binaries
despite source files saying 48KB + real impl. The build cache (some
combination of pip wheel cache, ninja .o cache, and pip's editable-install
behavior) was reusing prior intermediates. Symptoms that should have flagged
this earlier:

- `coarseBlocks=4 fineBlocks=3` reported by ctor for "48KB" runs (only
  consistent with smem <= 16KB, i.e., 32KB never produced this; 48KB
  cannot allow 4 blocks given 64KB max-smem-per-SM)
- `tileSegs=0 activeTiles=0` across all "real impl" runs (would be
  consistent with the no-op bypass)
- Identical timing/output across very different kernel configs

**Mandatory rebuild protocol going forward** (must do all three):

```
cd extensions/nvdiffrast-hip
rm -rf build/                                        # ninja artifacts
pip cache remove "nvdiffrast*" 2>/dev/null || true   # pip wheel cache
pip install . --no-build-isolation --force-reinstall --no-deps
```

The `tools/raster_repro/run.sh` should be updated to do this automatically
before each test, OR `program.md` should make it the agent's responsibility.

### Corrected LDS / Bug Understanding

| Smem | Real impl | Result | Occupancy | Method |
|------|-----------|--------|-----------|--------|
| 32KB | none | PASS | 4/3 | full pipeline (Test 53) + harness clean (Test 62) |
| 32KB | real fine | CRASH | 4/3 (compute), but FineSmem (~47KB) overflows the 32KB buffer | harness clean |
| 40KB | real both | CRASH | 1/1 | harness clean (Test 61) |
| 48KB | real both | CRASH | 1/1 | full pipeline (Test 52) + harness clean (Test 60) |

The original Test 52 conclusion (48KB+ crashes on RDNA3) is **vindicated**.
The mid-day "we overturned the LDS hypothesis" was wrong. Two real problems:

1. **`FineSmem` is ~47KB**, larger than any LDS budget that allows occupancy
   > 1. Casting it onto a smaller `s_smem` buffer = OOB writes = page faults.
2. **High-LDS kernel binaries** (>32KB) trigger the original bug pattern,
   regardless of whether FineSmem overflows. Test 52 used a 48KB union and
   still crashed; presumably the trigger is independent of the OOB.

To make progress: **reduce FineSmem size** to fit in <=32KB, then re-enable
real impl and re-validate.

### CRITICAL: VSCode session auto-revert traps Constants.hpp edits

When VSCode is running with Constants.hpp open in a tab, edits to that file
made via the API or shell tools (Edit, sed, etc.) get silently overwritten
by VSCode's in-memory buffer on its next auto-save. This caused Tests 65 and
67 to crash the GPU and Xorg: source said `CR_FINE_MAX_WARPS=12` (intended
fix) but the binary built with `=20` (VSCode's stale buffer wrote back),
producing an OOB-bound FineSmem on the smaller s_smem buffer.

Defenses now in `tools/raster_repro/run.sh`:

1. **Layer 1**: sed-modify Constants.hpp pre-build, verify the value stuck
   immediately (catches sub-second reverts).
2. **Layer 2**: pre-flight uses `COARSE_WARPS_OVERRIDE` / `FINE_WARPS_OVERRIDE`
   env vars, NOT whatever Constants.hpp currently shows on disk.
3. **Layer 3**: smoke-probe the BUILT BINARY by constructing a
   `RasterizeCudaContext` and parsing `fineWarps=N` from the ctor diagnostic.
   If it doesn't match the env override, abort BEFORE the harness runs.

If Layer 3 aborts, **the user must close Constants.hpp in VSCode without
saving** before retrying. Closing the tab removes VSCode's authoritative
in-memory copy, after which sed-edits stick through the build.

Test 67 vindicated all three layers: VSCode reverted Constants.hpp during
the build, smoke probe caught the mismatch, and the harness was prevented
from running -- no GPU fault, no display crash.

### Next Steps (REVISED after Test 59)

The harness is validated for non-crash workloads (Tests 54-57 all PASS, ~10000x
faster iteration than full pipeline) but cannot currently reproduce the
bug-triggering coarseRaster work. To make further progress:

1. **Get a real TRELLIS mesh dump.** The slat sampling failure on torch
   2.9.1+rocm6.4 / system ROCm 7.2.1 is the blocker. Two ways:
   a. Rebuild torchsparse against current torch (`pip install -e
      extensions/torchsparse --force-reinstall`) and retry capture.py
   b. Construct a mesh from a saved 3D asset (.glb / .ply) and feed it
      through `nvdiffrast.torch.rasterize` directly, bypassing the
      diffusion pipeline.
2. Re-run Test 57's config against the real-mesh input. If
   `tileSegs > 1000` and the harness still PASSES, the original bug is
   genuinely fixed (likely by ROCm 7.2.1 vs 7.1.1).
3. If the real-mesh input crashes: bisect coarse vs fine, then chase the
   `segIdx=-1` OOB Test 42 hinted at.
4. If the real-mesh input passes: end-to-end validation by re-enabling
   the full TRELLIS pipeline and running app.py.
