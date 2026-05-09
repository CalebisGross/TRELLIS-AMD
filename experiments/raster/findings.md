# Confirmed findings

Distilled from the test log. Each entry: what the bug is, how we found it,
and the fix that's in tree (or planned).

## Bug 1 — HIP kernarg buffer corruption with large by-value structs

**Symptom:** GPU page faults at addresses near pointer values embedded in
`CRParams`, consistently `tileSegData - 0x1000` (the unmapped guard page
just before the `tileSegData` allocation). Trigger: dispatch 3+ different
kernels each taking `CRParams` (~680 bytes) by value.

**Surface:** Tests 8-29 (RDNA3, ROCm 6.4.2). Originally misattributed to
"multi-dispatch corruption" / "stderr delay race"; actual mechanism was
the kernarg buffer.

**Confirmation:** Test 30. Changing kernel signature to
`const CRParams*` (8-byte kernarg) made the fault disappear with all 4
no-op kernels dispatched.

**Fix in tree:** All `rasterKernel` / `setupKernel` entry points take
`const CRParams* p`. Host hipMallocs a device CRParams once per
`RasterImpl` instance and points the kernarg at it.
[`RasterImpl_kernel.hip`](../../extensions/nvdiffrast-hip/csrc/common/hipraster/impl/RasterImpl_kernel.hip),
[`RasterImpl.cpp`](../../extensions/nvdiffrast-hip/csrc/common/hipraster/impl/RasterImpl.cpp).

## Bug 2 — Cross-stream ordering: NULL-stream hipMemcpy doesn't sync with non-default stream kernels

**Symptom:** With Bug 1 fix in place but still `hipMemcpy(d_crParams, ...)`
on the NULL stream, kernels read uninitialized device memory.
`triangleSetupImpl` saw `numTriangles=0` and short-circuited; pre-filling
the output with `0xAA` showed no kernel writes occurred at all.

**Surface:** Tests 31-34 (kernel writes silently absent).

**Confirmation:** Test 35. Switching to
`hipMemcpyAsync(d_crParams, ..., stream)` on the same stream as the
kernel launches restored real kernel output (`binSegs=565`,
`tileSegs=12527`, `activeTiles=16384`).

**Fix in tree:** [`RasterImpl.cpp`](../../extensions/nvdiffrast-hip/csrc/common/hipraster/impl/RasterImpl.cpp)
uses `hipMemcpyAsync` on the kernel's stream.

**By-reference impl signatures:** `triangleSetupImpl(const CRParams& p)`
etc. avoids a 680-byte by-value copy from device global memory inside the
kernel entry point.

## Bug 3 — High static __shared__ + real coarse/fine impls crash on RDNA3

**Symptom:** `rasterKernel` with ~44-48 KB of static `__shared__` (a
union of `CoarseSmem` and `FineSmem`) plus real `coarseRasterImpl` /
`fineRasterImpl` faults the GPU. Faults disappear when either the
LDS shrinks or the impls go no-op.

**Confirmation matrix (post stale-binary cleanup):**

| s_smem | Real coarse+fine | Result | Occupancy | Source |
|--------|------------------|--------|-----------|--------|
| 32 KB  | both no-op       | PASS   | 4/3       | Test 53, 62 |
| 32 KB  | real fine only   | CRASH  | 4/3       | (FineSmem ~47 KB overflows) |
| 40 KB  | real both        | CRASH  | 1/1       | Test 61 |
| 48 KB  | real both        | CRASH  | 1/1       | Test 52, 60 |

**Two distinct sub-issues:**

1. `FineSmem` (~47 KB) cannot fit any LDS budget that allows occupancy >1
   block/SM. Casting it onto a smaller `s_smem` buffer produces OOB
   writes.
2. Even at 48 KB (no overflow), real `coarseRaster` triggers an
   `hipErrorLaunchOutOfResources` (719) at launch with `regsPerThread`
   reported as 202-230 (Tests 83-85). The compiler over-allocates
   registers under high-LDS occupancy=1 conditions.

**Mitigation in tree (WIP):** Refactor moves `CoarseSmem.warpEmitMask`
and `warpEmitPrefixSum` (~33 KB) out of LDS into a per-block
`CoarseGlobalScratch` allocation. `CR_COARSE_WARPS` reduced 16→8,
`CR_FINE_MAX_WARPS` reduced 20→12 to bring `FineSmem` under 32 KB.
Real impls are still SAFE-MODE no-op; full refactor introduces Bug 6
below.

**Update 2026-05-09 (session 2 + 3):** The "register pressure" framing
in this section is incomplete. Test 86 confirmed printf gating dropped
VGPRs 202→179 but the launch still failed. Test 103 forced VGPRs to
128 via `__attribute__((amdgpu_num_vgpr(128)))` and it STILL failed.
Test 111 with an early `return;` after one write changed the error
from 719 to **700** (`hipErrorIllegalAddress`), revealing that the
real fault is a memory access, not a launch-resource shortfall. The
719 in Tests 73+ is the runtime masking the real error 700 because
the kernel binary is large and the launch validation layer reports
generic launch failure first. See [Bug 6](#bug-6--triheaderimisc-oob-on-rdna3-resolved-2026-05-09)
for the confirmed mechanism (and resolution).

## Bug 4 — segIdx=-1 OOB hint in coarseRaster (unconfirmed)

Test 42's `oldOfs <= 0` instrumentation (changed from `< 0` to `<= 0`)
was added to catch a suspected `segIdx == -1` value being used as a valid
index into `tileSegData`. Bounds-checked writes never logged an OOB,
but the check only covered writes, not reads. This is the next OOB to
chase once Bug 3 is resolved.

## Bug 5 — VSCode auto-revert trap (operational)

When VSCode is running with `Constants.hpp` open, edits made via the API
or shell tools get overwritten by VSCode's in-memory buffer on its next
auto-save. Symptom: source says `CR_FINE_MAX_WARPS=12`, build produces
binary with `=20`, FineSmem overflows the smaller `s_smem`, GPU + Xorg
crash. Burned us on Tests 65 and 67.

**Defense (in `tools/raster_repro/run.sh`):**

1. sed-modify `Constants.hpp` pre-build, verify the value stuck.
2. Pre-flight uses `COARSE_WARPS_OVERRIDE` / `FINE_WARPS_OVERRIDE` env
   vars rather than whatever Constants.hpp currently shows.
3. Smoke-probe the built binary: construct `RasterizeCudaContext`, parse
   `fineWarps=N` from the ctor diagnostic, abort if mismatched.

If layer 3 aborts: **close `Constants.hpp` in VSCode without saving**,
then retry.

Test 67 vindicated all three layers.

## Bug 6 — `triHeader[i].misc` OOB on RDNA3 (RESOLVED 2026-05-09)

**Resolution:** Bounds-check the two `triHeader[]` reads in
`coarseRasterImpl` (CoarseRaster.inl ~line 311) and the
`getTriangle` helper + per-fragment `triData` read in
`fineRasterImpl` (FineRaster.inl ~line 58 and ~363). Threads that
would have read out-of-range either skip the triangle (coarse) or
emit a z-fail fragment (fine). Test 132 validated 60 frames clean.
Commits 46b5668 (coarse) and 8342a20 (fine).

**Symptom:** Real `coarseRasterImpl` faults with
`hipErrorIllegalAddress` (700). Earlier tests reported error 719
(`hipErrorLaunchFailure`) because the larger kernel binary made the
runtime fall back to that more generic error. With an early `return;`
immediately after the suspected write (Test 111), the actual 700
surfaced. **The 700 was reported AT the write but the underlying OOB
was upstream**, at the triHeader read at line 320 — Test 123 confirmed
this when an earlier `return;` (before the triHeader read) made the
kernel pass cleanly.

**Confirmation (2026-05-09 session 3, Tests 91-126):**

The bisect of `coarseRasterImpl` first looked plausible at the
`s_warpEmitMask` write site (currPtr / LDS-relief refactor). All
those theories were red herrings — see [workarounds.md] for the
discarded "register-pressure" and "currPtr OOB" lines. Test 123
broke the case open: moving an early `return;` to BEFORE the
`triHeader[]` read at the top of the do-while body made the kernel
pass cleanly. Tests 124-126 instrumented `triHeader[dataIdx].misc`
and observed values like `4138327560` (0xF6A4_18B8) for
`triIdx=0` — uninitialized garbage, not a stale value from
aliasing.

Validation runs after the fix:

- Test 130: real coarse + safe fine, 10 frames — PASS
- Test 131: real coarse + real fine, 1 frame — PASS
- Test 132: full pipeline, 60 frames — PASS
- Test 133: full TRELLIS end-to-end mesh export, 300 frames — PASS
- Test 135: real-mesh harness (assets/T.ply, 61k tris, 120 frames) — PASS

Diagnostic during Test 133 (real TRELLIS mesh, 656k subtris/frame):
~7% of bin-queued triangles trigger the bounds-check else-branch
in `coarseRasterImpl` per frame. The fix culls them silently;
visual fidelity holds (mesh.mp4 fully recognizable across 300 frames).

### Phase C — root-cause hypothesis for the missing `.misc` writes

The defensive bounds check is in tree but the underlying invariant
violation is unresolved. The architecture (read against
[TriangleSetup.inl](../../extensions/nvdiffrast-hip/csrc/common/hipraster/impl/TriangleSetup.inl),
[BinRaster.inl](../../extensions/nvdiffrast-hip/csrc/common/hipraster/impl/BinRaster.inl),
and [PrivateDefs.hpp](../../extensions/nvdiffrast-hip/csrc/common/hipraster/impl/PrivateDefs.hpp)):

- `CRTriangleHeader` is 16 bytes (6×S16 + 1×U32). The U32 `.misc`
  packs depth+flipbits when `triSubtris[taskIdx] == 1`, or is
  `subtriBase` (an index back into triHeader[]) when `triSubtris[taskIdx] >= 2`.
- `triangleSetupImpl` writes `.misc` two ways:
  - Fast path (numSubtris=1): `setupTriangle()` writes the entire
    16-byte header via `*(uint4*)th = make_uint4(...)` at TriangleSetup.inl:173.
  - Multi-subtri path: `triHeader[taskIdx].misc = subtriBase` at
    TriangleSetup.inl:402 — a single 4-byte write at offset 12 of
    the 16-byte struct. Then the per-subtri `setupTriangle()` calls
    write FULL uint4s at `triHeader[subtriBase + i]`.
- `binRasterImpl` queues `triIdx*8 + subtriIdx` into binSegData
  ONLY for `triSubtris[triIdx] > 0` (BinRaster.inl:138-149).
- `coarseRasterImpl` then reads `triHeader[dataIdx].misc` at
  CoarseRaster.inl:328 to resolve the subtri base.

Three plausible AMD-specific failure modes:

1. **Partial-store coherency between kernels.** The 4-byte
   `.misc = subtriBase` write at TriangleSetup.inl:402 is a
   stand-alone DWORD store at offset 12. RDNA3 has a writeback L2;
   AMDGPU LLVM is supposed to emit a `s_dcache_wb` / `buffer_wbinvl1`
   on kernel exit, but if it only flushes the lines touched by
   uint4 writes (which dominate setupTriangle's body) it might miss
   a partial-line store from line 402. Plausible because uint4
   writes pass and the partial dword is what fails.
2. **Race window: BinRaster reads `triSubtris[]` updated, but
   `triHeader[].misc` not yet.** The kernel boundary should provide
   coherency, but if there is a missing fence the multi-subtri
   triangle gets queued before `.misc` is visible.
3. **BinRaster mistakenly queues triangles whose
   `triSubtris[taskIdx] == 0`.** This would mean the OOB `.misc` is
   a field that was never written at all (frustum-culled triangle).
   Should be ruled out by the `if (num)` guard at BinRaster.inl:138,
   but worth verifying.

**Phase C first test (highest signal per effort):**

Add a CR_DEBUG_OOB-gated diagnostic in `coarseRasterImpl` that, on
the OOB else-branch, prints both `dataIdx` AND
`((const U8*)p.triSubtris)[dataIdx + p.maxSubtris*blockIdx.z]`. The
result decides between hypotheses:

- `triSubtris[dataIdx] == 0` → BinRaster bug (hypothesis 3)
- `triSubtris[dataIdx] == 1` → BinRaster encoded subtriIdx<7 for a
  single-subtri triangle (impossible per BinRaster.inl:148-149 but
  worth confirming)
- `triSubtris[dataIdx] >= 2` and the value is consistent across
  reads → `.misc` write was lost (hypothesis 1 / coherency)

The diag printf will inflate VGPRs again — gate it behind
CR_DEBUG_OOB and run only on the harness with T.ply (already a
clean repro per Test 135's "deepest checkpoint = 0" trace).

## Stale-binary trap (operational)

Tests 55-59 reported PASS with stale 32 KB + no-op binaries despite the
source claiming 48 KB + real impls. The build cache (pip wheel cache,
ninja `.o` cache, editable-install behavior) reused prior intermediates.
Symptoms that should have flagged it: `coarseBlocks=4 fineBlocks=3`
ctor output for "48KB" runs (only consistent with smem ≤16 KB);
`tileSegs=0 activeTiles=0` across all "real impl" runs; identical
timing across very different configs.

**Mandatory rebuild protocol (now enforced by `run.sh`):**

```
cd extensions/nvdiffrast-hip
rm -rf build/
pip cache remove "nvdiffrast*" 2>/dev/null || true
pip install . --no-build-isolation --force-reinstall --no-deps
```
