# nvdiffrast HIP rasterizer investigation

Active investigation into GPU memory faults in the HIP-ported nvdiffrast
rasterizer when rendering real TRELLIS meshes (~290K-320K triangles) on
AMD RDNA3 (gfx1101).

## Why this directory exists

CLAUDE.md was bloated with hundreds of test entries. Project context is
operational and lives in CLAUDE.md; the lab notebook lives here.

- [findings.md](findings.md) — confirmed root causes and fixes
- [workarounds.md](workarounds.md) — failed workarounds (don't repeat)
- [daily/](daily/) — day-by-day log, one file per date
- `../tools/raster_repro/results.jsonl` — machine-readable harness log

## Current investigation state

- **Branch:** `wip-amd-raster-investigation`
- **Last commit:** `7343575` (example.py phase-aware offload)
- **Toolchain:** torch 2.10.0+rocm7.0 (HIP 7.0.51831), system ROCm 7.2.1
- **Kernel state:** Both real impls active. Bug 6 fixed via bounds checks at the `triHeader[]` read sites. Tests 132 (60-frame harness), 133 (end-to-end TRELLIS mesh, 300 frames @ 2048²), 135 (real-mesh harness fixture from `assets/T.ply`) all clean.
- **Static __shared__:** `s_smem[32768]` (32 KB) in `rasterKernel`.
- **Constants:** `CR_COARSE_WARPS=8`, `CR_FINE_MAX_WARPS=12` (canonical in
  `extensions/nvdiffrast-hip/csrc/common/cudaraster/impl/Constants.hpp`; hipify
  regenerates the HIP copy each build).
- **CRParams:** passed by `const CRParams*` (8-byte kernarg), copied to device
  via `hipMemcpyAsync` on the kernel's stream.

## Status (updated 2026-05-09 evening)

**Bug 6 RESOLVED.** Bounds-checking the `triHeader[i].misc` reads in
both `coarseRasterImpl` and `fineRasterImpl::getTriangle` (plus the
per-fragment `triData` read) eliminates the
`hipErrorIllegalAddress` (700) crash. The earlier "currPtr OOB" /
"register pressure" theories were red herrings — see Tests 91-130
in [daily/2026-05-09.md](daily/2026-05-09.md).

Validated:

- Test 132: 60 frames, harness, real coarse + real fine — clean
- Test 133: end-to-end TRELLIS mesh export (300-frame mesh.mp4 @ 2048², 656k subtris) — first ever real TRELLIS mesh on RDNA3
- Test 135: real-mesh harness fixture (`assets/T.ply`, 61k tris, 120 frames) — clean

See [findings.md § Bug 6](findings.md#bug-6--triheaderimisc-oob-on-rdna3-resolved-2026-05-09).

Next session attack:

1. **Phase C — root cause of `triHeader[i].misc` OOB.** The bounds-check
   fix is defensive; ~7% of triangles per frame still get silently
   culled. The real question: is `triangleSetup` failing to write
   `.misc` for some entries, or is `binRaster` pushing stale `triIdx`
   values into `binSegData`? Both possibilities need a bisect inside
   their own kernels.
2. **Larger real-mesh fixture.** Capture a TRELLIS-pipeline-generated
   GLB into `inputs.pt` (327k subtris) so the harness covers the full
   triangle distribution that crashed before. T.ply (61k tris) is a
   useful regression but doesn't stress the same scale.
3. Re-enable mesh-preview in `app.py` (currently has try/except SAFE-MODE
   wrap from the historical fault).

## Reproducing

```
source .venv/bin/activate
ATTN_BACKEND=sdpa XFORMERS_DISABLED=1 SPARSE_BACKEND=torchsparse \
  tools/raster_repro/run.sh <test_name> '<config_json>'
```

`run.sh` performs the mandatory rebuild protocol (clean ninja artifacts +
pip wheel cache) and runs three layered defenses against VSCode's
auto-revert of `Constants.hpp` (see [findings.md](findings.md#vscode-auto-revert-trap)).
Results land in `tools/raster_repro/results.jsonl` and (eventually,
once the hook is in place) auto-stub into today's daily file.

## Logging discipline

After each numbered test the `/log-test` skill writes a condensed entry to
the day's daily file. See [.claude/rules/test-logging.md](../../.claude/rules/test-logging.md).
