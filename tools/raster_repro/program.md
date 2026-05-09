# Raster reproducer — program.md

Karpathy-style methodology file for the nvdiffrast HIP rasterizer investigation.
The agent reads this, the human edits it.

## Goal (current phase)

Find the maximum LDS (`__shared__`) size per workgroup that does not cause a GPU
page fault on RDNA3 (gfx1101) under ROCm 7.1.1, when running the unified
`rasterKernel` with realistic inputs (~290K-320K triangles, 1024x1024).

After the threshold is pinned down, the next phases are:

1. Restructure `coarseRaster` and `fineRaster` shared-memory layouts to fit under the threshold.
2. Re-enable the real `coarseRasterImpl` / `fineRasterImpl` bodies.
3. Fix the residual `segIdx=-1` OOB in `coarseRaster` (Test 42 [OOB-D] logs).

## Iteration loop

```
1. Edit ONE knob in the rasterizer source (currently: __shared__ size constant in RasterImpl_kernel.hip).
2. Run: tools/raster_repro/run.sh <test_name> <config_json>
   - Auto-detects source-newer-than-installed-.so and does a CLEAN rebuild
     (rm -rf build/ + pip cache clear + force-reinstall) when needed.
   - Otherwise skips rebuild (~0.003s test cycle).
   - Runs the replay harness on the cached input dump.
   - Output: one JSON line appended to tools/raster_repro/results.jsonl.
3. Read the result. Decide next config. Repeat.
```

## CRITICAL: stale-binary trap

Tests 55-59 of the original investigation logged 5 PASS results that were
later proven to be stale 32KB+no-op binaries despite source files saying
48KB+real-impl. The cause: pip's wheel cache + ninja's .o cache + the .py
files not changing meant `pip install . --no-build-isolation` would silently
serve us a prior binary. To detect:

- Inconsistent `coarseBlocks` / `fineBlocks` in ctor stderr (e.g. ctor reports
  `coarseBlocks=4 fineBlocks=3` while source says `__shared__ char s_smem[49152]`
  -- 48KB cannot allow 4 blocks per SM given 64KB max-smem-per-SM).
- Identical timing/output across very different kernel configs.

The auto-rebuild guard in run.sh handles this. To bypass (only when you know
the binary is current and want to skip the .so existence check):

```
SKIP_REBUILD=1 tools/raster_repro/run.sh <name> <config>
```

## Result schema (one JSON object per line in results.jsonl)

```
{
  "ts": "2026-05-09T12:34:56Z",        # UTC timestamp
  "config": {                          # whatever the human is sweeping this run
    "lds_size": 32768,
    "coarse_noop": true,
    "fine_noop": true
  },
  "build": {
    "rocm_version": "7.1.1",
    "gpu_arch": "gfx1101",
    "git_sha": "<short>",
    "rebuilt": true
  },
  "result": {
    "status": "PASS" | "CRASH" | "BUILD_FAIL",
    "frames_completed": 120,           # mesh preview is 120 frames
    "fault_addr": "0x..." | null,      # parsed from dmesg if CRASH
    "elapsed_s": 12.4,
    "stderr_tail": "..."               # last ~20 lines if CRASH
  }
}
```

## Hard rules for the agent

1. **Do not touch `prepare`-equivalent files.** The cached input dump
   (`tools/raster_repro/inputs.pt`) is fixed. If it is missing, regenerate it
   per the "Capturing inputs" section below — do not try to rasterize anything
   else as a substitute.
2. **One knob at a time.** Sweeping multiple variables at once is forbidden in
   this phase. The current knob is `__shared__ char s_smem[N]` in
   `extensions/nvdiffrast-hip/csrc/common/hipraster/impl/RasterImpl_kernel.hip`.
3. **Always log.** Every test result, even crashes, BUILD_FAILs, or aborted
   runs, gets a row in `results.jsonl`. The schema above is the contract.
4. **Treat `results.jsonl` as append-only.** Never edit or delete prior rows.
5. **Update CLAUDE.md** with the test number, config, result, and conclusion
   immediately after each result lands. (See `.claude/rules/test-logging.md`.)
6. **Do not run `app.py`** during this phase. The harness is the only path that
   exercises the rasterizer.

## Capturing inputs (one-time setup)

Two paths. Both produce `tools/raster_repro/inputs.pt` in the same schema.

**Path A — live capture from a real TRELLIS run** (preferred when working):

```
RASTER_DUMP=tools/raster_repro/inputs.pt \
ATTN_BACKEND=sdpa XFORMERS_DISABLED=1 SPARSE_BACKEND=torchsparse \
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
python tools/raster_repro/capture.py example.py
```

**Path B — synthetic mesh** (fallback when the slat sampling stage is broken,
which is currently the case on Caleb's stack with torch 2.9.1+rocm6.4 / system
ROCm 7.2.1):

```
python tools/raster_repro/synth.py --output tools/raster_repro/inputs.pt
```

Synthetic input is a 400x400 heightfield grid (318,402 triangles), close
enough to the real workload (290K-320K) for the LDS-size investigation. Once
the live path is repaired, regenerate with Path A for the OOB fix phase.

## Fault containment

- `run.sh` runs the harness in a subprocess. A GPU page fault that kills the
  Python process is captured as `status: CRASH`.
- A GPU page fault that hangs the GPU but leaves the process alive is captured
  as a per-frame timeout (default 30s/frame) -> `status: CRASH`,
  `error: "frame_timeout"`.
- A fault that takes the host system down is the failure mode we cannot capture
  here. If you observe this twice in a row for the same config, STOP and ask
  the human before continuing.

## What "PASS" means in this phase

Replay completes 120 frames (matching the live mesh preview length) without:

- A GPU page fault (process crash or dmesg fault entry)
- A frame timeout
- A non-zero return code from the harness

A PASS does NOT mean the output is visually correct yet — coarse and fine
impls are still no-op while we hunt the LDS threshold. Output validation comes
in a later phase.

## What gets edited by the agent

- `extensions/nvdiffrast-hip/csrc/common/hipraster/impl/RasterImpl_kernel.hip`
  (specifically the `__shared__ char s_smem[N]` constant on the rasterKernel
  entry point — nothing else)

## What is OFF-LIMITS to the agent

- All other files under `extensions/nvdiffrast-hip/`
- `tools/raster_repro/{capture.py, harness.py, run.sh}` (the harness itself)
- `trellis/`, `app.py`, anything outside the rasterizer code path
