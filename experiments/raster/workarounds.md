# Failed workarounds — don't revisit

Tried and confirmed not to fix the current bug stack. Listed in roughly
chronological order. The "Test(s)" column references entries in
[daily/](daily/) and the legacy archive.

| Workaround | Test(s) | Result |
| --- | --- | --- |
| `hipDeviceSynchronize` between each kernel | 8 | CRASH |
| `hipStreamSynchronize` between each kernel | 20-24, 40 | CRASH |
| `static CRParams` (persistent host memory) | 25, 28 | CRASH |
| `static CRParams` + `usleep(1)` | 26 | CRASH |
| stderr delay between dispatches | 27, 29 | CRASH (delayed) |
| `hipStreamSynchronize` between dispatch pairs | 39 | CRASH after 3 frames |
| Bounds-check all `coarseRaster` writes | 42, 42b | CRASH (no OOB caught) |
| Two-stream dispatch (2 kernels per stream) | 44 | CRASH |
| Pad `tileSegData` with 4 KB guard | 45 | CRASH |
| Merge into 2 unified kernels + dynamic smem | 46 | CRASH |
| HIP Graph dispatch (single command buffer) | 47 | CRASH |
| Split into 2 `launchStages` calls (2 dispatches each) | 48 | CRASH |
| 1 dispatch of `rasterKernel` (coarseRaster only) | 49 | CRASH |
| No-op `rasterKernel` (`coarseRasterImpl` returns immediately) | 50 | CRASH |
| Static `__shared__` union (~48 KB) + real impls | 52 | CRASH |
| `__syncwarp` → `__builtin_amdgcn_wave_barrier` | 79 | CRASH |
| `minBlocks=1` launch_bounds + real coarse | 81 | CRASH (719) |
| `minBlocks=2` launch_bounds + real coarse (rocm7) | 84 | CRASH (719) |
| `CR_COARSE_WARPS=8` reduction (rocm7) | 85 | CRASH (719) |
| Torch upgrade to 2.10.0+rocm7.0 | 80 | CRASH (719) |

## Crash address pattern (legacy, RDNA3 + ROCm 6.4.2 era)

Every fault hit exactly 1 page (4096 bytes) before `p.tileSegData`:

| Test | Fault Address | tileSegData | Iterations |
|------|---------------|-------------|------------|
| 27   | 0x7a214d9ff000 | 0x7a214da00000 | 5 |
| 28   | 0x79a8eb3ff000 | 0x79a8eb400000 | 1 |
| 29   | 0x74b5b73ff000 | 0x74b5b7400000 | 4 |
| 35   | 0x703b363ff000 | (not logged)   | 1 |
| 37   | 0x747f4b1ff000 | 0x747f4b200000 | 5 |
| 39   | 0x7fd10f1ff000 | 0x7fd10f200000 | 3 |
| 42   | 0x7dcec89ff000 | 0x7dcec8a00000 | 3 |

After the rocm 7 upgrade and the `hipErrorLaunchOutOfResources` symptom,
faults no longer present as guard-page hits — they fail at launch.
