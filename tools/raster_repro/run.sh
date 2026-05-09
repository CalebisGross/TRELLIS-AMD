#!/usr/bin/env bash
# Driver: runs harness.py in a subprocess, captures result, appends to
# results.jsonl. Catches subprocess crashes (return code, missing stdout JSON).
#
# Usage:
#   tools/raster_repro/run.sh <test_name> [<config_json>] [-- <harness_args...>]
#
# Examples:
#   tools/raster_repro/run.sh test54_lds40k '{"lds_size":40960,"coarse_noop":true}'
#   tools/raster_repro/run.sh test54_lds40k '{"lds_size":40960}' -- --frames 60
#
# Environment:
#   PYTHON          python interpreter (default: python from active venv)
#   RESULTS_FILE    path to JSONL log (default: tools/raster_repro/results.jsonl)
#   SUBPROC_TIMEOUT overall subprocess timeout in seconds (default: 600)
#   SKIP_REBUILD    set to 1 to skip the auto-rebuild check (advanced use only)

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --- Constants.hpp warp-count override -------------------------------------
# CRITICAL: hipraster/impl/Constants.hpp is regenerated from
# cudaraster/impl/Constants.hpp by PyTorch's hipify on every build. So we
# sed the CUDA source (the canonical version), not the HIP one. Layer 3 then
# verifies the BUILT BINARY actually has the values we asked for.
#
# Set env vars to override; defaults match the AMD-tuned canonical values:
#   COARSE_WARPS_OVERRIDE  (default 8, reduced from 16 to fit RDNA3 register budget)
#   FINE_WARPS_OVERRIDE    (default 12, reduced from 20 to shrink FineSmem to 31KB)
NVDR_KSRC="extensions/nvdiffrast-hip/csrc/common/hipraster/impl"
NVDR_KSRC_CUDA="extensions/nvdiffrast-hip/csrc/common/cudaraster/impl"
COARSE_WARPS_OVERRIDE="${COARSE_WARPS_OVERRIDE:-8}"
FINE_WARPS_OVERRIDE="${FINE_WARPS_OVERRIDE:-12}"

if [[ "${SKIP_REBUILD:-0}" != "1" ]]; then
  sed -i.bak -E "s/^#define CR_COARSE_WARPS[[:space:]]+[0-9]+.*$/#define CR_COARSE_WARPS         $COARSE_WARPS_OVERRIDE/" "$NVDR_KSRC_CUDA/Constants.hpp"
  sed -i -E "s/^#define CR_FINE_MAX_WARPS[[:space:]]+[0-9]+.*$/#define CR_FINE_MAX_WARPS       $FINE_WARPS_OVERRIDE/" "$NVDR_KSRC_CUDA/Constants.hpp"
  rm -f "$NVDR_KSRC_CUDA/Constants.hpp.bak"
  CHECK_COARSE="$(grep -E '^#define CR_COARSE_WARPS\b' "$NVDR_KSRC_CUDA/Constants.hpp" | grep -oE '[0-9]+' | head -1)"
  CHECK_FINE="$(grep -E '^#define CR_FINE_MAX_WARPS\b' "$NVDR_KSRC_CUDA/Constants.hpp" | grep -oE '[0-9]+' | head -1)"
  if [[ "$CHECK_COARSE" != "$COARSE_WARPS_OVERRIDE" ]] || [[ "$CHECK_FINE" != "$FINE_WARPS_OVERRIDE" ]]; then
    echo "[run.sh] LAYER-1 ABORT: post-sed cudaraster Constants.hpp shows COARSE=$CHECK_COARSE FINE=$CHECK_FINE, wanted COARSE=$COARSE_WARPS_OVERRIDE FINE=$FINE_WARPS_OVERRIDE." >&2
    exit 4
  fi
fi

# --- Pre-flight risk check -------------------------------------------------
# Refuse to run configurations we know will fault the GPU and likely take
# down Xorg. Override with FORCE_RISKY=1 ONLY when running from a TTY, an
# SSH session that doesn't use the local X server, or the MI300X droplet.
LDS_SIZE="$(grep -oE 's_smem\[[0-9]+\]' "$NVDR_KSRC/RasterImpl_kernel.hip" 2>/dev/null | grep -oE '[0-9]+' || echo 0)"
COARSE_REAL=1
FINE_REAL=1
if grep -q "^  return; // SAFE-MODE" "$NVDR_KSRC/CoarseRaster.inl" 2>/dev/null; then COARSE_REAL=0; fi
if grep -q "^    return; // SAFE-MODE" "$NVDR_KSRC/FineRaster.inl" 2>/dev/null; then FINE_REAL=0; fi

# Pre-flight uses the override values (what we just sed'd in), not whatever
# Constants.hpp currently shows (could be reverted between sed and pre-flight).
COARSE_WARPS="$COARSE_WARPS_OVERRIDE"
FINE_WARPS="$FINE_WARPS_OVERRIDE"
# FineSmem  = cover_lut(6144) + WARPS*(5*256 U32 + 64 U64 + 80 U32) = 6144 + WARPS*2112
FINE_SMEM_SIZE=$(( 6144 + FINE_WARPS * 2112 ))
# CoarseSmem (post-refactor): warpEmitMask + warpEmitPrefixSum moved to global memory.
# Remaining: scalars+fixed(8356) + WARPS*(48 U32 scanTemp only) = 8356 + WARPS*192.
COARSE_SMEM_SIZE=$(( 8356 + COARSE_WARPS * 192 ))

ABORT_REASONS=""
# Rule 1: LDS > 32KB -> occupancy=1 -> known crash on RDNA3 (Tests 52, 60)
if [[ "$LDS_SIZE" -gt 32768 ]]; then
  ABORT_REASONS+="LDS_SIZE=$LDS_SIZE > 32768 (occupancy=1 crashes on RDNA3, Test 60). "
fi
# Rule 2: Real fineRaster needs LDS_SIZE >= computed FineSmem size
if [[ "$FINE_REAL" == "1" ]] && [[ "$LDS_SIZE" -lt "$FINE_SMEM_SIZE" ]]; then
  ABORT_REASONS+="real fineRasterImpl + LDS_SIZE=$LDS_SIZE < FineSmem(${FINE_SMEM_SIZE},warps=$FINE_WARPS) -> OOB. "
fi
# Rule 3: Real coarseRaster needs LDS_SIZE >= computed CoarseSmem size
if [[ "$COARSE_REAL" == "1" ]] && [[ "$LDS_SIZE" -lt "$COARSE_SMEM_SIZE" ]]; then
  ABORT_REASONS+="real coarseRasterImpl + LDS_SIZE=$LDS_SIZE < CoarseSmem(${COARSE_SMEM_SIZE},warps=$COARSE_WARPS) -> OOB. "
fi

if [[ -n "$ABORT_REASONS" ]]; then
  if [[ "${FORCE_RISKY:-0}" != "1" ]]; then
    echo "[run.sh] PRE-FLIGHT REFUSED" >&2
    echo "[run.sh]   Reasons: $ABORT_REASONS" >&2
    echo "[run.sh]   This config is known to crash the GPU and may take down Xorg." >&2
    echo "[run.sh]   To override (only from TTY/SSH where X is not affected): FORCE_RISKY=1" >&2
    exit 3
  else
    echo "[run.sh] WARNING: forced past pre-flight: $ABORT_REASONS" >&2
  fi
fi

# --- Auto-rebuild guard ----------------------------------------------------
# If any nvdiffrast-hip source file is newer than the installed .so, force a
# CLEAN rebuild (build dir + pip wheel cache + force-reinstall). This avoids
# the stale-binary trap from Tests 55-59 where pip's wheel cache silently
# served us 32KB-no-op binaries despite source files saying otherwise.
NVDR_SRC_DIR="extensions/nvdiffrast-hip/csrc"
PYTHON_BIN_PRE="${PYTHON:-python}"
# find_spec doesn't load the module, so HIP libs don't need to be present
INSTALLED_SO="$("$PYTHON_BIN_PRE" -c 'import importlib.util; s = importlib.util.find_spec("_nvdiffrast_c"); print(s.origin if s else "")' 2>/dev/null || echo "")"

if [[ "${SKIP_REBUILD:-0}" != "1" ]]; then
  NEEDS_REBUILD=0
  if [[ -z "$INSTALLED_SO" ]] || [[ ! -f "$INSTALLED_SO" ]]; then
    NEEDS_REBUILD=1
    REBUILD_REASON="installed .so missing"
  elif [[ -n "$(find "$NVDR_SRC_DIR" -newer "$INSTALLED_SO" -type f 2>/dev/null | head -n1)" ]]; then
    NEEDS_REBUILD=1
    REBUILD_REASON="source newer than installed .so"
  fi

  if [[ "$NEEDS_REBUILD" == "1" ]]; then
    echo "[run.sh] rebuild triggered: $REBUILD_REASON" >&2
    rm -rf extensions/nvdiffrast-hip/build/
    "$PYTHON_BIN_PRE" -m pip cache remove "nvdiffrast*" >/dev/null 2>&1 || true
    if ! (cd extensions/nvdiffrast-hip && "$PYTHON_BIN_PRE" -m pip install . --no-build-isolation --force-reinstall --no-deps >/tmp/raster_repro_build.log 2>&1); then
      echo "[run.sh] BUILD FAILED -- last 20 lines of build log:" >&2
      tail -n 20 /tmp/raster_repro_build.log >&2
      RC=2
      RESULT_BLOCK="{\"status\":\"BUILD_FAIL\",\"frames_completed\":0,\"elapsed_s\":0,\"error\":\"build failed; see /tmp/raster_repro_build.log\",\"first_frame_s\":null}"
      BUILD_FAILED=1
    fi
    echo "[run.sh] rebuild done" >&2
  fi
fi
BUILD_FAILED="${BUILD_FAILED:-0}"

# --- Layer 3: built-binary verification ------------------------------------
# Smoke-probe the binary by constructing a RasterizeCudaContext and parsing
# the ctor diagnostic. If fineWarps in the binary != FINE_WARPS_OVERRIDE,
# the build was contaminated (Constants.hpp was reverted mid-build).
if [[ "$BUILD_FAILED" != "1" ]] && [[ "${SKIP_REBUILD:-0}" != "1" ]]; then
  SMOKE_OUT="$(timeout 30 "$PYTHON_BIN_PRE" -c '
import os, sys
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["SPARSE_BACKEND"] = "torchsparse"
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
import torch
import nvdiffrast.torch as dr
ctx = dr.RasterizeCudaContext()
sys.stderr.flush()
' 2>&1 || true)"
  ACTUAL_FINE_BIN="$(echo "$SMOKE_OUT" | grep -oE 'fineWarps=[0-9]+' | grep -oE '[0-9]+' | head -1 || echo unknown)"
  if [[ "$ACTUAL_FINE_BIN" != "$FINE_WARPS_OVERRIDE" ]]; then
    echo "[run.sh] LAYER-3 ABORT: built binary reports fineWarps=$ACTUAL_FINE_BIN, expected $FINE_WARPS_OVERRIDE" >&2
    echo "[run.sh] Constants.hpp was reverted DURING the build. Close it in VSCode and retry." >&2
    echo "[run.sh] (Smoke probe stderr tail follows for debugging.)" >&2
    echo "$SMOKE_OUT" | tail -n 5 >&2
    exit 5
  fi
  echo "[run.sh] binary verified: fineWarps=$ACTUAL_FINE_BIN" >&2
fi

TEST_NAME="${1:-unnamed}"
CONFIG_JSON="${2-}"
[ -z "$CONFIG_JSON" ] && CONFIG_JSON='{}'
shift 2 2>/dev/null || true

# Drop the optional `--` separator if present.
if [[ "${1:-}" == "--" ]]; then shift; fi
HARNESS_ARGS=("$@")

PYTHON_BIN="${PYTHON:-python}"
RESULTS_FILE="${RESULTS_FILE:-tools/raster_repro/results.jsonl}"
SUBPROC_TIMEOUT="${SUBPROC_TIMEOUT:-600}"

mkdir -p "$(dirname "$RESULTS_FILE")"

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
ROCM_VERSION="$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
GPU_ARCH="$(rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx/ {print $2; exit}')"
GPU_ARCH="${GPU_ARCH:-unknown}"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

STDOUT_FILE="$(mktemp)"
STDERR_FILE="$(mktemp)"
trap 'rm -f "$STDOUT_FILE" "$STDERR_FILE"' EXIT

# Snapshot dmesg position so we can grep new fault entries after the run.
DMESG_BEFORE_LINES="$(dmesg 2>/dev/null | wc -l || echo 0)"

if [[ "$BUILD_FAILED" == "1" ]]; then
  RC=2
else
  set +e
  timeout --kill-after=10 "$SUBPROC_TIMEOUT" \
    "$PYTHON_BIN" tools/raster_repro/harness.py "${HARNESS_ARGS[@]}" \
    >"$STDOUT_FILE" 2>"$STDERR_FILE"
  RC=$?
  set -e
fi

# Pull harness's JSON line if it emitted one.
HARNESS_JSON="$(tail -n 1 "$STDOUT_FILE" | grep -E '^\{' || true)"
if [[ "$BUILD_FAILED" == "1" ]]; then
  HARNESS_JSON="$RESULT_BLOCK"
fi

# Look for "Memory access fault" in dmesg lines that appeared after the run.
DMESG_AFTER="$(dmesg 2>/dev/null | tail -n +$((DMESG_BEFORE_LINES + 1)) || true)"
FAULT_LINE="$(echo "$DMESG_AFTER" | grep -m1 'Memory access fault' || true)"
FAULT_ADDR="null"
if [[ -n "$FAULT_LINE" ]]; then
  FAULT_ADDR="\"$(echo "$FAULT_LINE" | grep -oE '0x[0-9a-fA-F]+' | head -n1 || echo unknown)\""
fi

STDERR_TAIL="$(tail -n 20 "$STDERR_FILE" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')"

# Build final JSON record.
if [[ -n "$HARNESS_JSON" ]]; then
  RESULT_BLOCK="$HARNESS_JSON"
else
  if [[ "$RC" -eq 124 ]]; then
    ERR="\"subprocess_timeout(${SUBPROC_TIMEOUT}s)\""
  else
    ERR="\"subprocess_exit_${RC}_no_json\""
  fi
  RESULT_BLOCK="{\"status\":\"CRASH\",\"frames_completed\":0,\"elapsed_s\":0,\"error\":${ERR},\"first_frame_s\":null}"
fi

# Splice extras into the result block.
FINAL_JSON="$(python3 -c '
import json, sys
result = json.loads(sys.argv[1])
result["fault_addr"] = json.loads(sys.argv[2])
result["stderr_tail"] = json.loads(sys.argv[3])
result["return_code"] = int(sys.argv[4])
record = {
    "ts": sys.argv[5],
    "test_name": sys.argv[6],
    "config": json.loads(sys.argv[7]),
    "build": {"git_sha": sys.argv[8], "rocm_version": sys.argv[9], "gpu_arch": sys.argv[10]},
    "result": result,
}
print(json.dumps(record))
' "$RESULT_BLOCK" "$FAULT_ADDR" "$STDERR_TAIL" "$RC" "$TS" "$TEST_NAME" "$CONFIG_JSON" "$GIT_SHA" "$ROCM_VERSION" "$GPU_ARCH")"

echo "$FINAL_JSON" >> "$RESULTS_FILE"
echo "$FINAL_JSON"

if [[ "$(echo "$FINAL_JSON" | python3 -c 'import json,sys;print(json.loads(sys.stdin.read())["result"]["status"])')" == "PASS" ]]; then
  exit 0
else
  exit 1
fi
