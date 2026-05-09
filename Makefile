# Convenience wrappers for the rasterizer reproducer harness.
# Real work lives in tools/raster_repro/. This Makefile is just shorthand
# so the run-by-typing path is shorter and harder to typo.
#
# Examples:
#   make synth                         # regenerate default icosphere input
#   make synth MODE=soup NUM_TRI=300000
#   make capture GLB=/path/to/asset.glb
#   make repro NAME=test_foo CFG='{"foo":1}'
#   make check                         # audit results.jsonl
#   make doctor                        # toolchain sanity report

PY := python3
REPRO_DIR := tools/raster_repro
INPUTS := $(REPRO_DIR)/inputs.pt
RESULTS := $(REPRO_DIR)/results.jsonl

# ---- defaults you can override on the command line --------------------------
MODE         ?= icosphere
SUBDIVISIONS ?= 7
NUM_TRI      ?= 300000
GRID_N       ?= 400
RESOLUTION   ?= 1024
SEED         ?= 42
GLB          ?=
NAME         ?=
CFG          ?= {}
FRAMES       ?= 120

.PHONY: help synth capture repro check doctor

help:
	@echo "Targets:"
	@echo "  synth          regenerate $(INPUTS) via tools/raster_repro/synth.py"
	@echo "                 (vars: MODE, SUBDIVISIONS, NUM_TRI, GRID_N, RESOLUTION, SEED)"
	@echo "  capture        load a .glb via capture_from_glb.py (var: GLB=path)"
	@echo "  repro          run the harness (vars: NAME=..., CFG='{...}')"
	@echo "  check          audit results.jsonl for schema drift"
	@echo "  doctor         print a one-shot toolchain sanity report"
	@echo ""
	@echo "All paths and the rebuild protocol are enforced inside run.sh."

synth:
	$(PY) $(REPRO_DIR)/synth.py \
	    --mode $(MODE) \
	    --subdivisions $(SUBDIVISIONS) \
	    --num-tri $(NUM_TRI) \
	    --grid-n $(GRID_N) \
	    --resolution $(RESOLUTION) \
	    --seed $(SEED) \
	    --output $(INPUTS)

capture:
	@if [ -z "$(GLB)" ]; then echo "set GLB=/path/to/asset.glb"; exit 2; fi
	$(PY) $(REPRO_DIR)/capture_from_glb.py \
	    --glb $(GLB) \
	    --resolution $(RESOLUTION) \
	    --output $(INPUTS)

repro:
	@if [ -z "$(NAME)" ]; then echo "set NAME=<test_name>"; exit 2; fi
	@$(REPRO_DIR)/run.sh '$(NAME)' '$(CFG)'

check:
	$(PY) $(REPRO_DIR)/check_results.py

doctor:
	@echo "=== git ==="
	@git rev-parse --abbrev-ref HEAD; git log -1 --format='%h %s'
	@echo "=== python ==="
	@$(PY) --version
	@echo "=== torch ==="
	@$(PY) -c "import torch; \
	    print('torch:', torch.__version__); \
	    print('hip:', getattr(torch.version,'hip',None)); \
	    print('cuda:', getattr(torch.version,'cuda',None)); \
	    print('available:', torch.cuda.is_available()); \
	    print('devices:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
	@echo "=== rocm ==="
	@if command -v rocminfo >/dev/null 2>&1; then \
	    rocminfo 2>/dev/null | grep -E '^\s*(Name:|Marketing Name:)' | head -6; \
	else echo "rocminfo not on PATH"; fi
	@if command -v rocm-smi >/dev/null 2>&1; then \
	    rocm-smi --showproductname 2>/dev/null | head -10; \
	else echo "rocm-smi not on PATH"; fi
	@echo "=== nvdiffrast-hip kernel constants ==="
	@grep -E 'CR_COARSE_WARPS|CR_FINE_MAX_WARPS' \
	    extensions/nvdiffrast-hip/csrc/common/cudaraster/impl/Constants.hpp 2>/dev/null \
	    || echo "(Constants.hpp not found)"
	@grep -E '__shared__ char s_smem' \
	    extensions/nvdiffrast-hip/csrc/common/hipraster/impl/RasterImpl_kernel.hip 2>/dev/null \
	    || true
	@echo "=== safe-mode flags ==="
	@grep -nE 'return; *// *SAFE-MODE' \
	    extensions/nvdiffrast-hip/csrc/common/hipraster/impl/CoarseRaster.inl \
	    extensions/nvdiffrast-hip/csrc/common/hipraster/impl/FineRaster.inl 2>/dev/null \
	    || echo "(no SAFE-MODE markers found)"
