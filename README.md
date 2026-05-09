# TRELLIS-AMD

**TRELLIS running on AMD GPUs with ROCm** - Image to 3D Asset Generation

This is a fork of [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) modified to run on AMD consumer GPUs (tested on RX 7800 XT with ROCm 7.2.1, torch 2.10.0+rocm7.0).

> **Status (May 2026): Fully operational on RX 7800 XT.** Tested end-to-end:
> image → 3D asset → textured GLB, including mesh rendering, hole filling, and
> texture baking on a 16 GB consumer card. The multi-month rasterizer
> investigation that unblocked this is documented in
> [experiments/raster/](experiments/raster/).

## Features

| Feature | Status | Timing |
|---------|--------|--------|
| ✅ 3D Model Generation | Working | ~45 seconds |
| ✅ Gaussian Splatting | Working (145+ it/s) | ~30 seconds |
| ✅ Gaussian Export (.ply) | Working | Instant |
| ✅ Mesh Extraction | Working | ~60 seconds |
| ✅ GLB Export with Textures | Working | **5-10 minutes** |

> **⚠️ GLB Export Takes 5-10 Minutes**: This is normal! The console will show progress through 5 steps. Your system will be under heavy load during texture baking - this is expected.

## Requirements

- AMD GPU (tested: RX 7800 XT, RDNA3 / gfx1101)
- ROCm 7.0+ (tested on system ROCm 7.2.1, torch 2.10.0+rocm7.0)
- Python 3.10+
- 16 GB VRAM (the pipeline is split into staged phases to fit; see [example.py](example.py))

## Quick Start

Install libsparsehash-dev(required for building torchsparse)


Ubuntu/Debian:
```bash
sudo apt-get install libsparsehash-dev
```
Fedora: 
```bash
sudo dnf install sparsehash-devel
```
Arch Linux
```bash
sudo pacman -S google-sparsehash
```
```bash
# Clone the repository
git clone https://github.com/CalebisGross/TRELLIS-AMD
cd TRELLIS-AMD

# Run the installation script
chmod +x install_amd.sh
./install_amd.sh

# Activate environment and run
source .venv/bin/activate
ATTN_BACKEND=sdpa XFORMERS_DISABLED=1 SPARSE_BACKEND=torchsparse \
  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python app.py
```

Then open http://localhost:7860 in your browser.

## What's Different from Original TRELLIS?

### Custom Extensions (AMD-compatible)

| Extension | Modification |
|-----------|-------------|
| **nvdiffrast-hip** | AMD-safe coarse rasterizer, HIP warp intrinsic macros |
| **diff-gaussian-rasterization** | Manual HIP build script, buffer initialization fixes |
| **torchsparse** | Built with `FORCE_CUDA=1` for HIP GPU backend |

### Application Modifications
- HIP rasterizer (CoarseRaster + FineRaster) bounds-check fix for the
  `triHeader[i].misc` OOB on RDNA3 — see [experiments/raster/findings.md](experiments/raster/findings.md#bug-6--triheaderimisc-oob-on-rdna3-resolved-2026-05-09)
- `_fill_holes` pole-clamps the Hammersley camera distribution so views
  directly above/below the mesh don't NaN `view_look_at` and hang
  coarseRaster (default num_views also dropped 1000 → 100 for speed)
- Added progress logging for GLB export
- `example.py` splits the pipeline into staged phases, moving idle
  submodels to CPU between phases so the full run fits in 16 GB VRAM

## Processing Time Reference

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| 3D Generation (Sampling) | ~45s | 12 steps of diffusion |
| Gaussian Export | Instant | Saves .ply file |
| GLB Export | **5-10 min** | Heavy CPU+GPU load is normal |

The GLB export shows progress in console:
```
[GLB Export] Starting GLB extraction (this takes 5-10 minutes)...
[GLB Export] Step 1/5: Mesh postprocessing...
[GLB Export] Step 2/5: UV parametrization...
[GLB Export] Step 3/5: Rendering multiview observations (100 views)...
[GLB Export] Step 4/5: Baking texture (2500 optimization steps)...
[GLB Export] Step 5/5: Finalizing GLB mesh...
[GLB Export] Complete!
```

## Known Limitations

1. **Performance**: Coarse rasterizer is serialized and slower than NVIDIA's warp-parallel version
2. **~7% silent triangle culls**: The Bug 6 bounds-check fix culls triangles
   with an out-of-range `triHeader[i].misc` from triangleSetup. Visual impact
   is small but the underlying invariant violation is unresolved. See
   [experiments/raster/findings.md](experiments/raster/findings.md#bug-6--triheaderimisc-oob-on-rdna3-resolved-2026-05-09)
   for the Phase C root-cause hypothesis.
3. **fill_holes uses 100 views, not 1000**: TRELLIS upstream rasterizes 1000
   Hammersley-distributed views to detect invisible faces. We clamp views
   away from the world-up poles (otherwise the HIP rasterizer hangs on
   degenerate view matrices) and use 100 views. Hole detection quality is
   visually indistinguishable, and step 1 of GLB extract is now ~10x faster.

## Troubleshooting

### GPU Hang/Crash
Ensure you're using ROCm 7.0+ and PyTorch built for ROCm (torch 2.10.0+rocm7.0
or newer is recommended).

### Empty Mesh
Confirm the input image actually has a foreground subject after rembg
background removal. If so, raise the `Mesh Simplify` slider toward 0 in
the UI to keep more triangles, or pass `simplify=0.0` to `to_glb()`.

### CUDA Symbol Errors  
Make sure you're using the AMD-modified extensions in this repo, not the original CUDA ones.

### torchsparse "no attribute" Error
Rebuild with: `cd extensions/torchsparse && CUDA_HOME=/opt/rocm FORCE_CUDA=1 pip install . --no-build-isolation`

## Credits

- Original [TRELLIS](https://github.com/microsoft/TRELLIS) by Microsoft
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast) by NVIDIA
- AMD GPU modifications developed through extensive debugging of HIP compatibility issues

## License

See original licenses for TRELLIS, nvdiffrast, and diff-gaussian-rasterization.
