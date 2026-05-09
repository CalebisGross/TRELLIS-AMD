# TRELLIS-AMD

**TRELLIS running on AMD GPUs with ROCm** - Image to 3D Asset Generation

This is a fork of [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) modified to run on AMD consumer GPUs (tested on RX 7800 XT with ROCm 6.4.2).

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

- AMD GPU (tested: RX 7800 XT, RDNA3)
- ROCm 6.4+ 
- Python 3.10+
- ~16GB VRAM recommended

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
- Switched to OpenGL rasterization backend (avoids HIP rasterizer bugs)
- Disabled `fill_holes` in mesh postprocessing (avoids visibility check issues)
- Added progress logging for GLB export

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

1. **Mesh Preview**: May show grey - the actual export works correctly
2. **fill_holes Disabled**: Small holes in meshes may not be filled
3. **Performance**: Simplified coarse rasterizer is slower than NVIDIA-optimized version

## Troubleshooting

### GPU Hang/Crash
Ensure you're using ROCm 6.4+ and PyTorch built for ROCm.

### Empty Mesh
Check that `fill_holes=False` is set in `trellis/utils/postprocessing_utils.py`.

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
