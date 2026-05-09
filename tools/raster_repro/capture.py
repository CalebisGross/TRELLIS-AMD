"""
One-shot capture of nvdiffrast.torch.rasterize() inputs.

Usage:
    RASTER_DUMP=tools/raster_repro/inputs.pt \\
    ATTN_BACKEND=sdpa XFORMERS_DISABLED=1 SPARSE_BACKEND=torchsparse \\
    python tools/raster_repro/capture.py [path/to/script.py]

Default driver script is example.py at the repo root. The patch dumps the
first rasterize() call's inputs to RASTER_DUMP, then os._exit(0) so we never
actually run the rasterizer kernel (which would risk crashing the host).
"""

import os
import runpy
import sys
import time

import torch


def _install_capture_patch(dump_path: str) -> None:
    import nvdiffrast.torch as dr

    original_rasterize = dr.rasterize
    captured = {"done": False}

    def patched_rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
        if captured["done"]:
            return original_rasterize(glctx, pos, tri, resolution, ranges, grad_db)
        captured["done"] = True

        pos_cpu = pos.detach().cpu().contiguous()
        tri_cpu = tri.detach().cpu().contiguous()
        ranges_cpu = ranges.detach().cpu().contiguous() if ranges is not None else None
        instance_mode = pos.dim() == 3

        payload = {
            "pos": pos_cpu,
            "tri": tri_cpu,
            "resolution": tuple(resolution),
            "ranges": ranges_cpu,
            "instance_mode": instance_mode,
            "pos_shape": tuple(pos.shape),
            "tri_shape": tuple(tri.shape),
            "captured_at": time.time(),
        }

        os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
        torch.save(payload, dump_path)

        sys.stderr.write(
            f"[capture] dumped to {dump_path}: pos={tuple(pos.shape)} "
            f"tri={tuple(tri.shape)} res={tuple(resolution)} "
            f"instance_mode={instance_mode}\n"
        )
        sys.stderr.flush()
        os._exit(0)

    dr.rasterize = patched_rasterize


def main() -> None:
    dump_path = os.environ.get("RASTER_DUMP")
    if not dump_path:
        sys.stderr.write("RASTER_DUMP env var must be set to a target file path\n")
        sys.exit(2)

    driver = sys.argv[1] if len(sys.argv) > 1 else "example.py"
    if not os.path.exists(driver):
        sys.stderr.write(f"driver script not found: {driver}\n")
        sys.exit(2)

    _install_capture_patch(dump_path)
    sys.stderr.write(f"[capture] patch installed, running {driver}\n")
    sys.stderr.flush()

    driver_dir = os.path.dirname(os.path.abspath(driver)) or os.getcwd()
    if driver_dir not in sys.path:
        sys.path.insert(0, driver_dir)

    sys.argv = [driver] + sys.argv[2:]
    runpy.run_path(driver, run_name="__main__")

    sys.stderr.write("[capture] driver finished without calling rasterize()\n")
    sys.exit(3)


if __name__ == "__main__":
    main()
