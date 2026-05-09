"""
Build a rasterizer input dump from a saved .glb (or .ply, .obj, .stl —
anything trimesh can load). Useful when the live TRELLIS pipeline can't
run end-to-end (e.g., slat sampling broken on the current toolchain) but
we have prior generated assets to test against.

The output matches the schema of synth.py / capture.py: a torch.save'd
payload of {pos, tri, resolution, ranges, instance_mode, ...} that
harness.py can replay.

Usage:
    python tools/raster_repro/capture_from_glb.py --glb path/to/asset.glb
    python tools/raster_repro/capture_from_glb.py --glb path --output /tmp/x.pt
    python tools/raster_repro/capture_from_glb.py --glb path --dry-run
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# Reuse the projection from synth.py to keep the camera consistent across
# every input the harness sees.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import _perspective_project  # noqa: E402


def load_mesh(glb_path: str):
    """Load a mesh file via trimesh, concatenating any sub-geometries."""
    import trimesh

    obj = trimesh.load(glb_path, force="mesh", process=False)
    if isinstance(obj, trimesh.Scene):
        # Combine all geometries in the scene into one mesh.
        meshes = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"no triangle meshes found in {glb_path}")
        obj = trimesh.util.concatenate(meshes)

    if not isinstance(obj, trimesh.Trimesh):
        raise ValueError(f"unexpected object type from trimesh.load: {type(obj)}")

    return obj


def normalize_mesh(verts: torch.Tensor, target_extent: float = 1.6):
    """Center the mesh on origin and scale so the longest bbox edge is target_extent.

    target_extent=1.6 puts the mesh well inside the unit clip volume after
    the default perspective projection (camera at z=+2, fov=45deg).
    """
    bbox_min = verts.min(dim=0).values
    bbox_max = verts.max(dim=0).values
    center = (bbox_min + bbox_max) * 0.5
    extent = (bbox_max - bbox_min).max().item()
    if extent <= 0:
        raise ValueError("degenerate mesh: zero extent")
    scale = target_extent / extent
    return (verts - center) * scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True,
                        help="path to a .glb / .ply / .obj / .stl asset")
    parser.add_argument("--output", default="tools/raster_repro/inputs.pt",
                        help="path to write the input dump")
    parser.add_argument("--batch", type=int, default=1,
                        help="instance-mode batch size")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="output frame resolution (square)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="skip recenter+rescale (use raw vertex positions)")
    parser.add_argument("--target-extent", type=float, default=1.6,
                        help="bbox max-edge target after normalization")
    parser.add_argument("--dry-run", action="store_true",
                        help="print payload schema and exit; do not write the file")
    args = parser.parse_args()

    if not os.path.isfile(args.glb):
        sys.stderr.write(f"[capture-glb] file not found: {args.glb}\n")
        return 2

    mesh = load_mesh(args.glb)
    verts_np = mesh.vertices.astype("float32")
    faces_np = mesh.faces.astype("int32")
    n_tri = faces_np.shape[0]
    n_vert = verts_np.shape[0]

    verts = torch.from_numpy(verts_np)
    if not args.no_normalize:
        verts = normalize_mesh(verts, target_extent=args.target_extent)

    pos = _perspective_project(verts)
    pos = pos.unsqueeze(0).expand(args.batch, -1, -1).contiguous().to(torch.float32)
    tri = torch.from_numpy(faces_np).contiguous()

    payload = {
        "pos": pos,
        "tri": tri,
        "resolution": (args.resolution, args.resolution),
        "ranges": None,
        "instance_mode": True,
        "pos_shape": tuple(pos.shape),
        "tri_shape": tuple(tri.shape),
        "captured_at": time.time(),
        "synthetic": False,
        "source_glb": os.path.abspath(args.glb),
        "capture_args": {
            "batch": args.batch,
            "resolution": args.resolution,
            "normalize": not args.no_normalize,
            "target_extent": args.target_extent,
        },
    }

    sys.stderr.write(
        f"[capture-glb] {args.glb}: pos={tuple(pos.shape)} "
        f"tri={tuple(tri.shape)} ({n_tri} triangles, {n_vert} verts) "
        f"res={args.resolution}x{args.resolution}\n"
    )

    if args.dry_run:
        sys.stderr.write(f"[capture-glb] dry-run: would write {args.output}\n")
        return 0

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, args.output)
    sys.stderr.write(f"[capture-glb] wrote {args.output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
