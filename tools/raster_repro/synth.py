"""
Synthesize a realistic-sized mesh and save in the same format capture.py
produces. Used when we can't run the full TRELLIS pipeline to capture real
inputs (e.g., when the slat sampling stage is broken on the current stack).

The mesh is a 400x400 heightfield grid tessellated into ~319K triangles, which
matches the size range observed in CLAUDE.md (290K-320K triangles for real
TRELLIS meshes). Positions span most of the clip volume with small per-vertex
z noise so coarse-rasterizer tile assignment sees varied work.

Usage:
    python tools/raster_repro/synth.py --output tools/raster_repro/inputs.pt

The output file is a torch.save dump with the same schema as capture.py.
"""

import argparse
import math
import os
import sys
import time

import torch


def synth_grid_mesh(grid_n: int, batch: int, seed: int):
    """Generate a [grid_n x grid_n] heightfield grid as a clip-space mesh.

    Triangles are small (~3 pixels at 1024x1024). Most fit in a single tile
    so coarseRaster takes a fast path -- NOT useful for testing tile-segment
    code. Use --mode soup for real coarseRaster stress.
    """
    g = torch.Generator().manual_seed(seed)

    xs = torch.linspace(-0.9, 0.9, grid_n)
    ys = torch.linspace(-0.9, 0.9, grid_n)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    z_noise = (torch.rand(grid_n, grid_n, generator=g) - 0.5) * 0.1
    radial = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    z = 0.3 * torch.cos(radial * math.pi) + z_noise

    pos_xyz = torch.stack([grid_x.flatten(), grid_y.flatten(), z.flatten()], dim=-1)
    pos = torch.cat([pos_xyz, torch.ones(pos_xyz.shape[0], 1)], dim=-1)
    pos = pos.unsqueeze(0).expand(batch, -1, -1).contiguous()

    idx = torch.arange(grid_n * grid_n).reshape(grid_n, grid_n)
    a = idx[:-1, :-1].flatten()
    b = idx[:-1, 1:].flatten()
    c = idx[1:, :-1].flatten()
    d = idx[1:, 1:].flatten()
    tri1 = torch.stack([a, b, d], dim=-1)
    tri2 = torch.stack([a, d, c], dim=-1)
    tri = torch.cat([tri1, tri2], dim=0).to(torch.int32)

    return pos.to(torch.float32), tri


def _perspective_project(verts_world: torch.Tensor) -> torch.Tensor:
    """Apply a typical perspective projection like mesh_renderer.py does.

    Camera at +z looking at origin, fov=45deg, near=0.1, far=10. Returns
    [N, 4] clip-space coords with non-trivial w (NOT all 1.0). This matches
    what dr.rasterize sees in the live TRELLIS workload.
    """
    eye_z = 2.0
    fov = math.pi / 4
    near = 0.1
    far = 10.0
    f = 1.0 / math.tan(fov / 2)

    extr = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -eye_z],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    proj = torch.tensor([
        [f,   0.0, 0.0, 0.0],
        [0.0, f,   0.0, 0.0],
        [0.0, 0.0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=torch.float32)

    full = proj @ extr
    homo = torch.cat([verts_world, torch.ones(verts_world.shape[0], 1)], dim=-1)
    clip = homo @ full.T
    return clip


def synth_icosphere_mesh(subdivisions: int, batch: int, seed: int):
    """Subdivided icosphere via trimesh, with a real perspective projection
    so vertices_clip has non-trivial w (matches mesh_renderer.py's output).

    subdivisions=6 -> 81,920 triangles
    subdivisions=7 -> 327,680 triangles (matches real TRELLIS scale)
    """
    import trimesh
    g = torch.Generator().manual_seed(seed)

    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.8)
    verts_np = sphere.vertices.astype("float32")
    faces_np = sphere.faces.astype("int32")

    verts = torch.from_numpy(verts_np)

    angle = (torch.rand(1, generator=g).item() - 0.5) * 0.4
    c, s = math.cos(angle), math.sin(angle)
    rot_y = torch.tensor([[c, 0.0, s],
                          [0.0, 1.0, 0.0],
                          [-s, 0.0, c]], dtype=torch.float32)
    angle_x = (torch.rand(1, generator=g).item() - 0.5) * 0.4
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    rot_x = torch.tensor([[1.0, 0.0, 0.0],
                          [0.0, cx, -sx],
                          [0.0, sx, cx]], dtype=torch.float32)
    verts = verts @ rot_y @ rot_x

    pos = _perspective_project(verts)
    pos = pos.unsqueeze(0).expand(batch, -1, -1).contiguous()
    tri = torch.from_numpy(faces_np)

    return pos.to(torch.float32), tri


def synth_soup_mesh(num_tri: int, batch: int, seed: int):
    """Random triangle soup. Each triangle spans a fraction of the viewport
    so coarseRaster has to write tile segments to the linked list, exercising
    the same code paths the real TRELLIS workload hits.

    Three vertices per triangle, no shared vertices -- every triangle is
    independent. Vertex layout: random center in [-0.9, 0.9] x [-0.9, 0.9],
    three offsets of magnitude ~0.05-0.3 around the center. That gives
    triangles spanning roughly 25x25 to 150x150 pixels (3-19 tiles per side).
    """
    g = torch.Generator().manual_seed(seed)

    centers = (torch.rand(num_tri, 2, generator=g) * 1.8) - 0.9
    sizes = (torch.rand(num_tri, 1, generator=g) * 0.25) + 0.05
    angles = torch.rand(num_tri, 3, generator=g) * (2 * math.pi)

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    radii = (torch.rand(num_tri, 3, generator=g) * 0.5 + 0.5) * sizes
    vert_x = centers[:, 0:1] + radii * cos_a
    vert_y = centers[:, 1:2] + radii * sin_a
    vert_z = (torch.rand(num_tri, 3, generator=g) - 0.5) * 0.4

    verts = torch.stack([vert_x, vert_y, vert_z, torch.ones_like(vert_x)], dim=-1)
    pos = verts.reshape(1, num_tri * 3, 4).expand(batch, -1, -1).contiguous()

    base = torch.arange(num_tri).unsqueeze(1) * 3
    tri = (base + torch.tensor([[0, 1, 2]])).to(torch.int32)

    return pos.to(torch.float32), tri


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tools/raster_repro/inputs.pt")
    parser.add_argument("--mode", choices=["grid", "soup", "icosphere"],
                        default="icosphere",
                        help="grid: heightfield (small tris). "
                             "soup: random large tris. "
                             "icosphere: subdivided sphere via trimesh (closest to real TRELLIS).")
    parser.add_argument("--grid-n", type=int, default=400,
                        help="grid mode only; 400 -> 318,402 triangles")
    parser.add_argument("--num-tri", type=int, default=300000,
                        help="soup mode only; default 300K matches real TRELLIS scale")
    parser.add_argument("--subdivisions", type=int, default=7,
                        help="icosphere mode only; 7 -> 327,680 triangles")
    parser.add_argument("--batch", type=int, default=1,
                        help="instance-mode batch size")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="output frame resolution (square)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "grid":
        pos, tri = synth_grid_mesh(args.grid_n, args.batch, args.seed)
    elif args.mode == "soup":
        pos, tri = synth_soup_mesh(args.num_tri, args.batch, args.seed)
    else:
        pos, tri = synth_icosphere_mesh(args.subdivisions, args.batch, args.seed)

    n_tri = tri.shape[0]
    n_vert = pos.shape[1]

    payload = {
        "pos": pos,
        "tri": tri,
        "resolution": (args.resolution, args.resolution),
        "ranges": None,
        "instance_mode": True,
        "pos_shape": tuple(pos.shape),
        "tri_shape": tuple(tri.shape),
        "captured_at": time.time(),
        "synthetic": True,
        "synth_args": {
            "mode": args.mode,
            "grid_n": args.grid_n,
            "num_tri": args.num_tri,
            "subdivisions": args.subdivisions,
            "batch": args.batch,
            "resolution": args.resolution,
            "seed": args.seed,
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(payload, args.output)

    sys.stderr.write(
        f"[synth] wrote {args.output}: pos={tuple(pos.shape)} "
        f"tri={tuple(tri.shape)} ({n_tri} triangles, {n_vert} verts) "
        f"res={args.resolution}x{args.resolution}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
