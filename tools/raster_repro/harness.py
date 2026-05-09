"""
Replay harness for the nvdiffrast HIP rasterizer.

Loads the captured input dump and runs nvdiffrast.torch.rasterize() in a tight
loop. Emits exactly one JSON line on stdout summarizing the run.

Output schema:
    {"status": "PASS"|"CRASH", "frames_completed": int, "elapsed_s": float,
     "error": str|null, "first_frame_s": float|null}

Process exit code:
    0 on PASS, 1 on CRASH (caught), other on internal harness error.
"""

import argparse
import json
import os
import signal
import sys
import time
import traceback

import torch


def _frame_timeout_handler(*args):
    del args  # signal handler signature requires (signum, frame)
    raise TimeoutError("frame timeout")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tools/raster_repro/inputs.pt",
                        help="path to the captured input dump")
    parser.add_argument("--frames", type=int, default=120,
                        help="number of replay iterations (matches mesh preview length)")
    parser.add_argument("--frame-timeout", type=int, default=30,
                        help="seconds before treating a stuck frame as a CRASH")
    parser.add_argument("--peel", action="store_true",
                        help="if set, replay with peeling_idx=0 (GLB-style)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        json.dump({"status": "CRASH", "frames_completed": 0, "elapsed_s": 0.0,
                   "error": f"input dump missing: {args.input}",
                   "first_frame_s": None}, sys.stdout)
        sys.stdout.write("\n")
        return 1

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    pos_cpu = payload["pos"]
    tri_cpu = payload["tri"]
    resolution = payload["resolution"]
    ranges_cpu = payload.get("ranges")

    import nvdiffrast.torch as dr

    device = torch.device("cuda")
    pos = pos_cpu.to(device).contiguous()
    tri = tri_cpu.to(device).contiguous()
    ranges = ranges_cpu.contiguous() if ranges_cpu is not None else None
    glctx = dr.RasterizeCudaContext()

    signal.signal(signal.SIGALRM, _frame_timeout_handler)

    started = time.time()
    first_frame_s = None
    frames_completed = 0
    err = None

    try:
        for i in range(args.frames):
            signal.alarm(args.frame_timeout)
            frame_t0 = time.time()

            if args.peel:
                with dr.DepthPeeler(glctx, pos, tri, resolution, ranges=ranges) as peeler:
                    peeler.rasterize_next_layer()
            else:
                dr.rasterize(glctx, pos, tri, resolution, ranges=ranges)

            torch.cuda.synchronize()
            signal.alarm(0)

            if i == 0:
                first_frame_s = time.time() - frame_t0
            frames_completed += 1
    except TimeoutError as e:
        err = f"frame_timeout after {frames_completed} frames"
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        signal.alarm(0)

    status = "PASS" if err is None and frames_completed == args.frames else "CRASH"
    out = {
        "status": status,
        "frames_completed": frames_completed,
        "elapsed_s": round(time.time() - started, 3),
        "error": err,
        "first_frame_s": round(first_frame_s, 3) if first_frame_s is not None else None,
    }
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
