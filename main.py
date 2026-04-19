"""
main.py – Computational Mirror Cup and Saucer Art
================================================
Usage:
    python main.py [--exp EXP_ID] [--shape SHAPE] [--res N] [--iters1 N] [--iters2 N] [--iters3 N]

EXP_ID:
    0 – synthetic A/B letters (quick test)
    1 – data/1-direct.png + data/1-reflect.png  (paper experiment 1)
    2 – data/2-direct.png + data/2-reflect.png  (paper experiment 2)

SHAPE: plane | saucer | random | tabula_scalata | shallow_bowl
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import cv2

from core.geometry  import precompute_reflection_grid
from core.solver    import MirrorArtSolver
from experiments    import create_base_shape, create_reflection_grid, get_paper_experiment
from export_utils   import export_obj_and_texture
from viewer         import visualize_result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(Id_path, Ir_path,
                 shape_type="saucer",
                 cup_type="cylinder",
                 resolution=(150, 150),
                 img_render_size=(512, 512),
                 iters1_large=300,
                 iters1_small=300,
                 iters2=200,
                 iters3=300,
                 device=None):
    """Full three-stage mirror art optimisation pipeline."""

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("  Computational Mirror Cup and Saucer Art")
    print("=" * 60)
    print(f"  Direct  : {Id_path}")
    print(f"  Reflect : {Ir_path}")
    print(f"  Shape   : {shape_type}  @ {resolution}")
    print(f"  Cup     : {cup_type}")
    print(f"  Render  : {img_render_size}  device={device}")
    print("=" * 60)

    t0 = time.time()

    # 1. Base shape + reflected grid ----------------------------------------
    print("\n[1] Building base heightfield…")
    grid_x, grid_y, base_h = create_base_shape(shape_type, resolutions=resolution)

    print("[2] Precomputing reflection grid (may take a while)…")
    P_2d = np.array([0.0, -5.5])   # camera top-view position
    grid_rx, grid_ry = create_reflection_grid(
        cup_type, grid_x, grid_y, P_2d, r=0.4)
    print(f"    done  ({time.time()-t0:.1f}s)")

    # 2. Solver setup -------------------------------------------------------
    print("[3] Setting up solver…")
    solver = MirrorArtSolver(
        Id_path, Ir_path, base_h,
        grid_x, grid_y, grid_rx, grid_ry,
        img_render_size=img_render_size,
        device=device
    )

    # 3. Stage 1: Black-White Enhancement -----------------------------------
    print("\n[Stage 1] Black-White Enhancement (two-step σ)…")
    h_s1, c_s1 = solver.solve_stage1(
        iters_large_sigma=iters1_large,
        iters_small_sigma=iters1_small
    )

    # Save intermediate rendered views
    _save_intermediate(solver, h_s1.cpu().numpy().reshape(*resolution),
                       c_s1.cpu().numpy().reshape(*resolution, 3),
                       "out_stage1")

    # 4. Stage 2: Sparse Spike Strategy -------------------------------------
    print("\n[Stage 2] Sparse Spike Strategy…")
    h_s2, c_s2 = solver.solve_stage2(iterations=iters2)
    _save_intermediate(solver, h_s2.cpu().numpy().reshape(*resolution),
                       c_s2.cpu().numpy().reshape(*resolution, 3),
                       "out_stage2")

    # 5. Stage 3: Texturing -------------------------------------------------
    print("\n[Stage 3] Texturing…")
    h_final, c_final = solver.solve_stage3_texturing(iterations=iters3)

    elapsed = time.time() - t0
    print(f"\n[DONE] Optimisation complete in {elapsed/60:.1f} min")

    # 6. Save final renders -------------------------------------------------
    print("[Output] Saving final rendered views…")
    img_d, img_r = solver.render_views(h_final, c_final)
    cv2.imwrite("out_direct.png",  cv2.cvtColor(img_d, cv2.COLOR_RGB2BGR))
    cv2.imwrite("out_reflect.png", cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))
    print("  out_direct.png  /  out_reflect.png")

    return grid_x, grid_y, h_final, c_final


def _save_intermediate(solver, h_np, c_np, prefix):
    try:
        img_d, img_r = solver.render_views(h_np, c_np)
        cv2.imwrite(f"{prefix}_direct.png",  cv2.cvtColor(img_d, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{prefix}_reflect.png", cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))
        print(f"    intermediate saved → {prefix}_direct.png / {prefix}_reflect.png")
    except Exception as e:
        print(f"    (intermediate save skipped: {e})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Mirror Cup and Saucer Art")
    p.add_argument("--exp",    type=int,   default=1,
                   help="Experiment id: 0=synthetic, 1=paper-1, 2=paper-2")
    p.add_argument("--shape",  type=str,   default="saucer",
                   help="Saucer shape: plane|saucer|random|tabula_scalata|"
                        "shallow_bowl|cone|saddle")
    p.add_argument("--cup",    type=str,   default="cylinder",
                   help="Mirror cup type: cylinder|ellipse|ngon4|ngon6|ngon8|ngonN "
                        "(Section 3.5 extension)")
    p.add_argument("--res",    type=int,   default=150,
                   help="Heightfield resolution (NxN, paper uses 150)")
    p.add_argument("--render_size", type=int, default=512,
                   help="Render image size (paper uses 512)")
    p.add_argument("--iters1", type=int,   default=500,
                   help="Stage 1 iters per sigma step (paper: until convergence)")
    p.add_argument("--iters2", type=int,   default=300,
                   help="Stage 2 iters (paper: until convergence)")
    p.add_argument("--iters3", type=int,   default=300,
                   help="Stage 3 texturing iters")
    p.add_argument("--no_viewer", action="store_true",
                   help="Skip launching the 3D viewer")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    Id_path, Ir_path = get_paper_experiment(args.exp)

    X, Y, H, C = run_pipeline(
        Id_path, Ir_path,
        shape_type      = args.shape,
        cup_type        = args.cup,
        resolution      = (args.res, args.res),
        img_render_size = (args.render_size, args.render_size),
        iters1_large    = args.iters1,
        iters1_small    = args.iters1,
        iters2          = args.iters2,
        iters3          = args.iters3,
    )

    print("\n[Output] Exporting 3D model…")
    export_obj_and_texture(X, Y, H, C, cup_type=args.cup)

    if not args.no_viewer:
        print("\n[Viewer] Launching 3D viewer…")
        visualize_result(X, Y, H, C)
