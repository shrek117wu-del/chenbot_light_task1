#!/usr/bin/env python3
"""
pro/main.py – Mirror Cup & Saucer Art  (Photo-Model Edition)
=============================================================

Improved pipeline that loads the actual photo saucer / cup OBJ models
and uses them as the base geometry for optimisation, instead of the
procedural heightfields used in the original main.py.

Usage (from repo root)
----------------------
    python -m pro.main                          # default: wavy saucer, straight cup, exp 1
    python -m pro.main --saucer stepped         # stepped saucer shape
    python -m pro.main --saucer smooth_rim --cup conical
    python -m pro.main --direct data/1-direct.png --reflect data/1-reflect.png
    python -m pro.main --res 100 --render_size 256 --iters1 100  # fast preview

Algorithm stages (identical to the paper):
  Stage 1 – Black-White Enhancement (two-step σ)
  Stage 2 – Sparse Spike Strategy   (proximal gradient)
  Stage 3 – Texturing               (colour optimisation on fixed geometry)
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import cv2

# ── Ensure repo root is on the Python path ────────────────────────────────────
_PRO_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PRO_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.geometry  import precompute_reflection_grid
from core.solver    import MirrorArtSolver
from export_utils   import export_obj_and_texture
from experiments    import get_paper_experiment
from pro.mesh_io    import (load_obj, normalize_saucer,
                             mesh_to_heightfield, get_cup_effective_radius)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_SAUCER_OBJ = {
    'wavy'       : 'photo_saucer_wavy.obj',
    'stepped'    : 'photo_saucer_stepped.obj',
    'smooth_rim' : 'photo_saucer_smooth_rim.obj',
}

_CUP_OBJ = {
    'straight' : 'photo_cup_straight.obj',
    'conical'  : 'photo_cup_conical.obj',
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(Id_path, Ir_path,
                 saucer_type='wavy',
                 cup_type='straight',
                 resolution=(150, 150),
                 img_render_size=(512, 512),
                 iters1_large=300,
                 iters1_small=300,
                 iters2=200,
                 iters3=300,
                 out_dir='pro/out',
                 device=None):
    """
    Full optimisation pipeline.

    1. Load photo saucer OBJ  →  normalise  →  heightfield grid
    2. Load photo cup   OBJ   →  extract effective reflection radius
    3. Compute reflection grid (vectorised, paper geometry)
    4. 3-stage optimisation   (Stage 1 / Stage 2 / Stage 3)
    5. Save rendered views + export textured OBJ

    Returns (grid_x, grid_y, h_final, c_final, solver).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 62)
    print("  Mirror Cup & Saucer Art  –  Photo-Model Edition")
    print("=" * 62)
    print(f"  Saucer  : {saucer_type}  ({_SAUCER_OBJ[saucer_type]})")
    print(f"  Cup     : {cup_type}  ({_CUP_OBJ[cup_type]})")
    print(f"  Direct  : {Id_path}")
    print(f"  Reflect : {Ir_path}")
    print(f"  Grid    : {resolution}   Render : {img_render_size}")
    print(f"  Device  : {device}")
    print("=" * 62)

    t0 = time.time()

    # ── 1. Load & normalise saucer ────────────────────────────────────────────
    print("\n[1] Loading saucer OBJ …")
    s_path = os.path.join(_REPO_ROOT, _SAUCER_OBJ[saucer_type])
    s_verts, s_faces = load_obj(s_path)
    s_verts_n = normalize_saucer(s_verts)
    print(f"    {len(s_verts)} vertices, {len(s_faces)} faces")
    print(f"    original z-range : [{s_verts[:,2].min():.4f}, {s_verts[:,2].max():.4f}]")
    print(f"    normalised z-range: [{s_verts_n[:,2].min():.4f}, {s_verts_n[:,2].max():.4f}]")

    # ── 2. Heightfield grid ───────────────────────────────────────────────────
    print("[2] Converting mesh to heightfield …")
    grid_x, grid_y, base_h = mesh_to_heightfield(s_verts_n, resolution)
    print(f"    grid {grid_x.shape}  h ∈ [{base_h.min():.4f}, {base_h.max():.4f}]"
          f"  ({time.time()-t0:.1f}s)")

    # ── 3. Cup radius & reflection grid ──────────────────────────────────────
    print("[3] Loading cup OBJ and computing reflection grid …")
    c_path = os.path.join(_REPO_ROOT, _CUP_OBJ[cup_type])
    c_verts, _ = load_obj(c_path)
    cup_r = get_cup_effective_radius(c_verts)
    print(f"    cup effective radius = {cup_r:.4f}")

    P_2d = np.array([0.0, -5.5])   # camera top-view XY position (paper §4)
    grid_rx, grid_ry = precompute_reflection_grid(grid_x, grid_y, P_2d, r=cup_r)
    print(f"    reflection grid done  ({time.time()-t0:.1f}s)")

    # ── 4. Solver ─────────────────────────────────────────────────────────────
    print("\n[4] Setting up solver …")
    solver = MirrorArtSolver(
        Id_path, Ir_path, base_h,
        grid_x, grid_y, grid_rx, grid_ry,
        img_render_size=img_render_size,
        device=device
    )

    # Stage 1: Black-White Enhancement
    print("\n[Stage 1] Black-White Enhancement (two-step σ) …")
    h_s1, c_s1 = solver.solve_stage1(
        iters_large_sigma=iters1_large,
        iters_small_sigma=iters1_small,
    )
    _save_stage(solver, h_s1, c_s1, resolution,
                os.path.join(out_dir, 'stage1'))

    # Stage 2: Sparse Spike Strategy
    print("\n[Stage 2] Sparse Spike Strategy …")
    h_s2, c_s2 = solver.solve_stage2(iterations=iters2)
    _save_stage(solver, h_s2, c_s2, resolution,
                os.path.join(out_dir, 'stage2'))

    # Stage 3: Texturing
    print("\n[Stage 3] Texturing …")
    h_final, c_final = solver.solve_stage3_texturing(iterations=iters3)

    elapsed = time.time() - t0
    print(f"\n✓ Optimisation complete in {elapsed / 60:.1f} min")

    # ── 5. Save final rendered views ──────────────────────────────────────────
    print("\n[Output] Saving final views …")
    img_d, img_r = solver.render_views(h_final, c_final)
    cv2.imwrite(os.path.join(out_dir, 'out_direct.png'),
                cv2.cvtColor(img_d, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, 'out_reflect.png'),
                cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))
    print(f"  {out_dir}/out_direct.png")
    print(f"  {out_dir}/out_reflect.png")

    # ── 6. Export textured 3-D model ─────────────────────────────────────────
    print("\n[Export] Saving textured OBJ …")
    export_obj_and_texture(
        grid_x, grid_y, h_final, c_final,
        obj_path=os.path.join(out_dir, 'saucer.obj'),
        tex_path=os.path.join(out_dir, 'texture.png'),
    )

    # ── 7. Copy viewer into output directory ──────────────────────────────────
    _copy_viewer(out_dir)

    print(f"\n→ All outputs in  {os.path.abspath(out_dir)}/")
    print("  Open pro/out/viewer.html in a browser for the 3-D view.")

    return grid_x, grid_y, h_final, c_final, solver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_stage(solver, h, c, resolution, prefix):
    """Render and save intermediate views after each optimisation stage."""
    try:
        H, W = resolution
        h_np = h.cpu().numpy().reshape(H, W)
        c_np = c.cpu().numpy().reshape(H, W, 3)
        img_d, img_r = solver.render_views(h_np, c_np)
        cv2.imwrite(f"{prefix}_direct.png",
                    cv2.cvtColor(img_d, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{prefix}_reflect.png",
                    cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))
        print(f"  saved → {prefix}_direct.png / {prefix}_reflect.png")
    except Exception as exc:
        print(f"  (intermediate save skipped: {exc})")


def _copy_viewer(out_dir):
    """Copy the pro/viewer.html into the output directory so it can be opened directly."""
    src = os.path.join(_PRO_DIR, 'viewer.html')
    dst = os.path.join(out_dir, 'viewer.html')
    if os.path.exists(src) and not os.path.exists(dst):
        import shutil
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description='Mirror Cup & Saucer Art – Photo-Model Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Paper experiment 1, wavy saucer, straight cup  (default)
  python -m pro.main

  # Custom images
  python -m pro.main --direct my_direct.png --reflect my_reflect.png

  # Stepped saucer + conical cup, paper experiment 2
  python -m pro.main --saucer stepped --cup conical --exp 2

  # Fast preview (lower resolution and fewer iterations)
  python -m pro.main --res 80 --render_size 256 --iters1 100 --iters2 50 --iters3 80
""")

    p.add_argument('--direct',      default=None,
                   help='Direct-view target image path (overrides --exp)')
    p.add_argument('--reflect',     default=None,
                   help='Reflection target image path (overrides --exp)')
    p.add_argument('--exp',         type=int, default=1,
                   help='Built-in paper experiment id: 0=synthetic A/B, 1=data/1-*, 2=data/2-*')

    p.add_argument('--saucer',      default='wavy',
                   choices=list(_SAUCER_OBJ.keys()),
                   help='Photo saucer model type (default: wavy)')
    p.add_argument('--cup',         default='straight',
                   choices=list(_CUP_OBJ.keys()),
                   help='Photo cup model type (default: straight)')

    p.add_argument('--res',         type=int, default=150,
                   help='Heightfield grid resolution N×N (paper: 150)')
    p.add_argument('--render_size', type=int, default=512,
                   help='Rendered image resolution (paper: 512)')

    p.add_argument('--iters1',      type=int, default=300,
                   help='Stage 1 iterations per σ step (paper: 500)')
    p.add_argument('--iters2',      type=int, default=200,
                   help='Stage 2 iterations (paper: 300)')
    p.add_argument('--iters3',      type=int, default=300,
                   help='Stage 3 (texturing) iterations (paper: 300)')

    p.add_argument('--out_dir',     default='pro/out',
                   help='Output directory (default: pro/out)')
    p.add_argument('--no_viewer',   action='store_true',
                   help='Skip launching the 3-D browser viewer after export')
    p.add_argument('--device',      default=None,
                   help='torch device: "cuda" or "cpu" (default: auto-detect)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    # Resolve target images
    if args.direct and args.reflect:
        Id_path, Ir_path = args.direct, args.reflect
    else:
        Id_path, Ir_path = get_paper_experiment(args.exp)

    grid_x, grid_y, h_final, c_final, solver = run_pipeline(
        Id_path, Ir_path,
        saucer_type     = args.saucer,
        cup_type        = args.cup,
        resolution      = (args.res, args.res),
        img_render_size = (args.render_size, args.render_size),
        iters1_large    = args.iters1,
        iters1_small    = args.iters1,
        iters2          = args.iters2,
        iters3          = args.iters3,
        out_dir         = args.out_dir,
        device          = args.device,
    )

    if not args.no_viewer:
        import http.server
        import socketserver
        import threading
        import webbrowser

        viewer_path = os.path.join(args.out_dir, 'viewer.html')
        if os.path.exists(viewer_path):
            abs_out = os.path.abspath(args.out_dir)
            PORT = 8080

            def _serve():
                os.chdir(abs_out)
                handler = http.server.SimpleHTTPRequestHandler

                class _ReuseServer(socketserver.TCPServer):
                    allow_reuse_address = True

                with _ReuseServer(('', PORT), handler) as httpd:
                    httpd.serve_forever()

            t = threading.Thread(target=_serve, daemon=True)
            t.start()
            webbrowser.open(f'http://localhost:{PORT}/viewer.html')
            print(f"\nViewer running at http://localhost:{PORT}/viewer.html")
            print("Press Enter to quit.")
            input()
