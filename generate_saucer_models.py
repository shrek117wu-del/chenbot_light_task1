"""
generate_saucer_models.py — Generate standalone saucer OBJ models for research use.

Generates 6 saucer OBJ files in models/saucers/:
  1. tabula_scalata  — corrugated/wave pattern (artist's original)
  2. dish            — flat centre with raised edges (paper Figure 1)
  3. shallow_bowl    — gentle parabolic bowl
  4. wave            — sinusoidal waves in X and Y (paper Figure 2)
  5. luycho_concentric — concentric ring/groove pattern (Luycho blue saucer)
  6. luycho_radial     — radial stripe/corrugation pattern (Luycho white saucer)

Usage:
    python generate_saucer_models.py [--output models/saucers] [--res 150]
"""
import argparse
import os
import numpy as np
import cv2

from experiments import create_base_shape


SAUCER_SHAPES = [
    "tabula_scalata",
    "dish",
    "shallow_bowl",
    "wave",
    "luycho_concentric",
    "luycho_radial",
]


def export_saucer_geometry_obj(grid_x, grid_y, h_grid, obj_path):
    """
    Export a plain-white saucer OBJ + MTL (no texture PNG needed).
    Uses the same Three.js Y-up coordinate convention as export_saucer_obj().

    Coordinate mapping: Python (x, y, z_height) → Three.js (x, z_height, -y)
    """
    H, W = grid_x.shape
    os.makedirs(os.path.dirname(os.path.abspath(obj_path)), exist_ok=True)

    mtl_name = os.path.splitext(os.path.basename(obj_path))[0] + ".mtl"
    mtl_path = os.path.join(os.path.dirname(os.path.abspath(obj_path)), mtl_name)

    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("o saucer\n")
        f.write("usemtl saucer_white\n\n")

        # Vertices: Three.js Y-up → v  x  height  -y
        for i in range(H):
            for j in range(W):
                vx =  float(grid_x[i, j])
                vy =  float(h_grid[i, j])   # height → Three.js Y
                vz = -float(grid_y[i, j])   # -py    → Three.js Z
                f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")

        # UV coordinates
        for i in range(H):
            for j in range(W):
                u = j / (W - 1)
                v = 1.0 - i / (H - 1)
                f.write(f"vt {u:.6f} {v:.6f}\n")

        # Faces: each quad → 2 CCW triangles (1-indexed)
        for i in range(H - 1):
            for j in range(W - 1):
                a = i * W + j + 1
                b = i * W + j + 2
                c = (i + 1) * W + j + 1
                d = (i + 1) * W + j + 2
                f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
                f.write(f"f {b}/{b} {d}/{d} {c}/{c}\n")

    # Plain white MTL — no texture, just solid diffuse white
    with open(mtl_path, "w") as f:
        f.write("newmtl saucer_white\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("Ns 0.0\n")
        f.write("illum 1\n")

    n_tris = (H - 1) * (W - 1) * 2
    print(f"  OBJ saved → {obj_path}  ({n_tris} triangles)")
    print(f"  MTL saved → {mtl_path}")
    return obj_path, mtl_path


def generate_saucer_models(output_dir="models/saucers", resolution=150):
    """Generate all 6 saucer OBJ models and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    res = (resolution, resolution)
    generated = []

    for shape in SAUCER_SHAPES:
        print(f"\nGenerating saucer: {shape}")
        grid_x, grid_y, h_grid = create_base_shape(shape, resolutions=res)
        obj_path = os.path.join(output_dir, f"{shape}.obj")
        export_saucer_geometry_obj(grid_x, grid_y, h_grid, obj_path)
        generated.append(obj_path)

    return generated


def main():
    p = argparse.ArgumentParser(
        description="Generate standalone saucer OBJ models for all 6 saucer types")
    p.add_argument("--output", type=str, default="models/saucers",
                   help="Output directory for saucer OBJ files")
    p.add_argument("--res", type=int, default=150,
                   help="Mesh resolution (NxN, paper default=150)")
    args = p.parse_args()

    print("=" * 60)
    print("  Generating Saucer OBJ Models")
    print("=" * 60)
    print(f"  Output dir : {args.output}")
    print(f"  Resolution : {args.res}×{args.res}")
    print(f"  Shapes     : {', '.join(SAUCER_SHAPES)}")
    print("=" * 60)

    generated = generate_saucer_models(args.output, args.res)

    print(f"\n{'='*60}")
    print(f"  Generated {len(generated)} saucer models:")
    for f in generated:
        size_kb = os.path.getsize(f) / 1024
        print(f"    {f}  ({size_kb:.0f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
