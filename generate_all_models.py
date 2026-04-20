"""
generate_all_models.py — Generate all saucer and cup OBJ models for research use.

Generates:
  models/saucers/  — 6 saucer OBJ files (tabula_scalata, dish, shallow_bowl,
                       wave, luycho_concentric, luycho_radial)
  models/cups/     — 3 cup OBJ files (cylinder, luycho_tapered, luycho_straight)

Usage:
    python generate_all_models.py [--output models] [--res 150]
"""
import argparse
import os

from generate_saucer_models import generate_saucer_models, SAUCER_SHAPES
from export_utils import (
    export_cylinder_obj,
    export_tapered_cup_obj,
)


def generate_cup_models(output_dir="models/cups"):
    """Generate all 3 cup OBJ models and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    # 1. Standard cylinder (paper default: radius=0.4, height=2.0)
    cylinder_path = os.path.join(output_dir, "cylinder.obj")
    print("\nGenerating cup: cylinder")
    export_cylinder_obj(radius=0.4, height=2.0, segments=128, obj_path=cylinder_path)
    generated.append(cylinder_path)

    # 2. Luycho tapered cup (frustum): bottom r=0.25, top r=0.4, height=2.0
    tapered_path = os.path.join(output_dir, "luycho_tapered.obj")
    print("\nGenerating cup: luycho_tapered")
    export_tapered_cup_obj(r_bottom=0.25, r_top=0.4, height=2.0, segments=128,
                           obj_path=tapered_path)
    generated.append(tapered_path)

    # 3. Luycho straight cup: radius=0.38, height=1.8
    straight_path = os.path.join(output_dir, "luycho_straight.obj")
    print("\nGenerating cup: luycho_straight")
    export_cylinder_obj(radius=0.38, height=1.8, segments=128, obj_path=straight_path)
    generated.append(straight_path)

    return generated


def main():
    p = argparse.ArgumentParser(
        description="Generate all saucer and cup OBJ models")
    p.add_argument("--output", type=str, default="models",
                   help="Root output directory (saucers go to <output>/saucers/, "
                        "cups to <output>/cups/)")
    p.add_argument("--res", type=int, default=150,
                   help="Saucer mesh resolution (NxN, paper default=150)")
    args = p.parse_args()

    saucer_dir = os.path.join(args.output, "saucers")
    cup_dir    = os.path.join(args.output, "cups")

    print("=" * 60)
    print("  Generating All Mirror Art Models")
    print("=" * 60)
    print(f"  Saucer dir : {saucer_dir}")
    print(f"  Cup dir    : {cup_dir}")
    print(f"  Resolution : {args.res}×{args.res} (saucers)")
    print("=" * 60)

    # Generate saucers
    print("\n--- SAUCERS ---")
    saucer_files = generate_saucer_models(saucer_dir, args.res)

    # Generate cups
    print("\n--- CUPS ---")
    cup_files = generate_cup_models(cup_dir)

    all_files = saucer_files + cup_files

    print(f"\n{'='*60}")
    print(f"  Summary: {len(all_files)} model files generated")
    print(f"\n  Saucers ({len(saucer_files)}):")
    for f in saucer_files:
        size_kb = os.path.getsize(f) / 1024
        print(f"    {f}  ({size_kb:.0f} KB)")
    print(f"\n  Cups ({len(cup_files)}):")
    for f in cup_files:
        size_kb = os.path.getsize(f) / 1024
        print(f"    {f}  ({size_kb:.0f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
