#!/usr/bin/env python3
"""
Complete application: Mirror Cup and Saucer Art Generator

Input:
  - Direct-view image
  - Reflected-view image
  - Base saucer shape (heightfield image or flat)

Output:
  - Refined saucer geometry (OBJ mesh)
  - Optimized saucer texture (PNG)
  - Visualization of all views

Usage:
  python app.py --direct direct.png --reflected reflected.png \
                --base_shape base.png --output output_dir/
"""

import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from core.geometry import (
    SaucerGeometry,
    optimize_geometry,
    black_white_enhancement,
    sparse_spike_movement,
)
from core.texture import SaucerTexture, optimize_texture
from core.renderer import MirrorCupSaucerRenderer
from core.sdf_utils import export_obj


def load_image(path: str, res: int) -> np.ndarray:
    """Load and resize an image to (res, res) grayscale or RGB."""
    img = Image.open(path).convert("RGB").resize((res, res), Image.LANCZOS)
    return np.array(img, dtype=np.float64) / 255.0


def load_heightfield(path: str, res: int) -> np.ndarray:
    """Load a grayscale image as a heightfield."""
    img = Image.open(path).convert("L").resize((res, res), Image.LANCZOS)
    hf = np.array(img, dtype=np.float64) / 255.0
    hf = (hf - 0.5) * 0.5  # center around 0, range [-0.25, 0.25]
    return hf


def run_pipeline(
    direct_path: str,
    reflected_path: str,
    base_shape_path: str = None,
    output_dir: str = "output",
    resolution: int = 256,
    geom_iterations: int = 150,
    tex_iterations: int = 300,
    cup_radius: float = 1.0,
    cup_height: float = 2.0,
    saucer_radius: float = 3.0,
):
    """Full optimization pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    res = resolution

    print("=" * 60)
    print("  Computational Mirror Cup and Saucer Art")
    print("  Wu et al., ACM TOG 2022 (doi:10.1145/3517120)")
    print("=" * 60)

    # 1. Load inputs
    print("\n[1/6] Loading input images...")
    direct_target = load_image(direct_path, res)
    reflected_target = load_image(reflected_path, res)

    if base_shape_path and os.path.exists(base_shape_path):
        base_shape = load_heightfield(base_shape_path, res)
        print(f"  Base shape loaded from {base_shape_path}")
    else:
        base_shape = np.zeros((res, res))
        print("  Using flat base shape")

    # 2. Initialize geometry & renderer
    print("\n[2/6] Initializing geometry and renderer...")
    geometry = SaucerGeometry(
        resolution=res,
        saucer_radius=saucer_radius,
        cup_radius=cup_radius,
        base_shape=base_shape,
    )
    renderer = MirrorCupSaucerRenderer(
        img_res=res,
        cup_radius=cup_radius,
        cup_height=cup_height,
        saucer_radius=saucer_radius,
    )
    renderer.update_reflection_map(geometry.heightfield)

    # 3. Optimize geometry
    print(f"\n[3/6] Optimizing geometry ({geom_iterations} iterations)...")

    def reflection_map_fn(hf):
        from core.reflection import build_reflection_map
        _, _, valid = build_reflection_map(
            res, cup_radius, cup_height, saucer_radius,
            saucer_heightfield=hf
        )
        from core.geometry import SaucerGeometry as SG
        g = SG(res, saucer_radius, cup_radius)
        g.base_shape = base_shape
        g.displacement = hf - base_shape
        n = g.compute_normals()
        return n[..., 2] * valid.astype(float)

    direct_gray = np.mean(direct_target, axis=-1)
    reflected_gray = np.mean(reflected_target, axis=-1)

    info_geom = optimize_geometry(
        geometry,
        direct_gray,
        reflected_gray,
        reflection_map_fn,
        n_iterations=geom_iterations,
        lr=0.005,
    )

    # 4. Apply enhancement techniques from the paper
    print("\n[4/6] Applying black-white enhancement and sparse spike refinement...")
    geometry.displacement = black_white_enhancement(geometry.displacement)
    geometry.displacement = sparse_spike_movement(
        geometry.displacement, geometry.mask, n_spikes=30, spike_amplitude=0.05
    )

    # 5. Optimize texture
    print(f"\n[5/6] Optimizing texture ({tex_iterations} iterations)...")
    renderer.update_reflection_map(geometry.heightfield)
    texture = SaucerTexture(resolution=res)

    info_tex = optimize_texture(
        texture,
        direct_target,
        reflected_target,
        renderer.direct_uv,
        renderer.direct_valid,
        renderer.reflected_uv,
        renderer.reflected_valid,
        geometry.mask,
        n_iterations=tex_iterations,
    )

    # 6. Export results
    print("\n[6/6] Exporting results...")

    # Render final views
    direct_render, reflected_render = renderer.render_both(texture.image)

    # Save texture
    tex_path = os.path.join(output_dir, "saucer_texture.png")
    Image.fromarray((texture.image * 255).astype(np.uint8)).save(tex_path)

    # Save heightfield
    np.save(os.path.join(output_dir, "heightfield.npy"), geometry.heightfield)

    # Save mesh (OBJ)
    vertices = geometry.to_mesh_vertices()
    faces = geometry.to_mesh_faces()
    uvs = geometry.to_mesh_uvs()
    export_obj(
        vertices, faces, uvs,
        "saucer_texture.png",
        os.path.join(output_dir, "saucer.obj"),
        os.path.join(output_dir, "saucer.mtl"),
    )

    # Save visualization
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes[0, 0].imshow(direct_target)
    axes[0, 0].set_title("Direct Target (Input)")
    axes[0, 1].imshow(reflected_target)
    axes[0, 1].set_title("Reflected Target (Input)")
    axes[0, 2].imshow(geometry.heightfield, cmap="terrain")
    axes[0, 2].set_title("Refined Geometry (Height Field)")
    axes[0, 3].imshow(geometry.compute_normals()[..., 2], cmap="gray")
    axes[0, 3].set_title("Surface Normal Map")
    axes[1, 0].imshow(np.clip(direct_render, 0, 1))
    axes[1, 0].set_title("Rendered Direct View")
    axes[1, 1].imshow(np.clip(reflected_render, 0, 1))
    axes[1, 1].set_title("Rendered Reflected View")
    axes[1, 2].imshow(texture.image)
    axes[1, 2].set_title("Optimized Saucer Texture")

    # Loss curves
    ax_loss = axes[1, 3]
    ax_loss.plot(info_geom["losses"], label="Geometry Loss")
    ax_loss.plot(info_tex["losses"], label="Texture Loss")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Optimization Convergence")
    ax_loss.legend()
    ax_loss.grid(True)

    for ax in axes.flat:
        if ax != ax_loss:
            ax.axis("off")

    plt.suptitle(
        "Computational Mirror Cup and Saucer Art — Results", fontsize=18
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results_overview.png"), dpi=150)
    plt.close()

    print(f"\n✅ Pipeline complete! Results in: {output_dir}/")
    print(f"   - saucer.obj        (3D mesh)")
    print(f"   - saucer.mtl        (material)")
    print(f"   - saucer_texture.png (texture)")
    print(f"   - heightfield.npy   (geometry data)")
    print(f"   - results_overview.png (visualization)")


def main():
    parser = argparse.ArgumentParser(
        description="Computational Mirror Cup and Saucer Art Generator"
    )
    parser.add_argument(
        "--direct", type=str, required=True, help="Path to direct-view target image"
    )
    parser.add_argument(
        "--reflected", type=str, required=True,
        help="Path to reflected-view target image",
    )
    parser.add_argument(
        "--base_shape", type=str, default=None,
        help="Path to base shape heightfield (grayscale PNG). Default: flat.",
    )
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--resolution", type=int, default=256, help="Grid resolution")
    parser.add_argument(
        "--geom_iter", type=int, default=150, help="Geometry optimization iterations"
    )
    parser.add_argument(
        "--tex_iter", type=int, default=300, help="Texture optimization iterations"
    )
    parser.add_argument("--cup_radius", type=float, default=1.0)
    parser.add_argument("--cup_height", type=float, default=2.0)
    parser.add_argument("--saucer_radius", type=float, default=3.0)

    args = parser.parse_args()

    run_pipeline(
        direct_path=args.direct,
        reflected_path=args.reflected,
        base_shape_path=args.base_shape,
        output_dir=args.output,
        resolution=args.resolution,
        geom_iterations=args.geom_iter,
        tex_iterations=args.tex_iter,
        cup_radius=args.cup_radius,
        cup_height=args.cup_height,
        saucer_radius=args.saucer_radius,
    )


if __name__ == "__main__":
    main()