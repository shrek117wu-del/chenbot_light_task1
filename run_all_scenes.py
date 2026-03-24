#!/usr/bin/env python3
"""
Run all paper scenes: optimize geometry + texture, save results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from core.geometry import SaucerGeometry, optimize_geometry
from core.texture import SaucerTexture, optimize_texture
from core.renderer import MirrorCupSaucerRenderer
from scenes.paper_scenes import ALL_SCENES


def run_scene(scene_fn, output_dir: str = "outputs", res: int = 256):
    scene = scene_fn(res)
    name = scene["name"]
    print(f"\n{'='*60}")
    print(f"  Running scene: {name}")
    print(f"  {scene['description']}")
    print(f"{'='*60}\n")

    scene_dir = os.path.join(output_dir, name)
    os.makedirs(scene_dir, exist_ok=True)

    # Setup
    renderer = MirrorCupSaucerRenderer(img_res=res)
    geometry = SaucerGeometry(
        resolution=res, base_shape=scene["base_shape"]
    )

    # Build reflection map
    renderer.update_reflection_map(geometry.heightfield)

    # --- Step 1: Optimize geometry ---
    def reflection_map_fn(hf):
        from core.reflection import build_reflection_map
        _, _, valid = build_reflection_map(res, saucer_heightfield=hf)
        # Simplified: return normal-based shading
        from core.geometry import SaucerGeometry as SG
        g = SG(res)
        g.displacement = hf
        n = g.compute_normals()
        return n[..., 2] * valid.astype(float)

    info_geom = optimize_geometry(
        geometry,
        scene["direct_target"],
        scene["reflected_target"],
        reflection_map_fn,
        n_iterations=100,
        lr=0.005,
        verbose=True,
    )

    # --- Step 2: Optimize texture ---
    texture = SaucerTexture(resolution=res)
    renderer.update_reflection_map(geometry.heightfield)

    info_tex = optimize_texture(
        texture,
        scene["direct_target"],
        scene["reflected_target"],
        renderer.direct_uv,
        renderer.direct_valid,
        renderer.reflected_uv,
        renderer.reflected_valid,
        geometry.mask,
        n_iterations=200,
        verbose=True,
    )

    # --- Step 3: Render final results ---
    direct_img, reflected_img = renderer.render_both(texture.image)

    # Save images
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0, 0].imshow(scene["direct_target"], cmap="gray")
    axes[0, 0].set_title("Direct Target")
    axes[0, 1].imshow(scene["reflected_target"], cmap="gray")
    axes[0, 1].set_title("Reflected Target")
    axes[0, 2].imshow(geometry.heightfield, cmap="terrain")
    axes[0, 2].set_title("Optimized Height Field")
    axes[1, 0].imshow(np.clip(direct_img, 0, 1))
    axes[1, 0].set_title("Rendered Direct View")
    axes[1, 1].imshow(np.clip(reflected_img, 0, 1))
    axes[1, 1].set_title("Rendered Reflected View")
    axes[1, 2].imshow(texture.image)
    axes[1, 2].set_title("Optimized Saucer Texture")
    for ax in axes.flat:
        ax.axis("off")
    plt.suptitle(f"Scene: {name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(scene_dir, f"{name}_results.png"), dpi=150)
    plt.close()

    # Save texture as image
    tex_img = Image.fromarray((texture.image * 255).astype(np.uint8))
    tex_img.save(os.path.join(scene_dir, f"{name}_texture.png"))

    # Save heightfield
    np.save(os.path.join(scene_dir, f"{name}_heightfield.npy"), geometry.heightfield)

    print(f"  Results saved to {scene_dir}/")
    return geometry, texture


def main():
    os.makedirs("outputs", exist_ok=True)
    for scene_fn in ALL_SCENES:
        run_scene(scene_fn, output_dir="outputs", res=256)
    print("\n✅ All scenes completed!")


if __name__ == "__main__":
    main()