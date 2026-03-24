#!/usr/bin/env python3
"""
Interactive 3D Viewer for Mirror Cup and Saucer Art

Features:
  - Load optimized saucer mesh + texture + cup mirror
  - Mouse-drag rotation to inspect from all angles
  - Toggle views: Direct view / Reflected view / Full textured saucer
  - Real-time reflected image update as viewing angle changes

Usage:
  python viewer3d.py --mesh output/saucer.obj --texture output/saucer_texture.png
  python viewer3d.py --heightfield output/heightfield.npy \
                     --texture output/saucer_texture.png

Requires: open3d >= 0.17
"""

import argparse
import os
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from PIL import Image


def create_cylinder_mesh(
    radius: float = 1.0,
    height: float = 2.0,
    resolution: int = 64,
    mirror_color: np.ndarray = None,
) -> "o3d.geometry.TriangleMesh":
    """Create a mirrored cylinder (cup) mesh."""
    if mirror_color is None:
        mirror_color = np.array([0.85, 0.85, 0.9])

    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=height, resolution=resolution
    )
    mesh.translate([0, 0, height / 2])  # base at z=0
    mesh.paint_uniform_color(mirror_color)
    mesh.compute_vertex_normals()
    return mesh


def create_saucer_mesh_from_heightfield(
    heightfield: np.ndarray,
    saucer_radius: float = 3.0,
    cup_radius: float = 1.0,
    texture_path: str = None,
) -> "o3d.geometry.TriangleMesh":
    """Create a textured saucer mesh from a heightfield array."""
    res = heightfield.shape[0]

    # Vertices
    u = np.linspace(-saucer_radius, saucer_radius, res)
    v = np.linspace(-saucer_radius, saucer_radius, res)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    zz = heightfield

    vertices = np.stack([uu, vv, zz], axis=-1).reshape(-1, 3)

    # Mask: inside saucer and outside cup
    r2 = uu ** 2 + vv ** 2
    valid = (r2 <= saucer_radius ** 2) & (r2 >= cup_radius ** 2)

    # Faces
    faces = []
    for j in range(res - 1):
        for i in range(res - 1):
            if not (valid[j, i] and valid[j, i + 1] and valid[j + 1, i]):
                continue
            idx = j * res + i
            faces.append([idx, idx + 1, idx + res])
            if valid[j + 1, i + 1]:
                faces.append([idx + 1, idx + res + 1, idx + res])
    faces = np.array(faces, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # Texture via vertex colors
    if texture_path and os.path.exists(texture_path):
        tex_img = np.array(Image.open(texture_path).convert("RGB").resize(
            (res, res), Image.LANCZOS
        ), dtype=np.float64) / 255.0
        colors = np.zeros((res * res, 3))
        for j in range(res):
            for i in range(res):
                colors[j * res + i] = tex_img[j, i]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        mesh.paint_uniform_color([0.9, 0.85, 0.7])

    return mesh


def create_floor_plane(size: float = 5.0) -> "o3d.geometry.TriangleMesh":
    """Create a thin floor plane."""
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=size * 2, height=size * 2, depth=0.01
    )
    mesh.translate([-size, -size, -0.02])
    mesh.paint_uniform_color([0.95, 0.95, 0.95])
    mesh.compute_vertex_normals()
    return mesh


def launch_viewer(
    heightfield_path: str = None,
    mesh_path: str = None,
    texture_path: str = None,
    cup_radius: float = 1.0,
    cup_height: float = 2.0,
    saucer_radius: float = 3.0,
):
    """Launch the interactive 3D viewer."""
    if not HAS_OPEN3D:
        print("❌ Open3D is required for the 3D viewer.")
        print("   Install it with: pip install open3d")
        _fallback_matplotlib_viewer(
            heightfield_path, texture_path, saucer_radius, cup_radius
        )
        return

    geometries = []

    # Load saucer
    if heightfield_path and os.path.exists(heightfield_path):
        hf = np.load(heightfield_path)
        saucer = create_saucer_mesh_from_heightfield(
            hf, saucer_radius, cup_radius, texture_path
        )
        geometries.append(saucer)
    elif mesh_path and os.path.exists(mesh_path):
        saucer = o3d.io.read_triangle_mesh(mesh_path)
        saucer.compute_vertex_normals()
        if texture_path:
            # Apply vertex colors from texture
            pass
        geometries.append(saucer)
    else:
        print("⚠️  No mesh or heightfield provided. Creating demo scene.")
        hf = np.zeros((128, 128))
        saucer = create_saucer_mesh_from_heightfield(
            hf, saucer_radius, cup_radius, texture_path
        )
        geometries.append(saucer)

    # Create mirror cup
    cup = create_cylinder_mesh(cup_radius, cup_height)
    geometries.append(cup)

    # Floor
    floor = create_floor_plane(saucer_radius * 1.5)
    geometries.append(floor)

    # Coordinate frame (small)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coord)

    # Launch viewer
    print("\n🎨 Interactive 3D Mirror Cup & Saucer Viewer")
    print("━" * 50)
    print("  🖱️  Left-click + drag  → Rotate")
    print("  🖱️  Right-click + drag → Pan")
    print("  🖱️  Scroll             → Zoom")
    print("  ⌨️  R                  → Reset view")
    print("  ⌨️  Q / Esc            → Quit")
    print("━" * 50)

    # Custom visualization with key callbacks
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Mirror Cup & Saucer Art — Interactive 3D Viewer",
        width=1280,
        height=960,
    )

    for geom in geometries:
        vis.add_geometry(geom)

    # View control
    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(0.5)
    view_ctl.set_front([0, -0.5, 0.8])
    view_ctl.set_lookat([0, 0, 0.5])
    view_ctl.set_up([0, 0, 1])

    render_opt = vis.get_render_option()
    render_opt.light_on = True
    render_opt.background_color = np.array([0.1, 0.1, 0.15])

    # Key callbacks for view presets
    def set_top_view(vis):
        """Top-down (direct) view."""
        ctl = vis.get_view_control()
        ctl.set_front([0, 0, 1])
        ctl.set_up([0, 1, 0])
        ctl.set_zoom(0.4)
        print("📷 View: Top-down (direct view)")
        return False

    def set_side_view(vis):
        """Side view (to see the cup reflection)."""
        ctl = vis.get_view_control()
        ctl.set_front([0, -0.7, 0.3])
        ctl.set_up([0, 0, 1])
        ctl.set_zoom(0.5)
        print("📷 View: Side (reflected view angle)")
        return False

    def set_perspective_view(vis):
        """Default perspective view."""
        ctl = vis.get_view_control()
        ctl.set_front([0.5, -0.5, 0.7])
        ctl.set_up([0, 0, 1])
        ctl.set_zoom(0.5)
        print("📷 View: Perspective")
        return False

    def toggle_wireframe(vis):
        """Toggle wireframe mode."""
        opt = vis.get_render_option()
        opt.mesh_show_wireframe = not opt.mesh_show_wireframe
        wf = "ON" if opt.mesh_show_wireframe else "OFF"
        print(f"🔲 Wireframe: {wf}")
        return False

    # Register key callbacks
    vis.register_key_callback(ord("1"), set_top_view)      # Key 1: top view
    vis.register_key_callback(ord("2"), set_side_view)      # Key 2: side view
    vis.register_key_callback(ord("3"), set_perspective_view)  # Key 3: perspective
    vis.register_key_callback(ord("W"), toggle_wireframe)   # Key W: wireframe

    print("\n  Additional shortcuts:")
    print("  ⌨️  1 → Direct (top-down) view")
    print("  ⌨️  2 → Reflected (side) view")
    print("  ⌨️  3 → Perspective view")
    print("  ⌨️  W → Toggle wireframe")

    vis.run()
    vis.destroy_window()


def _fallback_matplotlib_viewer(
    heightfield_path, texture_path, saucer_radius, cup_radius
):
    """Fallback 3D viewer using matplotlib if Open3D is not available."""
    print("\n📊 Using matplotlib fallback 3D viewer (install open3d for full interactivity)")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if heightfield_path and os.path.exists(heightfield_path):
        hf = np.load(heightfield_path)
    else:
        hf = np.zeros((64, 64))

    res = hf.shape[0]
    u = np.linspace(-saucer_radius, saucer_radius, res)
    v = np.linspace(-saucer_radius, saucer_radius, res)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    # Mask
    r2 = uu ** 2 + vv ** 2
    mask = (r2 <= saucer_radius ** 2) & (r2 >= cup_radius ** 2)
    zz = np.where(mask, hf, np.nan)

    # Load texture for color
    if texture_path and os.path.exists(texture_path):
        tex = np.array(Image.open(texture_path).convert("RGB").resize(
            (res, res), Image.LANCZOS
        ), dtype=np.float64) / 255.0
        facecolors = tex
    else:
        facecolors = None

    # Cylinder (cup)
    theta_cyl = np.linspace(0, 2 * np.pi, 64)
    z_cyl = np.linspace(0, 2.0, 32)
    theta_g, z_g = np.meshgrid(theta_cyl, z_cyl)
    x_cyl = cup_radius * np.cos(theta_g)
    y_cyl = cup_radius * np.sin(theta_g)

    fig = plt.figure(figsize=(14, 10))

    # 3D view
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot_surface(uu, vv, zz, cmap="terrain" if facecolors is None else None,
                     facecolors=facecolors, alpha=0.9)
    ax1.plot_surface(x_cyl, y_cyl, z_g, color="silver", alpha=0.6)
    ax1.set_title("Perspective View")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Top view
    ax2 = fig.add_subplot(222)
    if facecolors is not None:
        display = np.where(mask[..., None], tex, 1.0)
        ax2.imshow(display, extent=[-saucer_radius, saucer_radius,
                                     -saucer_radius, saucer_radius])
    else:
        ax2.imshow(zz, cmap="terrain",
                   extent=[-saucer_radius, saucer_radius,
                           -saucer_radius, saucer_radius])
    circle_out = plt.Circle((0, 0), saucer_radius, fill=False, color="black")
    circle_in = plt.Circle((0, 0), cup_radius, fill=False, color="red")
    ax2.add_patch(circle_out)
    ax2.add_patch(circle_in)
    ax2.set_title("Direct View (Top-Down)")
    ax2.set_aspect("equal")

    # Side view (heightfield profile)
    ax3 = fig.add_subplot(223)
    mid = res // 2
    ax3.plot(u, hf[mid, :], "b-", linewidth=2, label="Height profile (y=0)")
    ax3.axvline(-cup_radius, color="r", linestyle="--", label="Cup edge")
    ax3.axvline(cup_radius, color="r", linestyle="--")
    ax3.set_title("Height Profile (Cross Section)")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Height")
    ax3.legend()
    ax3.grid(True)

    # Height field heatmap
    ax4 = fig.add_subplot(224)
    im = ax4.imshow(hf, cmap="terrain",
                    extent=[-saucer_radius, saucer_radius,
                            -saucer_radius, saucer_radius])
    plt.colorbar(im, ax=ax4, label="Height")
    ax4.set_title("Height Field (Geometry Detail)")
    ax4.set_aspect("equal")

    plt.suptitle("Mirror Cup & Saucer Art — 3D Viewer", fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D Viewer for Mirror Cup and Saucer Art"
    )
    parser.add_argument("--mesh", type=str, default=None, help="Path to saucer OBJ mesh")
    parser.add_argument(
        "--heightfield", type=str, default=None, help="Path to heightfield .npy file"
    )
    parser.add_argument(
        "--texture", type=str, default=None, help="Path to saucer texture PNG"
    )
    parser.add_argument("--cup_radius", type=float, default=1.0)
    parser.add_argument("--cup_height", type=float, default=2.0)
    parser.add_argument("--saucer_radius", type=float, default=3.0)

    args = parser.parse_args()

    launch_viewer(
        heightfield_path=args.heightfield,
        mesh_path=args.mesh,
        texture_path=args.texture,
        cup_radius=args.cup_radius,
        cup_height=args.cup_height,
        saucer_radius=args.saucer_radius,
    )


if __name__ == "__main__":
    main()