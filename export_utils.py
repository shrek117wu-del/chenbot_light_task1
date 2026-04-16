"""
Export the optimised saucer as a textured OBJ + MTL + PNG,
plus a separate OBJ for the mirror cylinder.

Coordinate convention used by the 3-D viewer (Three.js, Y-up):
  Python (x, y, z_height)  →  Three.js (x, z_height, -y)

Paper world coords:
  saucer xy∈[-0.5,0.5]×[-1.5,-0.5], height z∈[0,1.5]
  → Three.js: x∈[-0.5,0.5], Three_y=height, Three_z∈[0.5,1.5]

Mirror cylinder:
  centre at Python origin (0,0), height h=2.0
  → Three.js: (0, h/2, 0) centre = (0, 1.0, 0)
"""
import os
import math
import numpy as np
import cv2


def export_saucer_obj(grid_x, grid_y, h_grid, color_grid,
                      obj_path="saucer.obj",
                      tex_path="texture.png"):
    """
    Write a textured OBJ file for the optimised saucer.
    UV coordinates map directly to image pixels of the texture atlas.

    Args:
        grid_x, grid_y : [H, W] numpy — saucer XY in world coords
        h_grid         : [H, W] numpy — height (Z in world, Y in Three.js)
        color_grid     : [H, W, 3] numpy in [0, 1] RGB — per-vertex colours
        obj_path       : output .obj file
        tex_path       : output texture PNG file
    """
    H, W = grid_x.shape

    # ---- Texture -------------------------------------------------------
    # color_grid is [H, W, 3] in [0, 1] RGB
    tex_rgb = (color_grid.clip(0, 1) * 255).astype(np.uint8)

    # Ensure the texture is at least showing something (avoid grey/blank)
    # Upscale to a nicer resolution for better quality (e.g. 512x512)
    target_size = max(512, H, W)
    tex_up = cv2.resize(tex_rgb, (target_size, target_size),
                        interpolation=cv2.INTER_LINEAR)
    tex_bgr = tex_up[:, :, ::-1]                        # RGB → BGR for cv2
    cv2.imwrite(tex_path, tex_bgr)
    print(f"  texture saved → {tex_path}  ({target_size}×{target_size} px)")

    # MTL path: only the basename (OBJ and MTL sit in the same dir)
    mtl_name = os.path.splitext(os.path.basename(obj_path))[0] + ".mtl"
    mtl_path  = os.path.join(os.path.dirname(obj_path), mtl_name)
    tex_name  = os.path.basename(tex_path)

    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("o saucer\n")
        f.write("usemtl saucer_mat\n\n")

        # Vertices: Three.js Y-up →  v  x  height  -y
        # Python saucer y ∈ [-1.5, -0.5] → Three.js z ∈ [0.5, 1.5] (in front)
        for i in range(H):
            for j in range(W):
                vx =  float(grid_x[i, j])
                vy =  float(h_grid[i, j])     # height → Three.js Y
                vz = -float(grid_y[i, j])     # -py    → Three.js Z (saucer in +Z)
                f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")

        # UV coordinates: u = j/(W-1),  v = 1 - i/(H-1)  (OpenGL V-flip)
        for i in range(H):
            for j in range(W):
                u = j / (W - 1)
                v = 1.0 - i / (H - 1)
                f.write(f"vt {u:.6f} {v:.6f}\n")

        # Faces (1-indexed).  Each quad → 2 CCW triangles.
        for i in range(H - 1):
            for j in range(W - 1):
                a = i * W + j + 1          # 1-indexed
                b = i * W + j + 2
                c = (i + 1) * W + j + 1
                d = (i + 1) * W + j + 2
                f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
                f.write(f"f {b}/{b} {d}/{d} {c}/{c}\n")

    # ---- MTL -----------------------------------------------------------
    with open(mtl_path, "w") as f:
        f.write("newmtl saucer_mat\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("Ns 0.0\n")
        f.write("illum 1\n")
        f.write(f"map_Kd {tex_name}\n")

    print(f"  OBJ saved → {obj_path}  ({(H-1)*(W-1)*2} triangles)")


def export_cylinder_obj(radius=0.4, height=2.0, segments=128,
                        obj_path="cylinder.obj"):
    """
    Write an OBJ for the mirror cylinder (for the 3-D viewer).
    Three.js Y-up: cylinder axis along Y, bottom at y=0, top at y=height.
    In the paper the cup is centred at the origin (x=0, z=0 in Three.js).
    """
    verts, uvs, normals = [], [], []
    faces = []

    for seg in range(segments + 1):
        theta = 2.0 * math.pi * seg / segments
        x =  radius * math.cos(theta)
        z =  radius * math.sin(theta)
        nx = math.cos(theta)
        nz = math.sin(theta)
        # bottom vertex
        verts.append((x, 0.0,    z))
        normals.append((nx, 0.0, nz))
        uvs.append((seg / segments, 0.0))
        # top vertex
        verts.append((x, height, z))
        normals.append((nx, 0.0, nz))
        uvs.append((seg / segments, 1.0))

    for seg in range(segments):
        a = seg * 2 + 1        # 1-indexed
        b = seg * 2 + 2
        c = (seg + 1) * 2 + 1
        d = (seg + 1) * 2 + 2
        faces.append((a, b, d, c))

    with open(obj_path, "w") as f:
        f.write("o cylinder\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for vt in uvs:
            f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
        for vn in normals:
            f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
        for a, b, d, c in faces:
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {d}/{d}/{d}\n")
            f.write(f"f {a}/{a}/{a} {d}/{d}/{d} {c}/{c}/{c}\n")

    print(f"  cylinder OBJ saved → {obj_path}  ({segments*2} triangles)")


def export_obj_and_texture(grid_x, grid_y, h_grid, color_grid,
                           obj_path="saucer.obj", tex_path="texture.png"):
    """Main entry point: export saucer + cylinder OBJs."""
    export_saucer_obj(grid_x, grid_y, h_grid, color_grid, obj_path, tex_path)
    cyl_path = obj_path.replace("saucer.obj", "cylinder.obj")
    export_cylinder_obj(obj_path=cyl_path)
