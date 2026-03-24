"""
SDF / height-field utilities for mesh generation and export.
"""

import numpy as np
from typing import Tuple


def heightfield_to_sdf_grid(
    heightfield: np.ndarray,
    saucer_radius: float = 3.0,
    grid_res: int = 64,
    z_range: Tuple[float, float] = (-0.5, 1.0),
) -> np.ndarray:
    """
    Convert a 2D height field to a 3D SDF grid for volumetric operations.
    """
    hf_res = heightfield.shape[0]
    sdf = np.ones((grid_res, grid_res, grid_res))

    x = np.linspace(-saucer_radius, saucer_radius, grid_res)
    y = np.linspace(-saucer_radius, saucer_radius, grid_res)
    z = np.linspace(z_range[0], z_range[1], grid_res)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Map (xx, yy) to heightfield indices
    ui = ((xx / saucer_radius + 1.0) * 0.5 * (hf_res - 1)).astype(int)
    vi = ((yy / saucer_radius + 1.0) * 0.5 * (hf_res - 1)).astype(int)
    ui = np.clip(ui, 0, hf_res - 1)
    vi = np.clip(vi, 0, hf_res - 1)

    h_vals = heightfield[vi, ui]
    sdf = zz - h_vals  # positive above surface, negative below

    return sdf


def export_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_path: str,
    obj_path: str,
    mtl_path: str,
):
    """Export mesh as OBJ with material and texture reference."""
    mtl_name = mtl_path.split("/")[-1].replace(".mtl", "")
    with open(mtl_path, "w") as f:
        f.write(f"newmtl {mtl_name}\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd {texture_path}\n")

    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_path.split('/')[-1]}\n")
        f.write(f"usemtl {mtl_name}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for face in faces:
            f.write(
                f"f {face[0]+1}/{face[0]+1} "
                f"{face[1]+1}/{face[1]+1} "
                f"{face[2]+1}/{face[2]+1}\n"
            )