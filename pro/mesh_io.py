"""
pro/mesh_io.py – OBJ Mesh I/O and Heightfield Conversion
=========================================================

Loads the photo saucer/cup OBJ models and converts them into the regular
Cartesian heightfield grid used by the core optimisation solver.

Paper coordinate system (used throughout core/):
  • Saucer : x∈[-0.5, 0.5], y∈[-1.5, -0.5]  → centre at (0, -1, 0), radius 0.5
  • Cup    : cylindrical mirror centred at origin, radius ≈ 0.4
  • Camera : (0, -5.5, 5),  z-axis is UP
"""

import numpy as np
from scipy.interpolate import griddata


# ---------------------------------------------------------------------------
# OBJ loading
# ---------------------------------------------------------------------------

def load_obj(path):
    """
    Load a Wavefront OBJ file.

    Returns
    -------
    verts : [N, 3] float32 numpy array  (x, y, z)
    faces : [F, 3] int32  numpy array   (0-indexed vertex indices)
    """
    verts, faces = [], []
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('v '):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                # Support "v", "v/vt", "v//vn", "v/vt/vn"
                idxs = [int(p.split('/')[0]) - 1 for p in parts]
                if len(idxs) == 3:
                    faces.append(idxs)
                elif len(idxs) == 4:
                    # Split quad into two triangles
                    faces.append([idxs[0], idxs[1], idxs[2]])
                    faces.append([idxs[0], idxs[2], idxs[3]])
                elif len(idxs) > 4:
                    # Fan triangulation for n-gons
                    for k in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[k], idxs[k + 1]])
    return (np.array(verts, dtype=np.float32),
            np.array(faces,  dtype=np.int32))


# ---------------------------------------------------------------------------
# Saucer normalisation
# ---------------------------------------------------------------------------

def normalize_saucer(verts, target_radius=0.5, target_center=(0.0, -1.0)):
    """
    Rescale and reposition a saucer mesh into the paper coordinate system.

    The photo_saucer OBJ models are full-disc meshes centred at the origin
    with radius ≈ 1.6.  This function maps them to the paper geometry:
      • Centre     → (0, -1, 0)  in world XY
      • Radius     → 0.5
      • Height (z) → scaled proportionally

    Parameters
    ----------
    verts         : [N, 3] original vertices
    target_radius : desired disc radius in paper units   (default 0.5)
    target_center : (cx, cy) destination disc centre      (default (0, -1))

    Returns
    -------
    [N, 3] normalised vertices
    """
    xy = verts[:, :2]
    # Bounding-box centre of the original mesh
    bbox_center = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    # Largest vertex radius from that centre
    orig_radius = float(np.max(np.linalg.norm(xy - bbox_center, axis=1)))
    if orig_radius < 1e-8:
        orig_radius = 1.0

    scale = target_radius / orig_radius

    out = verts.copy()
    out[:, 0] = (verts[:, 0] - bbox_center[0]) * scale + target_center[0]
    out[:, 1] = (verts[:, 1] - bbox_center[1]) * scale + target_center[1]
    out[:, 2] = verts[:, 2] * scale   # height scaled proportionally
    return out


# ---------------------------------------------------------------------------
# Mesh → regular Cartesian heightfield
# ---------------------------------------------------------------------------

def mesh_to_heightfield(verts, resolution=(150, 150)):
    """
    Sample a 3-D mesh (already in paper coordinates) onto a regular
    Cartesian heightfield grid.

    Uses scipy.interpolate.griddata (linear, then nearest for extrapolation)
    so grid points outside the mesh's convex hull are still filled in.

    Parameters
    ----------
    verts      : [N, 3] mesh vertices in paper coordinate system
    resolution : (H, W) output grid resolution

    Returns
    -------
    grid_x : [H, W] float32  – x coordinates
    grid_y : [H, W] float32  – y coordinates
    grid_z : [H, W] float32  – interpolated height values
    """
    H, W = resolution

    xs = np.linspace(-0.5,  0.5, W, dtype=np.float32)
    ys = np.linspace(-1.5, -0.5, H, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    src_xy = verts[:, :2]
    src_z  = verts[:, 2]

    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Linear interpolation on the Delaunay triangulation
    z_lin = griddata(src_xy, src_z, query, method='linear')

    # Fill NaNs (outside convex hull) with nearest-neighbour
    if np.isnan(z_lin).any():
        z_nn = griddata(src_xy, src_z, query, method='nearest')
        z_lin = np.where(np.isnan(z_lin), z_nn, z_lin)

    grid_z = z_lin.reshape(H, W).astype(np.float32)
    return grid_x, grid_y, grid_z


# ---------------------------------------------------------------------------
# Cup profile helpers
# ---------------------------------------------------------------------------

def get_cup_effective_radius(cup_verts, low_fraction=0.25):
    """
    Return the effective 2-D reflection radius of the cup at the saucer
    surface level (near z = 0).

    For a straight cup this equals its stated radius (≈ 0.4).
    For a conical cup we use the bottom quarter of the cup height.

    Parameters
    ----------
    cup_verts    : [N, 3] cup OBJ vertices (z is height, 0 at base)
    low_fraction : fraction of total height counted as "near the saucer"

    Returns
    -------
    float – effective radius
    """
    z = cup_verts[:, 2]
    z_max = float(z.max())
    # Select vertices in the lower portion
    mask = z <= max(low_fraction * z_max, 1e-6)
    if mask.sum() == 0:
        mask = np.ones(len(z), dtype=bool)
    radii = np.sqrt(cup_verts[mask, 0] ** 2 + cup_verts[mask, 1] ** 2)
    return float(np.mean(radii))
