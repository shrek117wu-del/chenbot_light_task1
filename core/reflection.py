"""
Reflection mapping for a cylindrical mirror cup on a flat/deformed saucer.

Implements the key ray-tracing-based reflection mapping from the paper:
  "Computational Mirror Cup and Saucer Art" (Wu et al., ACM TOG 2022)

For a cylindrical mirror of radius R and height H centered at the origin,
the observer is at position eye = (0, 0, eye_z).  For each pixel in the
*reflected* target image we solve for the corresponding point on the saucer.
"""

import numpy as np
from typing import Tuple, Optional


def cylinder_normal(p: np.ndarray) -> np.ndarray:
    """Outward unit normal on a vertical cylinder x^2+y^2=R^2 (ignoring z)."""
    n = np.zeros_like(p)
    n[..., 0] = p[..., 0]
    n[..., 1] = p[..., 1]
    length = np.sqrt(n[..., 0] ** 2 + n[..., 1] ** 2 + 1e-12)
    n[..., 0] /= length
    n[..., 1] /= length
    return n


def reflect_vector(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Reflect direction d about normal n.  d points *toward* the surface."""
    dot = np.sum(d * n, axis=-1, keepdims=True)
    return d - 2.0 * dot * n


def ray_cylinder_intersect(
    origin: np.ndarray, direction: np.ndarray, radius: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intersect rays with an infinite vertical cylinder x^2+y^2 = R^2.
    Returns (t, hit_mask).  t is the *first positive* intersection distance.
    """
    ox, oy = origin[..., 0], origin[..., 1]
    dx, dy = direction[..., 0], direction[..., 1]

    a = dx ** 2 + dy ** 2
    b = 2.0 * (ox * dx + oy * dy)
    c = ox ** 2 + oy ** 2 - radius ** 2

    disc = b ** 2 - 4.0 * a * c
    hit = disc >= 0

    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) / (2.0 * a + 1e-12)
    t2 = (-b + sqrt_disc) / (2.0 * a + 1e-12)

    # We want the first positive t
    t = np.where((t1 > 1e-6), t1, t2)
    hit = hit & (t > 1e-6)
    return t, hit


def ray_plane_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_z: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Intersect rays with a horizontal plane z = plane_z."""
    dz = direction[..., 2]
    t = (plane_z - origin[..., 2]) / (dz + 1e-12)
    hit = t > 1e-6
    return t, hit


def build_reflection_map(
    img_res: int,
    cup_radius: float = 1.0,
    cup_height: float = 2.0,
    saucer_radius: float = 3.0,
    eye_pos: np.ndarray = None,
    saucer_heightfield: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a mapping from reflected-view pixel → saucer (x, y) coordinate.

    Parameters
    ----------
    img_res : int
        Square image resolution.
    cup_radius : float
        Radius of the cylindrical mirror cup.
    cup_height : float
        Height of the cylindrical mirror cup.
    saucer_radius : float
        Radius of the saucer disc.
    eye_pos : (3,) array
        Observer eye position; default (0, -4*saucer_radius, 3*cup_height).
    saucer_heightfield : (img_res, img_res) array or None
        Optional heightfield displacement of the saucer (z offset).

    Returns
    -------
    saucer_uv : (img_res, img_res, 2) float
        For each reflected-view pixel, the (u, v) coordinate on the saucer
        in [0,1]^2.  (-1,-1) marks invalid pixels.
    cup_uv : (img_res, img_res, 2) float
        The (theta/2pi, z/H) coordinate on the cup surface.
    valid : (img_res, img_res) bool
    """
    if eye_pos is None:
        eye_pos = np.array([0.0, -4.0 * saucer_radius, 3.0 * cup_height])

    # Build a grid of look-at directions covering the cup from the eye
    # We parameterize the "virtual screen" so it subtends the cup
    u = np.linspace(-1, 1, img_res)
    v = np.linspace(-1, 1, img_res)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    # Target points on the cup surface (cylindrical parameterization)
    theta = np.pi * uu  # [-pi, pi]
    z = 0.5 * (vv + 1.0) * cup_height  # [0, H]

    cup_pts = np.stack(
        [cup_radius * np.cos(theta), cup_radius * np.sin(theta), z], axis=-1
    )

    # Ray from eye to cup point
    direction = cup_pts - eye_pos[None, None, :]
    dir_len = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = direction / (dir_len + 1e-12)

    # Check if ray hits the cylinder first
    t_cyl, hit_cyl = ray_cylinder_intersect(
        np.broadcast_to(eye_pos, direction.shape), direction, cup_radius
    )

    # Hit point on cylinder
    hit_pt = eye_pos[None, None, :] + t_cyl[..., None] * direction
    # Clamp z to [0, cup_height]
    z_ok = (hit_pt[..., 2] >= 0) & (hit_pt[..., 2] <= cup_height)
    hit_cyl = hit_cyl & z_ok

    # Normal at hit point
    normals = cylinder_normal(hit_pt)

    # Reflected direction
    ref_dir = reflect_vector(direction, normals)

    # Intersect reflected ray with saucer plane (z = 0 + heightfield)
    if saucer_heightfield is not None:
        # Iterative approach: start with z=0, then refine
        plane_z = 0.0
        for _ in range(3):
            t_plane, hit_plane = ray_plane_intersect(hit_pt, ref_dir, plane_z)
            saucer_pt = hit_pt + t_plane[..., None] * ref_dir
            # Look up height at that saucer position
            sx = saucer_pt[..., 0]
            sy = saucer_pt[..., 1]
            su = (sx / saucer_radius + 1.0) * 0.5
            sv = (sy / saucer_radius + 1.0) * 0.5
            iu = np.clip((su * (img_res - 1)).astype(int), 0, img_res - 1)
            iv = np.clip((sv * (img_res - 1)).astype(int), 0, img_res - 1)
            plane_z_map = saucer_heightfield[iv, iu]
            plane_z = np.mean(plane_z_map)
    else:
        plane_z = 0.0

    t_plane, hit_plane = ray_plane_intersect(hit_pt, ref_dir, plane_z)
    saucer_pt = hit_pt + t_plane[..., None] * ref_dir

    # Convert saucer point to UV
    sx = saucer_pt[..., 0]
    sy = saucer_pt[..., 1]
    su = (sx / saucer_radius + 1.0) * 0.5
    sv = (sy / saucer_radius + 1.0) * 0.5

    # Within-saucer mask
    r2 = sx ** 2 + sy ** 2
    in_saucer = r2 <= saucer_radius ** 2
    # Not inside cup
    outside_cup = r2 >= cup_radius ** 2

    valid = hit_cyl & hit_plane & in_saucer & outside_cup

    saucer_uv = np.stack([su, sv], axis=-1)
    saucer_uv[~valid] = -1.0

    # Cup UV
    cup_theta = np.arctan2(hit_pt[..., 1], hit_pt[..., 0])
    cup_u = (cup_theta / np.pi + 1.0) * 0.5
    cup_v = hit_pt[..., 2] / cup_height
    cup_uv = np.stack([cup_u, cup_v], axis=-1)
    cup_uv[~valid] = -1.0

    return saucer_uv, cup_uv, valid


def build_direct_view_map(
    img_res: int,
    saucer_radius: float = 3.0,
    cup_radius: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the trivial direct view: orthographic top-down view of the saucer.

    Returns
    -------
    saucer_uv : (img_res, img_res, 2)
    valid : (img_res, img_res) bool
    """
    u = np.linspace(0, 1, img_res)
    v = np.linspace(0, 1, img_res)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    saucer_uv = np.stack([uu, vv], axis=-1)

    # Mask: inside saucer but outside cup
    x = (uu - 0.5) * 2.0 * saucer_radius
    y = (vv - 0.5) * 2.0 * saucer_radius
    r2 = x ** 2 + y ** 2
    valid = (r2 <= saucer_radius ** 2) & (r2 >= cup_radius ** 2)

    return saucer_uv, valid