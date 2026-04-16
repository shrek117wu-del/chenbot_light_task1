"""
Geometry utilities: reflection computation and heightfield triangle mesh generation.
"""
import numpy as np
import torch
from scipy.optimize import root_scalar


# ---------------------------------------------------------------------------
# 2-D cylindrical mirror reflection
# ---------------------------------------------------------------------------

def compute_reflection_2d(P, Q, r=0.4):
    """
    Given:
      P (2-D camera position, in the horizontal plane),
      Q (2-D saucer point),
      cylindrical mirror radius r,
    find the 2-D position R of the virtual image of Q as seen from P.

    The algorithm searches for the reflection point T on the circle |T|=r such
    that ∠PTV = ∠QTV (angle of incidence = angle of reflection).
    This is equivalent to the bisector of (P-T) and (Q-T) being parallel to
    the normal T at the circle.
    """
    dist_Q = np.linalg.norm(Q)
    if dist_Q <= r:
        return Q.copy()   # inside the mirror (shouldn't happen)

    def objective(theta_T):
        T      = r * np.array([np.cos(theta_T), np.sin(theta_T)])
        vec_TP = P - T
        vec_TQ = Q - T
        n      = T / r   # outward normal
        dP = vec_TP / (np.linalg.norm(vec_TP) + 1e-10)
        dQ = vec_TQ / (np.linalg.norm(vec_TQ) + 1e-10)
        # Snell condition: (dP + dQ) parallel to n  →  cross product = 0
        bis = dP + dQ
        return bis[0] * n[1] - bis[1] * n[0]

    # Coarse scan to find sign changes
    thetas = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    vals   = [objective(th) for th in thetas]

    roots = []
    for i in range(len(thetas) - 1):
        if vals[i] * vals[i + 1] <= 0.0:
            try:
                res = root_scalar(objective,
                                  bracket=[thetas[i], thetas[i + 1]],
                                  method='brentq', xtol=1e-9)
                if res.converged:
                    roots.append(res.root)
            except Exception:
                pass

    if not roots:
        return Q.copy()

    # Among valid roots pick the one where both P and Q are on the same side
    # of the tangent (both outside the circle) – i.e. the physically valid one
    best = None
    for rt in roots:
        T     = r * np.array([np.cos(rt), np.sin(rt)])
        n     = T / r
        vec_TP = P - T
        vec_TQ = Q - T
        if np.dot(vec_TP, n) > 0 and np.dot(vec_TQ, n) > 0:
            best = rt
            break

    if best is None:
        best = roots[0]

    T          = r * np.array([np.cos(best), np.sin(best)])
    tan_dir    = np.array([-np.sin(best), np.cos(best)])  # tangent at T
    n          = T / r
    vec_TQ     = Q - T
    # Reflect vec_TQ across the tangent (flip normal component)
    proj_tan   = np.dot(vec_TQ, tan_dir) * tan_dir
    proj_norm  = np.dot(vec_TQ, n)       * n
    vec_TR     = proj_tan - proj_norm    # reflected direction
    R          = T + vec_TR
    return R


def precompute_reflection_grid(grid_x, grid_y, P_2d, r=0.4):
    """
    For an (H, W) saucer heightfield with top-view 2-D coordinates
    (grid_x, grid_y), compute the reflected positions (R_x, R_y) for every
    vertex using cylindrical-mirror reflection.
    """
    H, W = grid_x.shape
    R_x  = np.zeros_like(grid_x)
    R_y  = np.zeros_like(grid_y)
    for i in range(H):
        for j in range(W):
            q    = np.array([grid_x[i, j], grid_y[i, j]])
            R    = compute_reflection_2d(P_2d, q, r=r)
            R_x[i, j] = R[0]
            R_y[i, j] = R[1]
    return R_x, R_y


# ---------------------------------------------------------------------------
# Triangle mesh generation for a heightfield grid
# ---------------------------------------------------------------------------

def get_grid_triangles(H, W):
    """
    Generate CCW triangle indices for a heightfield of shape (H, W).
    Returns int32 array of shape (2*(H-1)*(W-1), 3).
    """
    tris = []
    for i in range(H - 1):
        for j in range(W - 1):
            a = i * W + j
            b = i * W + j + 1
            c = (i + 1) * W + j
            d = (i + 1) * W + j + 1
            tris.append([a, b, c])  # lower-left triangle
            tris.append([b, d, c])  # upper-right triangle
    return np.array(tris, dtype=np.int32)


# ---------------------------------------------------------------------------
# Per-face UV assignment (for the texturing stage, Section 3.4.2)
# ---------------------------------------------------------------------------

def assign_face_uvs(face_uv_d, face_uv_r, face_vis_d, face_vis_r):
    """
    Given the projected 2-D coords of each triangular face in the direct view
    and in the reflected view, assign texture UVs following Section 3.4.2:

      Case 1: visible only in direct  → UV from direct projection
      Case 2: visible only in reflect → UV from reflect projection
      Case 3: visible in both        → UV from reflect projection
              (pixel in Id modified to average of Id and Ir colours)
      Case 4: invisible in both       → UV is arbitrary (zeros)

    face_uv_d / face_uv_r : [F, 3, 2] NDC coords of triangle vertices
    face_vis_d / face_vis_r: [F] boolean – triangle visible in that view

    Returns face_uvs [F, 3, 2] and face_view [F] int  (1/2/3/4)
    """
    F = face_uv_d.shape[0]
    face_uvs  = np.zeros((F, 3, 2), dtype=np.float32)
    face_view = np.zeros(F, dtype=np.int32)

    for fi in range(F):
        vd = face_vis_d[fi]
        vr = face_vis_r[fi]
        if vd and not vr:
            face_uvs[fi]  = face_uv_d[fi]
            face_view[fi] = 1
        elif vr and not vd:
            face_uvs[fi]  = face_uv_r[fi]
            face_view[fi] = 2
        elif vd and vr:
            face_uvs[fi]  = face_uv_r[fi]
            face_view[fi] = 3
        else:
            face_uvs[fi]  = face_uv_d[fi]  # arbitrary
            face_view[fi] = 4

    return face_uvs, face_view
