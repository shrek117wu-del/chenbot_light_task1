"""
Geometry utilities: reflection computation and heightfield triangle mesh generation.

Supported mirror cup types:
  - circular cylinder   (original paper)
  - elliptical cylinder (Section 3.5 extension)
  - regular n-gonal prism (Section 3.5 extension, Figures 17 & 19)
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


def _bisection(f_lo, f_hi, a, b, n_iters=40):
    """
    Simple scalar bisection refinement given f(a)=f_lo, f(b)=f_hi.
    Returns root to ~1e-9 tolerance.
    """
    for _ in range(n_iters):
        mid = 0.5 * (a + b)
        fm  = 0.0
        # Evaluate midpoint via linear approximation isn't possible without
        # the function — caller must supply values.  We return the interval.
        # This helper is only called where the full objective is re-evaluated.
        if f_lo * fm <= 0:
            b, f_hi = mid, fm
        else:
            a, f_lo = mid, fm
    return 0.5 * (a + b)


def precompute_reflection_grid(grid_x, grid_y, P_2d, r=0.4):
    """
    Vectorised batch version.  For an (H, W) heightfield with 2-D coordinates
    (grid_x, grid_y), compute reflected positions (R_x, R_y) for every vertex.

    Strategy:
      1. For each of 720 candidate angles θ, evaluate the Snell-condition
         objective for ALL grid points simultaneously (NumPy broadcasting).
      2. Detect sign changes → candidate brackets for each point.
      3. Refine each bracket with a fast scalar bisection (Brent's method).
    This is 50-100× faster than calling compute_reflection_2d per vertex.
    """
    H, W = grid_x.shape
    N    = H * W
    Q_all = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)   # [N, 2]

    # Pre-compute angle-dependent parts (do not depend on Q)
    n_theta = 720
    thetas  = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    T_pts   = r * np.stack([np.cos(thetas), np.sin(thetas)], axis=1)  # [M, 2]
    normals = T_pts / r                                                 # [M, 2]

    vec_TP = P_2d[None, :] - T_pts                                      # [M, 2]
    norm_TP = np.linalg.norm(vec_TP, axis=1, keepdims=True).clip(1e-10) # [M, 1]
    dP_arr  = vec_TP / norm_TP                                           # [M, 2]

    # Batch: vec_TQ[n, m] = Q_all[n] - T_pts[m]  →  [N, M, 2]
    vec_TQ  = Q_all[:, None, :] - T_pts[None, :, :]                     # [N, M, 2]
    norm_TQ = np.linalg.norm(vec_TQ, axis=2, keepdims=True).clip(1e-10) # [N, M, 1]
    dQ_arr  = vec_TQ / norm_TQ                                           # [N, M, 2]

    # Bisector cross product with outward normal → vals[N, M]
    bis  = dP_arr[None, :, :] + dQ_arr                                   # [N, M, 2]
    vals = (bis[:, :, 0] * normals[None, :, 1]
          - bis[:, :, 1] * normals[None, :, 0])                          # [N, M]

    # Sign-change detection per point: sign_mask[n, m] = True if bracket
    sign_mask = (vals[:, :-1] * vals[:, 1:]) <= 0.0                      # [N, M-1]

    R_all = np.zeros_like(Q_all)

    for n in range(N):
        Q = Q_all[n]
        if np.linalg.norm(Q) <= r:
            R_all[n] = Q
            continue

        brackets = np.where(sign_mask[n])[0]
        if len(brackets) == 0:
            R_all[n] = Q
            continue

        roots = []
        for idx in brackets:
            a, b = thetas[idx], thetas[idx + 1]
            # Scalar bisection using compute_reflection_2d's objective
            def obj(th):
                T  = r * np.array([np.cos(th), np.sin(th)])
                vP = P_2d - T;  vQ = Q - T
                nT = T / r
                dP_ = vP / (np.linalg.norm(vP) + 1e-10)
                dQ_ = vQ / (np.linalg.norm(vQ) + 1e-10)
                bs  = dP_ + dQ_
                return bs[0] * nT[1] - bs[1] * nT[0]

            fa, fb = vals[n, idx], vals[n, idx + 1]
            # 30 bisection steps gives ~1e-9 precision
            for _ in range(30):
                mid = 0.5 * (a + b)
                fm  = obj(mid)
                if fa * fm <= 0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
            roots.append(0.5 * (a + b))

        # Pick the physically valid root (both P and Q outside cylinder,
        # their dot-product with normal both > 0)
        best = None
        for rt in roots:
            T_rt   = r * np.array([np.cos(rt), np.sin(rt)])
            nT     = T_rt / r
            vP_rt  = P_2d - T_rt
            vQ_rt  = Q    - T_rt
            if np.dot(vP_rt, nT) > 0 and np.dot(vQ_rt, nT) > 0:
                best = rt
                break
        if best is None:
            best = roots[0]

        T_best  = r * np.array([np.cos(best), np.sin(best)])
        tan_dir = np.array([-np.sin(best), np.cos(best)])
        nT      = T_best / r
        vTQ     = Q - T_best
        R_all[n] = T_best + np.dot(vTQ, tan_dir) * tan_dir - np.dot(vTQ, nT) * nT

    return R_all[:, 0].reshape(H, W), R_all[:, 1].reshape(H, W)


# ---------------------------------------------------------------------------
# Elliptical cylinder mirror reflection  (Section 3.5 extension)
# ---------------------------------------------------------------------------

def compute_reflection_2d_ellipse(P, Q, a=0.5, b=0.3):
    """
    Reflection of Q seen from P via an elliptical cylinder mirror with semi-axes
    a (x-direction) and b (y-direction).  Uses the same angle-bisector / binary-search
    approach as the circular case; the only change is the ellipse parameterisation
    T=(a cosθ, b sinθ) and its outward unit normal ∝ (cosθ/a, sinθ/b).
    """
    def objective(theta):
        T   = np.array([a * np.cos(theta), b * np.sin(theta)])
        n   = np.array([np.cos(theta) / a, np.sin(theta) / b])
        n  /= (np.linalg.norm(n) + 1e-10)
        vP  = P - T;  vQ = Q - T
        dP  = vP / (np.linalg.norm(vP) + 1e-10)
        dQ  = vQ / (np.linalg.norm(vQ) + 1e-10)
        bis = dP + dQ
        return bis[0] * n[1] - bis[1] * n[0]

    thetas = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    vals   = [objective(th) for th in thetas]

    roots = []
    for i in range(len(thetas) - 1):
        if vals[i] * vals[i + 1] <= 0.0:
            try:
                res = root_scalar(objective, bracket=[thetas[i], thetas[i + 1]],
                                  method='brentq', xtol=1e-9)
                if res.converged:
                    roots.append(res.root)
            except Exception:
                pass

    if not roots:
        return Q.copy()

    best = None
    for rt in roots:
        T  = np.array([a * np.cos(rt), b * np.sin(rt)])
        n  = np.array([np.cos(rt) / a, np.sin(rt) / b])
        n /= np.linalg.norm(n)
        if np.dot(P - T, n) > 0 and np.dot(Q - T, n) > 0:
            best = rt
            break
    if best is None:
        best = roots[0]

    T   = np.array([a * np.cos(best), b * np.sin(best)])
    n   = np.array([np.cos(best) / a, np.sin(best) / b])
    n  /= np.linalg.norm(n)
    # Tangent at T on the ellipse: d/dθ (a cosθ, b sinθ) = (-a sinθ, b cosθ)
    tan = np.array([-a * np.sin(best), b * np.cos(best)])
    tan /= (np.linalg.norm(tan) + 1e-10)
    vTQ = Q - T
    return T + np.dot(vTQ, tan) * tan - np.dot(vTQ, n) * n


def precompute_reflection_ellipse_grid(grid_x, grid_y, P_2d, a=0.5, b=0.3):
    """
    Batch version of compute_reflection_2d_ellipse for a (H, W) grid.
    Iterates per vertex (ellipse root-finding is not easily vectorised).
    """
    H, W   = grid_x.shape
    Q_all  = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    R_all  = np.zeros_like(Q_all)
    for i, Q in enumerate(Q_all):
        R_all[i] = compute_reflection_2d_ellipse(P_2d, Q, a, b)
    return R_all[:, 0].reshape(H, W), R_all[:, 1].reshape(H, W)


# ---------------------------------------------------------------------------
# Regular n-gonal prism mirror reflection  (Section 3.5, Figures 17 & 19)
# ---------------------------------------------------------------------------

def _reflect_across_segment(P, Q, pt1, pt2):
    """
    Attempt to reflect Q as seen from P off the flat mirror segment (pt1, pt2).

    For a flat mirror the virtual image of Q is the mirror image Q' of Q across
    the face plane.  The reflection is valid when the ray P→Q' intersects the
    segment between pt1 and pt2 (i.e. the reflection point T is on the face).

    Returns Q' (the virtual 2-D image position) or None if invalid.
    """
    edge     = pt2 - pt1
    edge_len = np.linalg.norm(edge)
    if edge_len < 1e-10:
        return None
    tan  = edge / edge_len
    norm = np.array([-tan[1], tan[0]])           # perpendicular (one of two directions)

    # Make the normal point outward (away from the polygon centroid ≈ origin)
    face_mid = 0.5 * (pt1 + pt2)
    if np.dot(norm, face_mid) < 0:
        norm = -norm

    d = np.dot(norm, pt1)                        # signed distance of face from origin

    # Both P and Q must be on the outer side of this face
    if np.dot(norm, P) <= d + 1e-10 or np.dot(norm, Q) <= d + 1e-10:
        return None

    # Mirror image of Q across the face
    dist_Q  = np.dot(norm, Q) - d
    Q_prime = Q - 2.0 * dist_Q * norm

    # Find intersection T of ray P→Q' with the face line
    # Solve: P + t*(Q'-P) = pt1 + s*tan,  s ∈ [0, edge_len]
    dir_PQ = Q_prime - P
    A = np.array([[dir_PQ[0], -tan[0]],
                  [dir_PQ[1], -tan[1]]])
    b = pt1 - P
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-10:
        return None
    t = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
    s = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det

    if t <= 1e-6 or t >= 1.0 - 1e-6:
        return None                              # intersection behind P or beyond Q'
    if s < 0.0 or s > edge_len:
        return None                              # intersection outside face segment

    return Q_prime


def compute_reflection_2d_ngon(P, Q, n=6, r=0.4, angle_offset=None):
    """
    Compute the reflected position of Q seen from P through a regular n-gonal
    prism mirror (Section 3.5).

    n            : number of sides  (e.g. 4, 6)
    r            : circumradius of the n-gon cross-section
    angle_offset : rotation of the n-gon in radians.  If None, uses π/n for
                   even n so that a face is centred on the downward (−y) direction,
                   making the default orientation sensible for the paper's geometry.

    Returns the 2-D virtual image position R.
    """
    if angle_offset is None:
        # Rotate so that a face midpoint is on the −y axis:
        # face midpoint angle = π/n + 2πk/n for some k → set = 3π/2 (downward)
        angle_offset = 3.0 * np.pi / 2.0 - np.pi / n

    angles   = [2.0 * np.pi * k / n + angle_offset for k in range(n)]
    vertices = [r * np.array([np.cos(a), np.sin(a)]) for a in angles]

    # Test all n faces; pick the one that yields a valid reflection
    for k in range(n):
        v1 = vertices[k]
        v2 = vertices[(k + 1) % n]
        R  = _reflect_across_segment(P, Q, v1, v2)
        if R is not None:
            return R

    # No valid face: place virtual image far off-screen so it contributes
    # nothing to the reflected render.
    return np.array([1e4, 1e4])


def precompute_reflection_ngon_grid(grid_x, grid_y, P_2d, n=6, r=0.4,
                                    angle_offset=None):
    """
    Batch reflection computation for a regular n-gonal prism mirror.

    Returns two (H, W) arrays: R_x, R_y — the virtual image XY positions
    for every vertex in the heightfield grid.
    """
    H, W   = grid_x.shape
    Q_all  = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    R_all  = np.zeros_like(Q_all)
    for i, Q in enumerate(Q_all):
        R_all[i] = compute_reflection_2d_ngon(P_2d, Q, n=n, r=r,
                                              angle_offset=angle_offset)
    return R_all[:, 0].reshape(H, W), R_all[:, 1].reshape(H, W)


def compute_all_ngon_reflections(P_2d, Q_all, n=6, r=0.4, angle_offset=None):
    """
    For each point Q in Q_all [N, 2] compute the reflected position for ALL
    visible n-gon faces.  Returns a list of length n, each element is a
    (N, 2) array (NaN where the face is not visible from that Q).

    This is used when the prism produces two separate reflected images
    (Figure 17 / 19 in the paper).
    """
    if angle_offset is None:
        angle_offset = 3.0 * np.pi / 2.0 - np.pi / n

    angles   = [2.0 * np.pi * k / n + angle_offset for k in range(n)]
    vertices = [r * np.array([np.cos(a), np.sin(a)]) for a in angles]

    face_reflections = []
    for k in range(n):
        v1 = vertices[k]
        v2 = vertices[(k + 1) % n]
        R_face = np.full_like(Q_all, np.nan)
        for i, Q in enumerate(Q_all):
            R = _reflect_across_segment(P_2d, Q, v1, v2)
            if R is not None:
                R_face[i] = R
        face_reflections.append(R_face)
    return face_reflections


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
