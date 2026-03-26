#!/usr/bin/env python3
"""
Fixed example: Run the blue cat scene (Scene 2) from demo_luycho_scenes.py
Fixes applied:
  1. numpy.ptp() → np.ptp() for NumPy 2.0 compatibility
  2. matplotlib color clamped to [0,1] range
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# ═══════════════════════════════════════════════════════════════
# Helper functions (from demo_luycho_scenes.py)
# ═══════════════════════════════════════════════════════════════

def _coord_grid(res):
    u = np.linspace(-1, 1, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    r = np.sqrt(uu**2 + vv**2)
    theta = np.arctan2(vv, uu)
    return uu, vv, r, theta

def _rgb(res, r, g, b):
    img = np.zeros((res, res, 3))
    img[..., 0] = r; img[..., 1] = g; img[..., 2] = b
    return img

def _blend(bg, fg, alpha):
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    return bg * (1 - a) + fg * a

def _ellipse_mask(res, cx, cy, rx, ry):
    u = np.linspace(0, 1, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    return (((uu - cx) / rx)**2 + ((vv - cy) / ry)**2 <= 1.0).astype(float)

def _circle_mask_abs(res, cx, cy, radius):
    u = np.linspace(0, 1, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    return (((uu - cx)**2 + (vv - cy)**2) <= radius**2).astype(float)

def _rect_mask(res, x0, y0, x1, y1):
    u = np.linspace(0, 1, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    return ((uu >= x0) & (uu <= x1) & (vv >= y0) & (vv <= y1)).astype(float)

def _triangle_mask(res, x0, y0, x1, y1, x2, y2):
    u = np.linspace(0, 1, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    def sign(px, py, ax, ay, bx, by):
        return (px - bx) * (ay - by) - (ax - bx) * (py - by)
    d1 = sign(uu, vv, x0, y0, x1, y1)
    d2 = sign(uu, vv, x1, y1, x2, y2)
    d3 = sign(uu, vv, x2, y2, x0, y0)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return (~(has_neg & has_pos)).astype(float)


# ═══════════════════════════════════════════════════════════════
# Scene 2: Blue saucer — Black cat
# ═══════════════════════════════════════════════════════════════

def make_scene2_reflected(res=512):
    """Sitting black cat — reflected in the silver mirror cup."""
    bg = _rgb(res, 0.25, 0.45, 0.80)
    body = _ellipse_mask(res, 0.50, 0.55, 0.10, 0.16)
    head = _circle_mask_abs(res, 0.50, 0.34, 0.07)
    ear_l = _triangle_mask(res, 0.43, 0.34, 0.40, 0.22, 0.46, 0.28)
    ear_r = _triangle_mask(res, 0.57, 0.34, 0.54, 0.28, 0.60, 0.22)
    u = np.linspace(0, 1, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    tail_x = 0.62 + 0.08 * np.sin(3 * np.pi * (vv - 0.50))
    tail = ((np.abs(uu - tail_x) < 0.015) & (vv > 0.50) & (vv < 0.72)).astype(float)
    tail = gaussian_filter(tail, sigma=2)
    cat = np.clip(body + head + ear_l + ear_r + tail, 0, 1)
    cat_color = _rgb(res, 0.08, 0.08, 0.10)
    img = bg.copy()
    img = _blend(img, cat_color, cat)
    eye_l = _circle_mask_abs(res, 0.47, 0.32, 0.012)
    eye_r = _circle_mask_abs(res, 0.53, 0.32, 0.012)
    img = _blend(img, _rgb(res, 0.95, 0.90, 0.20), eye_l + eye_r)
    belly = _ellipse_mask(res, 0.50, 0.52, 0.06, 0.08)
    img = _blend(img, _rgb(res, 0.75, 0.40, 0.55), belly * 0.5)
    for dy in [-0.01, 0, 0.01]:
        whisker = _rect_mask(res, 0.32, 0.345+dy, 0.44, 0.348+dy)
        whisker += _rect_mask(res, 0.56, 0.345+dy, 0.68, 0.348+dy)
        img = _blend(img, _rgb(res, 0.9, 0.9, 0.9), np.clip(whisker, 0, 1) * 0.7)
    tile_y = vv > 0.72
    tile_check = ((np.floor(uu*8).astype(int) + np.floor(vv*8).astype(int)) % 2)
    tile_pattern = (tile_check * tile_y).astype(float)
    img = _blend(img, _rgb(res, 0.30, 0.55, 0.75), tile_pattern * 0.4)
    return np.clip(img, 0, 1)


def make_scene2_direct(res=512):
    """Distorted black cat on blue tile floor — direct view on saucer."""
    uu, vv, r, theta = _coord_grid(res)
    bg = _rgb(res, 0.22, 0.42, 0.78)
    img = bg.copy()
    u01 = np.linspace(0, 1, res)
    uu01, vv01 = np.meshgrid(u01, u01, indexing="xy")
    check = ((np.floor(uu01*10).astype(int) + np.floor(vv01*10).astype(int)) % 2)
    img = _blend(img, _rgb(res, 0.18, 0.38, 0.70), check.astype(float) * 0.3)
    cat_body = _ellipse_mask(res, 0.48, 0.50, 0.15, 0.28)
    cat_head = _ellipse_mask(res, 0.48, 0.20, 0.08, 0.06)
    ear_l = _triangle_mask(res, 0.42, 0.20, 0.38, 0.10, 0.44, 0.15)
    ear_r = _triangle_mask(res, 0.54, 0.20, 0.52, 0.15, 0.58, 0.10)
    tail = _ellipse_mask(res, 0.62, 0.60, 0.03, 0.18)
    cat = np.clip(cat_body + cat_head + ear_l + ear_r + tail, 0, 1)
    img = _blend(img, _rgb(res, 0.06, 0.06, 0.08), cat * 0.9)
    belly = _ellipse_mask(res, 0.48, 0.48, 0.08, 0.14)
    img = _blend(img, _rgb(res, 0.70, 0.35, 0.50), belly * 0.4)
    mouse = _ellipse_mask(res, 0.30, 0.78, 0.03, 0.02)
    mouse_tail = _rect_mask(res, 0.24, 0.775, 0.30, 0.785)
    img = _blend(img, _rgb(res, 0.80, 0.80, 0.75), (mouse + mouse_tail) * 0.8)
    return np.clip(img, 0, 1)


# ═══════════════════════════════════════════════════════════════
# Algorithm core
# ═══════════════════════════════════════════════════════════════

def cylinder_normal(p):
    n = np.zeros_like(p)
    n[..., 0] = p[..., 0]; n[..., 1] = p[..., 1]
    length = np.sqrt(n[..., 0]**2 + n[..., 1]**2 + 1e-12)
    n[..., 0] /= length; n[..., 1] /= length
    return n

def reflect_vec(d, n):
    dot = np.sum(d * n, axis=-1, keepdims=True)
    return d - 2.0 * dot * n

def ray_cyl_hit(origin, direction, R):
    ox, oy = origin[..., 0], origin[..., 1]
    dx, dy = direction[..., 0], direction[..., 1]
    a = dx**2 + dy**2
    b = 2*(ox*dx + oy*dy)
    c = ox**2 + oy**2 - R**2
    disc = b**2 - 4*a*c
    hit = disc >= 0
    sd = np.sqrt(np.maximum(disc, 0))
    t1 = (-b - sd) / (2*a + 1e-12)
    t2 = (-b + sd) / (2*a + 1e-12)
    t = np.where(t1 > 1e-6, t1, t2)
    hit = hit & (t > 1e-6)
    return t, hit

def build_reflection_map(res, cup_R=1.0, cup_H=2.0, saucer_R=3.0):
    eye = np.array([0.0, -4*saucer_R, 3*cup_H])
    u = np.linspace(-1, 1, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    theta = np.pi * uu
    z = 0.5 * (vv + 1) * cup_H
    cup_pts = np.stack([cup_R*np.cos(theta), cup_R*np.sin(theta), z], -1)
    d = cup_pts - eye[None, None, :]
    d /= (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)
    eye_bc = np.broadcast_to(eye, d.shape)
    t, hit = ray_cyl_hit(eye_bc, d, cup_R)
    hp = eye[None, None, :] + t[..., None] * d
    z_ok = (hp[..., 2] >= 0) & (hp[..., 2] <= cup_H)
    hit = hit & z_ok
    n = cylinder_normal(hp)
    rd = reflect_vec(d, n)
    dz = rd[..., 2]
    t2 = -hp[..., 2] / (dz + 1e-12)
    hit2 = t2 > 1e-6
    sp = hp + t2[..., None] * rd
    su = (sp[..., 0] / saucer_R + 1) * 0.5
    sv = (sp[..., 1] / saucer_R + 1) * 0.5
    r2 = sp[..., 0]**2 + sp[..., 1]**2
    valid = hit & hit2 & (r2 <= saucer_R**2) & (r2 >= cup_R**2)
    uv = np.stack([su, sv], -1)
    uv[~valid] = -1
    return uv, valid

def build_direct_map(res, saucer_R=3.0, cup_R=1.0):
    u = np.linspace(0, 1, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    x = (uu-0.5)*2*saucer_R; y = (vv-0.5)*2*saucer_R
    r2 = x**2 + y**2
    valid = (r2 <= saucer_R**2) & (r2 >= cup_R**2)
    return np.stack([uu, vv], -1), valid

def bilinear_warp(src, uv, valid):
    if src.ndim == 2: src = src[..., None]
    sh, sw, C = src.shape
    h, w = uv.shape[:2]
    out = np.zeros((h, w, C))
    u = np.clip(uv[..., 0], 0, 1) * (sw-1)
    v = np.clip(uv[..., 1], 0, 1) * (sh-1)
    u0 = np.clip(np.floor(u).astype(int), 0, sw-1)
    v0 = np.clip(np.floor(v).astype(int), 0, sh-1)
    u1 = np.clip(u0+1, 0, sw-1); v1 = np.clip(v0+1, 0, sh-1)
    fu = u-u0; fv = v-v0
    for c in range(C):
        out[..., c] = (src[v0,u0,c]*(1-fu)*(1-fv) + src[v0,u1,c]*fu*(1-fv) +
                       src[v1,u0,c]*(1-fu)*fv + src[v1,u1,c]*fu*fv) * valid
    return out if C > 1 else out[..., 0]

def heightfield_normals(hf):
    dx = np.zeros_like(hf); dy = np.zeros_like(hf)
    dx[:, 1:-1] = (hf[:, 2:] - hf[:, :-2]) / 2
    dy[1:-1, :] = (hf[2:, :] - hf[:-2, :]) / 2
    nz = np.ones_like(hf)
    length = np.sqrt(dx**2 + dy**2 + nz**2)
    return np.stack([-dx/length, -dy/length, nz/length], -1)

def optimize_texture_adam(
    direct_target, reflected_target, direct_uv, direct_valid,
    reflected_uv, reflected_valid, mask, res, n_iter=300, lr=0.06
):
    if direct_target.ndim == 2:
        direct_target = np.stack([direct_target]*3, -1)
    if reflected_target.ndim == 2:
        reflected_target = np.stack([reflected_target]*3, -1)
    tex = np.ones((res, res, 3)) * 0.5
    m = np.zeros_like(tex); va = np.zeros_like(tex)
    losses = []
    dv_j, dv_i = np.where(direct_valid)
    dv_vi = np.clip((direct_uv[dv_j, dv_i, 1] * (res-1)).astype(int), 0, res-1)
    dv_ui = np.clip((direct_uv[dv_j, dv_i, 0] * (res-1)).astype(int), 0, res-1)
    rv_j, rv_i = np.where(reflected_valid)
    rv_vi = np.clip((reflected_uv[rv_j, rv_i, 1] * (res-1)).astype(int), 0, res-1)
    rv_ui = np.clip((reflected_uv[rv_j, rv_i, 0] * (res-1)).astype(int), 0, res-1)
    for it in range(n_iter):
        dr = bilinear_warp(tex, direct_uv, direct_valid)
        rr = bilinear_warp(tex, reflected_uv, reflected_valid)
        if dr.ndim == 2: dr = np.stack([dr]*3, -1)
        if rr.ndim == 2: rr = np.stack([rr]*3, -1)
        diff_d = dr - direct_target
        diff_r = rr - reflected_target
        loss = np.mean(diff_d**2) + np.mean(diff_r**2)
        losses.append(loss)
        if it % 100 == 0:
            print(f"    tex iter {it:4d}  loss={loss:.6f}")
        grad = np.zeros_like(tex)
        cnt = np.ones((res, res, 1)) * 1e-8
        np.add.at(grad, (dv_vi, dv_ui), 2.0 * diff_d[dv_j, dv_i])
        np.add.at(cnt,  (dv_vi, dv_ui), np.ones((len(dv_j), 1)))
        np.add.at(grad, (rv_vi, rv_ui), 2.0 * diff_r[rv_j, rv_i])
        np.add.at(cnt,  (rv_vi, rv_ui), np.ones((len(rv_j), 1)))
        grad /= cnt
        grad *= mask[..., None]
        m = 0.9*m + 0.1*grad
        va = 0.999*va + 0.001*grad**2
        mh = m / (1 - 0.9**(it+1))
        vh = va / (1 - 0.999**(it+1))
        tex -= lr * mh / (np.sqrt(vh) + 1e-8)
        tex = np.clip(tex, 0, 1)
    return tex, losses


# ═══════════════════════════════════════════════════════════════
# Main: Run blue cat scene with all fixes
# ═══════════════════════════════════════════════════════════════

def main():
    res = 256
    out_dir = "blue_cat_fixed_output"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Blue Cat Scene (Fixed) — Luycho x oyow")
    print("=" * 60)

    from skimage.transform import resize as sk_resize
    direct_full = make_scene2_direct(512)
    reflected_full = make_scene2_reflected(512)

    direct_img = sk_resize(direct_full, (res, res, 3), anti_aliasing=True)
    reflected_img = sk_resize(reflected_full, (res, res, 3), anti_aliasing=True)

    uu, vv, r, theta = _coord_grid(res)
    base_shape = 0.05 * np.sin(2 * np.pi * 14 * r)

    cup_R, cup_H, sauc_R = 1.0, 2.0, 3.0
    saucer_color = (0.22, 0.42, 0.78)

    Image.fromarray((np.clip(direct_img, 0, 1)*255).astype(np.uint8)).save(
        os.path.join(out_dir, "input_direct.png"))
    Image.fromarray((np.clip(reflected_img, 0, 1)*255).astype(np.uint8)).save(
        os.path.join(out_dir, "input_reflected.png"))

    # FIX 1: Use np.ptp() instead of .ptp() for NumPy 2.0 compatibility
    bs_norm = (base_shape - base_shape.min()) / (np.ptp(base_shape) + 1e-8)
    Image.fromarray((bs_norm * 255).astype(np.uint8)).save(
        os.path.join(out_dir, "input_base_shape.png"))

    mask = ((r <= 1.0) & (r >= cup_R / sauc_R)).astype(float)

    print("  [1/4] Refining saucer geometry...")
    direct_gray = np.mean(direct_img, axis=-1)
    disp = np.zeros((res, res))
    for gi in range(80):
        hf = base_shape + disp * mask
        nrm = heightfield_normals(hf)
        shading = nrm[..., 2]
        err = (shading - direct_gray) * mask
        adj = gaussian_filter(-0.003 * err, sigma=2.0)
        disp += adj * mask
        disp = np.clip(disp, -0.25, 0.25)
    med = np.median(np.abs(disp[disp != 0])) if np.any(disp != 0) else 1e-8
    disp = np.tanh(2.0 * disp / (med + 1e-8)) * med
    final_hf = base_shape + disp * mask

    print("  [2/4] Building reflection mapping...")
    ref_uv, ref_valid = build_reflection_map(res, cup_R, cup_H, sauc_R)
    dir_uv, dir_valid = build_direct_map(res, sauc_R, cup_R)

    print("  [3/4] Optimizing saucer texture (300 iterations)...")
    texture, losses = optimize_texture_adam(
        direct_img, reflected_img,
        dir_uv, dir_valid, ref_uv, ref_valid,
        mask, res, n_iter=300, lr=0.06
    )

    print("  [4/4] Rendering results...")
    direct_render = bilinear_warp(texture, dir_uv, dir_valid)
    reflected_render = bilinear_warp(texture, ref_uv, ref_valid)
    if direct_render.ndim == 2: direct_render = np.stack([direct_render]*3, -1)
    if reflected_render.ndim == 2: reflected_render = np.stack([reflected_render]*3, -1)

    Image.fromarray((np.clip(texture, 0, 1)*255).astype(np.uint8)).save(
        os.path.join(out_dir, "saucer_texture.png"))

    fig = plt.figure(figsize=(24, 14))
    sc = saucer_color
    fig.patch.set_facecolor((0.05, 0.05, 0.08))

    gs = fig.add_gridspec(2, 5, hspace=0.30, wspace=0.20)

    def _show(pos, img, title, cmap=None):
        ax = fig.add_subplot(pos)
        ax.imshow(np.clip(img, 0, 1) if cmap is None else img, cmap=cmap)
        ax.set_title(title, fontsize=10, color="white", fontweight="bold")
        ax.axis("off")
        return ax

    _show(gs[0, 0], direct_img, "Direct View Target")
    _show(gs[0, 1], reflected_img, "Reflected View Target")
    _show(gs[0, 2], bs_norm, "Base Shape", cmap="terrain")

    _show(gs[1, 0], direct_render, "Rendered Direct View")
    _show(gs[1, 1], reflected_render, "Rendered Reflected View")
    _show(gs[1, 2], texture, "Optimized Texture")
    _show(gs[1, 3], final_hf, "Height Field", cmap="terrain")

    # FIX 2: Clamp color to [0,1] range for matplotlib
    ax_loss = fig.add_subplot(gs[1, 4])
    loss_color = tuple(min(c * 1.5, 1.0) for c in sc)
    ax_loss.plot(losses, color=loss_color, linewidth=1.2)
    ax_loss.set_facecolor((0.08, 0.08, 0.12))
    ax_loss.set_title("Loss Curve", fontsize=10, color="white")
    ax_loss.set_xlabel("Iteration", color="white")
    ax_loss.set_ylabel("MSE Loss", color="white")
    ax_loss.tick_params(colors="white")
    ax_loss.grid(True, alpha=0.15, color="white")

    ax_circ = fig.add_subplot(gs[0, 3:5])
    circ_tex = texture * mask[..., None]
    saucer_bg = np.ones((res, res, 3)) * np.array(sc)
    circ_display = np.where(mask[..., None] > 0.5, circ_tex, saucer_bg)
    ax_circ.imshow(np.clip(circ_display, 0, 1))
    ax_circ.set_title("Texture on Circular Saucer", fontsize=10, color="white")
    ax_circ.axis("off")

    plt.suptitle(
        "Blue Cat Scene (Fixed) — Computational Mirror Cup & Saucer Art",
        fontsize=14, color="white", fontweight="bold", y=0.98
    )
    fig_path = os.path.join(out_dir, "blue_cat_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"\nDone! Output saved to: {out_dir}/")


if __name__ == "__main__":
    main()
