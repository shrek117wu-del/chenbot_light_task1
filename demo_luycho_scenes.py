#!/usr/bin/env python3
"""
Demo: Reproduce the 4 Luycho × oyow mirror cup & saucer art scenes
from the provided photographs using the Computational Mirror Cup and Saucer Art
algorithm (Wu et al., ACM TOG 2022, doi:10.1145/3517120).

Scene 1 (Image 4): Green saucer — Jungle dinosaur (direct) → Running person (reflected)
Scene 2 (Image 3): Blue saucer — Black cat on tiles (direct) → Sitting cat (reflected)
Scene 3 (Image 2): Green saucer — Castle in forest (direct) → Running person (reflected)
Scene 4 (Image 1): Dark blue saucer — Ocean turtle scene (direct) → Yellow turtle (reflected)

All input images are programmatically generated to match the photographs,
then fed through the full optimization pipeline.

Usage:
    python demo_luycho_scenes.py                    # Run all 4 scenes
    python demo_luycho_scenes.py --scene 1          # Run scene 1 only
    python demo_luycho_scenes.py --scene 2 --viewer # Run scene 2 + 3D viewer
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ══════════════════════════════════════════════════════════════════════════
# PART A — Programmatic generation of the 4 Luycho scene input images
# ══════════════════════════════════════════════════════════════════════════

def _coord_grid(res):
    """Return (uu, vv) in [-1,1]^2 and r, theta."""
    u = np.linspace(-1, 1, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    r = np.sqrt(uu**2 + vv**2)
    theta = np.arctan2(vv, uu)
    return uu, vv, r, theta


def _annular_mask(res, outer=1.0, inner=0.33):
    _, _, r, _ = _coord_grid(res)
    return ((r <= outer) & (r >= inner)).astype(float)


def _rgb(res, r, g, b):
    """Solid color image."""
    img = np.zeros((res, res, 3))
    img[..., 0] = r; img[..., 1] = g; img[..., 2] = b
    return img


def _blend(bg, fg, alpha):
    """Alpha blend fg over bg."""
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    return bg * (1 - a) + fg * a


def _ellipse_mask(res, cx, cy, rx, ry):
    """Ellipse mask centered at (cx, cy) with radii (rx, ry), coords in [0,1]."""
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
    """Triangle defined by 3 vertices in [0,1]^2."""
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


# ────────────────────────────────────────────────────────────────────
# Scene 1: Green saucer — Running person (reflected in cup)
#   Direct view: Distorted jungle scene with dinosaur/creature + foliage
#   Reflected view: Running human figure (pink/gold silhouette)
# ────────────────────────────────────────────────────────────────────

def make_scene1_reflected(res=512):
    """Running person silhouette — what appears in the mirror cup."""
    bg = _rgb(res, 0.18, 0.65, 0.45)  # green background

    # Running person body (simplified stick figure silhouette)
    # Torso
    torso = _ellipse_mask(res, 0.50, 0.40, 0.06, 0.12)
    # Head
    head = _circle_mask_abs(res, 0.50, 0.26, 0.045)
    # Left arm (raised)
    arm_l = _ellipse_mask(res, 0.40, 0.34, 0.08, 0.025)
    # Right arm (back)
    arm_r = _ellipse_mask(res, 0.58, 0.38, 0.07, 0.025)
    # Left leg (forward stride)
    leg_l = _ellipse_mask(res, 0.43, 0.58, 0.025, 0.12)
    # Right leg (back stride)
    leg_r = _ellipse_mask(res, 0.56, 0.56, 0.025, 0.10)

    person = np.clip(torso + head + arm_l + arm_r + leg_l + leg_r, 0, 1)
    person_color = _rgb(res, 0.95, 0.75, 0.55)  # golden/skin tone

    # Yellow ball (in the air)
    ball = _circle_mask_abs(res, 0.42, 0.18, 0.03)
    ball_color = _rgb(res, 0.95, 0.90, 0.30)

    img = bg.copy()
    img = _blend(img, person_color, person)
    img = _blend(img, ball_color, ball)

    # Pink accent on body
    pink_accent = _ellipse_mask(res, 0.50, 0.42, 0.04, 0.08)
    img = _blend(img, _rgb(res, 0.90, 0.50, 0.60), pink_accent * 0.5)

    return np.clip(img, 0, 1)


def make_scene1_direct(res=512):
    """Distorted jungle/dinosaur scene on the green saucer — direct view."""
    uu, vv, r, theta = _coord_grid(res)

    # Green background with wave distortion
    bg = _rgb(res, 0.15, 0.60, 0.42)

    # Foliage leaves (multiple green patches)
    leaf_colors = [
        (0.10, 0.50, 0.30), (0.20, 0.70, 0.40), (0.08, 0.55, 0.35),
        (0.25, 0.65, 0.38), (0.12, 0.58, 0.32),
    ]
    img = bg.copy()

    np.random.seed(42)
    for _ in range(12):
        cx = np.random.uniform(0.15, 0.85)
        cy = np.random.uniform(0.15, 0.85)
        rx = np.random.uniform(0.04, 0.12)
        ry = np.random.uniform(0.06, 0.15)
        leaf = _ellipse_mask(res, cx, cy, rx, ry)
        col = leaf_colors[np.random.randint(len(leaf_colors))]
        img = _blend(img, _rgb(res, *col), leaf * 0.7)

    # Dinosaur/creature (pink blob, distorted)
    dino_body = _ellipse_mask(res, 0.45, 0.55, 0.10, 0.15)
    dino_head = _circle_mask_abs(res, 0.45, 0.38, 0.05)
    dino_tail = _ellipse_mask(res, 0.55, 0.65, 0.12, 0.03)
    dino = np.clip(dino_body + dino_head + dino_tail, 0, 1)
    dino_color = _rgb(res, 0.85, 0.45, 0.55)
    img = _blend(img, dino_color, dino * 0.8)

    # Castle/dome structure
    dome = _ellipse_mask(res, 0.50, 0.70, 0.12, 0.08)
    tower1 = _rect_mask(res, 0.42, 0.60, 0.46, 0.72)
    tower2 = _rect_mask(res, 0.54, 0.62, 0.58, 0.72)
    castle = np.clip(dome + tower1 + tower2, 0, 1)
    img = _blend(img, _rgb(res, 0.90, 0.85, 0.80), castle * 0.6)

    # Ground / path
    ground = _rect_mask(res, 0.20, 0.78, 0.80, 0.90)
    img = _blend(img, _rgb(res, 0.25, 0.55, 0.35), ground * 0.5)

    return np.clip(img, 0, 1)


# ────────────────────────────────────────────────────────────────────
# Scene 2: Blue saucer — Black cat (reflected in cup)
#   Direct view: Distorted cat on tile floor
#   Reflected view: Black cat sitting upright
# ────────────────────────────────────────────────────────────────────

def make_scene2_reflected(res=512):
    """Sitting black cat — reflected in the silver mirror cup."""
    bg = _rgb(res, 0.25, 0.45, 0.80)  # blue background

    # Cat body (sitting pose)
    body = _ellipse_mask(res, 0.50, 0.55, 0.10, 0.16)
    # Head
    head = _circle_mask_abs(res, 0.50, 0.34, 0.07)
    # Ears (triangles)
    ear_l = _triangle_mask(res, 0.43, 0.34, 0.40, 0.22, 0.46, 0.28)
    ear_r = _triangle_mask(res, 0.57, 0.34, 0.54, 0.28, 0.60, 0.22)
    # Tail (curved)
    u = np.linspace(0, 1, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    tail_x = 0.62 + 0.08 * np.sin(3 * np.pi * (vv - 0.50))
    tail = ((np.abs(uu - tail_x) < 0.015) & (vv > 0.50) & (vv < 0.72)).astype(float)
    tail = gaussian_filter(tail, sigma=2)

    cat = np.clip(body + head + ear_l + ear_r + tail, 0, 1)
    cat_color = _rgb(res, 0.08, 0.08, 0.10)  # black
    img = bg.copy()
    img = _blend(img, cat_color, cat)

    # Eyes (yellow)
    eye_l = _circle_mask_abs(res, 0.47, 0.32, 0.012)
    eye_r = _circle_mask_abs(res, 0.53, 0.32, 0.012)
    img = _blend(img, _rgb(res, 0.95, 0.90, 0.20), eye_l + eye_r)

    # Pink belly accent
    belly = _ellipse_mask(res, 0.50, 0.52, 0.06, 0.08)
    img = _blend(img, _rgb(res, 0.75, 0.40, 0.55), belly * 0.5)

    # Whiskers (thin lines)
    for dy in [-0.01, 0, 0.01]:
        whisker = _rect_mask(res, 0.32, 0.345 + dy, 0.44, 0.348 + dy)
        whisker += _rect_mask(res, 0.56, 0.345 + dy, 0.68, 0.348 + dy)
        img = _blend(img, _rgb(res, 0.9, 0.9, 0.9), np.clip(whisker, 0, 1) * 0.7)

    # Floor tile pattern
    tile_y = vv > 0.72
    tile_check = ((np.floor(uu * 8).astype(int) + np.floor(vv * 8).astype(int)) % 2)
    tile_pattern = (tile_check * tile_y).astype(float)
    img = _blend(img, _rgb(res, 0.30, 0.55, 0.75), tile_pattern * 0.4)

    return np.clip(img, 0, 1)


def make_scene2_direct(res=512):
    """Distorted black cat on blue tile floor — direct view on saucer."""
    uu, vv, r, theta = _coord_grid(res)
    bg = _rgb(res, 0.22, 0.42, 0.78)

    img = bg.copy()

    # Tile floor (perspective distorted checkerboard)
    u01 = np.linspace(0, 1, res)
    uu01, vv01 = np.meshgrid(u01, u01, indexing="xy")
    check = ((np.floor(uu01 * 10).astype(int) + np.floor(vv01 * 10).astype(int)) % 2)
    img = _blend(img, _rgb(res, 0.18, 0.38, 0.70), check.astype(float) * 0.3)

    # Distorted cat silhouette (elongated, anamorphic)
    cat_body = _ellipse_mask(res, 0.48, 0.50, 0.15, 0.28)  # very tall
    cat_head = _ellipse_mask(res, 0.48, 0.20, 0.08, 0.06)
    ear_l = _triangle_mask(res, 0.42, 0.20, 0.38, 0.10, 0.44, 0.15)
    ear_r = _triangle_mask(res, 0.54, 0.20, 0.52, 0.15, 0.58, 0.10)
    tail = _ellipse_mask(res, 0.62, 0.60, 0.03, 0.18)

    cat = np.clip(cat_body + cat_head + ear_l + ear_r + tail, 0, 1)
    img = _blend(img, _rgb(res, 0.06, 0.06, 0.08), cat * 0.9)

    # Pink belly (distorted)
    belly = _ellipse_mask(res, 0.48, 0.48, 0.08, 0.14)
    img = _blend(img, _rgb(res, 0.70, 0.35, 0.50), belly * 0.4)

    # Mouse toy
    mouse = _ellipse_mask(res, 0.30, 0.78, 0.03, 0.02)
    mouse_tail = _rect_mask(res, 0.24, 0.775, 0.30, 0.785)
    img = _blend(img, _rgb(res, 0.80, 0.80, 0.75), (mouse + mouse_tail) * 0.8)

    return np.clip(img, 0, 1)


# ────────────────────────────────────────────────────────────────────
# Scene 3: Green saucer variant (same theme as scene 1, different angle)
#   Direct view: Castle/greenhouse in jungle
#   Reflected view: Running person
# ────────────────────────────────────────────────────────────────────

def make_scene3_reflected(res=512):
    """Running person — same theme as scene 1 but slightly different."""
    img = make_scene1_reflected(res)
    # Slight color variation
    img[..., 1] = np.clip(img[..., 1] * 1.05, 0, 1)
    return img


def make_scene3_direct(res=512):
    """Greenhouse/castle in jungle — direct view."""
    uu, vv, r, theta = _coord_grid(res)
    bg = _rgb(res, 0.12, 0.55, 0.38)
    img = bg.copy()

    # Greenhouse dome (larger, centered)
    dome = _ellipse_mask(res, 0.50, 0.55, 0.18, 0.14)
    dome_top = _triangle_mask(res, 0.32, 0.55, 0.50, 0.30, 0.68, 0.55)
    greenhouse = np.clip(dome + dome_top, 0, 1)
    img = _blend(img, _rgb(res, 0.85, 0.90, 0.85), greenhouse * 0.5)

    # Grid lines on greenhouse
    u01 = np.linspace(0, 1, res)
    uu01, vv01 = np.meshgrid(u01, u01, indexing="xy")
    grid_h = ((np.floor(vv01 * 20).astype(int) % 2) == 0).astype(float)
    grid_v = ((np.floor(uu01 * 20).astype(int) % 2) == 0).astype(float)
    grid = np.clip(grid_h + grid_v, 0, 1)
    img = _blend(img, _rgb(res, 0.80, 0.85, 0.80), grid * greenhouse * 0.15)

    # Dinosaur inside greenhouse
    dino = _ellipse_mask(res, 0.50, 0.55, 0.07, 0.10)
    img = _blend(img, _rgb(res, 0.85, 0.45, 0.55), dino * 0.7)

    # Foliage around
    np.random.seed(101)
    for _ in range(15):
        cx = np.random.uniform(0.1, 0.9)
        cy = np.random.uniform(0.1, 0.9)
        rx = np.random.uniform(0.03, 0.10)
        ry = np.random.uniform(0.05, 0.12)
        leaf = _ellipse_mask(res, cx, cy, rx, ry)
        g = np.random.uniform(0.45, 0.65)
        img = _blend(img, _rgb(res, 0.10, g, 0.30), leaf * 0.6)

    # Ground flowers (small dots)
    np.random.seed(202)
    for _ in range(8):
        fx = np.random.uniform(0.2, 0.8)
        fy = np.random.uniform(0.75, 0.90)
        flower = _circle_mask_abs(res, fx, fy, 0.012)
        img = _blend(img, _rgb(res, 0.95, 0.85, 0.90), flower)

    return np.clip(img, 0, 1)


# ────────────────────────────────────────────────────────────────────
# Scene 4: Dark blue saucer — Yellow turtle (reflected in cup)
#   Direct view: Distorted underwater scene with turtle
#   Reflected view: Yellow sea turtle swimming
# ────────────────────────────────────────────────────────────────────

def make_scene4_reflected(res=512):
    """Yellow sea turtle — reflected in the mirror cup."""
    bg = _rgb(res, 0.15, 0.22, 0.55)  # dark blue ocean

    # Turtle shell (main body)
    shell = _ellipse_mask(res, 0.50, 0.45, 0.14, 0.10)
    shell_color = _rgb(res, 0.90, 0.80, 0.20)  # yellow

    # Head
    head = _ellipse_mask(res, 0.50, 0.30, 0.05, 0.06)

    # Flippers
    flip_fl = _ellipse_mask(res, 0.36, 0.38, 0.07, 0.03)  # front left
    flip_fr = _ellipse_mask(res, 0.64, 0.38, 0.07, 0.03)  # front right
    flip_bl = _ellipse_mask(res, 0.40, 0.54, 0.05, 0.025)  # back left
    flip_br = _ellipse_mask(res, 0.60, 0.54, 0.05, 0.025)  # back right

    turtle = np.clip(shell + head + flip_fl + flip_fr + flip_bl + flip_br, 0, 1)

    img = bg.copy()
    img = _blend(img, shell_color, turtle)

    # Shell pattern (hexagonal markings)
    u01 = np.linspace(0, 1, res)
    uu01, vv01 = np.meshgrid(u01, u01, indexing="xy")
    hex_pattern = (np.sin(30 * uu01) * np.sin(30 * vv01) > 0.3).astype(float)
    shell_detail = shell * hex_pattern
    img = _blend(img, _rgb(res, 0.70, 0.60, 0.15), shell_detail * 0.3)

    # Eyes
    eye = _circle_mask_abs(res, 0.50, 0.28, 0.010)
    img = _blend(img, _rgb(res, 0.1, 0.1, 0.1), eye)

    # Water bubbles
    np.random.seed(77)
    for _ in range(6):
        bx = np.random.uniform(0.2, 0.8)
        by = np.random.uniform(0.15, 0.75)
        bubble = _circle_mask_abs(res, bx, by, 0.015)
        img = _blend(img, _rgb(res, 0.40, 0.55, 0.80), bubble * 0.4)

    # Green seaweed/coral at bottom
    for sx in np.linspace(0.15, 0.85, 6):
        seaweed = _ellipse_mask(res, sx, 0.80, 0.02, 0.10)
        img = _blend(img, _rgb(res, 0.15, 0.50, 0.35), seaweed * 0.7)

    return np.clip(img, 0, 1)


def make_scene4_direct(res=512):
    """Distorted underwater turtle scene — direct view on dark blue saucer."""
    uu, vv, r, theta = _coord_grid(res)
    bg = _rgb(res, 0.12, 0.18, 0.50)
    img = bg.copy()

    # Ocean gradient (darker at edges)
    depth = np.clip(1.0 - r * 0.5, 0, 1)
    img *= depth[..., None]
    img = np.clip(img + 0.08, 0, 1)

    # Distorted turtle (anamorphic — stretched radially)
    shell = _ellipse_mask(res, 0.48, 0.45, 0.18, 0.25)
    head = _ellipse_mask(res, 0.48, 0.18, 0.06, 0.08)
    flippers = (
        _ellipse_mask(res, 0.30, 0.35, 0.08, 0.04) +
        _ellipse_mask(res, 0.66, 0.35, 0.08, 0.04) +
        _ellipse_mask(res, 0.35, 0.65, 0.06, 0.035) +
        _ellipse_mask(res, 0.61, 0.65, 0.06, 0.035)
    )
    turtle = np.clip(shell + head + flippers, 0, 1)
    img = _blend(img, _rgb(res, 0.82, 0.72, 0.18), turtle * 0.8)

    # Shell markings
    u01 = np.linspace(0, 1, res)
    uu01, vv01 = np.meshgrid(u01, u01, indexing="xy")
    marks = (np.sin(25 * uu01) * np.sin(25 * vv01) > 0.4).astype(float)
    img = _blend(img, _rgb(res, 0.65, 0.55, 0.12), marks * shell * 0.25)

    # Seaweed/coral
    np.random.seed(88)
    for _ in range(10):
        sx = np.random.uniform(0.1, 0.9)
        sy = np.random.uniform(0.70, 0.95)
        sw = _ellipse_mask(res, sx, sy, 0.025, 0.08)
        g = np.random.uniform(0.30, 0.55)
        img = _blend(img, _rgb(res, 0.10, g, 0.30), sw * 0.6)

    # Pink coral accent
    coral = _ellipse_mask(res, 0.25, 0.80, 0.06, 0.05)
    img = _blend(img, _rgb(res, 0.75, 0.35, 0.50), coral * 0.5)

    return np.clip(img, 0, 1)


# ══════════════════════════════════════════════════════════════════════════
# PART B — Saucer base shapes (wave/ripple pattern matching photographs)
# ══════════════════════════════════════════════════════════════════════════

def base_wave_concentric(res=512, n_rings=12, amplitude=0.06):
    """Concentric wave/ripple base shape matching the Luycho saucers."""
    _, _, r, _ = _coord_grid(res)
    return amplitude * np.sin(2 * np.pi * n_rings * r)


# ══════════════════════════════════════════════════════════════════════════
# PART C — Algorithm Core (self-contained)
# ══════════════════════════════════════════════════════════════════════════

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

def build_reflection_map(res, cup_R=1.0, cup_H=2.0, saucer_R=3.0, eye=None):
    if eye is None:
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
    reflected_uv, reflected_valid, mask, res, n_iter=500, lr=0.06
):
    """Optimise saucer texture via Adam to match both views simultaneously."""
    if direct_target.ndim == 2:
        direct_target = np.stack([direct_target]*3, -1)
    if reflected_target.ndim == 2:
        reflected_target = np.stack([reflected_target]*3, -1)

    # Initialize texture with weighted average hint
    tex = np.ones((res, res, 3)) * 0.5
    m = np.zeros_like(tex); va = np.zeros_like(tex)
    losses = []

    # Pre-compute valid index arrays for speed
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
            print(f"      tex iter {it:4d}  loss={loss:.6f}")

        # Vectorized scatter
        grad = np.zeros_like(tex)
        cnt = np.ones((res, res, 1)) * 1e-8

        # Direct gradient scatter
        np.add.at(grad, (dv_vi, dv_ui), 2.0 * diff_d[dv_j, dv_i])
        np.add.at(cnt,  (dv_vi, dv_ui), np.ones((len(dv_j), 1)))

        # Reflected gradient scatter
        np.add.at(grad, (rv_vi, rv_ui), 2.0 * diff_r[rv_j, rv_i])
        np.add.at(cnt,  (rv_vi, rv_ui), np.ones((len(rv_j), 1)))

        grad /= cnt
        grad *= mask[..., None]

        # Smoothness
        lap = (np.roll(tex,1,0)+np.roll(tex,-1,0)+np.roll(tex,1,1)+np.roll(tex,-1,1)-4*tex)
        grad += 0.005 * (-2 * lap) * mask[..., None]

        # Adam update
        m = 0.9*m + 0.1*grad
        va = 0.999*va + 0.001*grad**2
        mh = m / (1 - 0.9**(it+1))
        vh = va / (1 - 0.999**(it+1))
        tex -= lr * mh / (np.sqrt(vh) + 1e-8)
        tex = np.clip(tex, 0, 1)

    return tex, losses

def refine_geometry(base_shape, direct_target, mask, res, n_iter=80):
    """Refine saucer geometry to encode direct-view image in surface normals."""
    if direct_target.ndim == 3:
        direct_target = np.mean(direct_target, axis=-1)
    disp = np.zeros((res, res))
    for gi in range(n_iter):
        hf = base_shape + disp * mask
        nrm = heightfield_normals(hf)
        shading = nrm[..., 2]
        err = (shading - direct_target) * mask
        adj = gaussian_filter(-0.003 * err, sigma=2.0)
        disp += adj * mask
        disp = np.clip(disp, -0.25, 0.25)
    # Black-white enhancement
    med = np.median(np.abs(disp[disp != 0])) if np.any(disp != 0) else 1e-8
    disp = np.tanh(2.0 * disp / (med + 1e-8)) * med
    return base_shape + disp * mask

def export_obj(verts, faces, uvs, tex_fn, obj_path, mtl_path):
    mtl_name = os.path.basename(mtl_path).replace(".mtl", "")
    with open(mtl_path, "w") as f:
        f.write(f"newmtl {mtl_name}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd {tex_fn}\n")
    with open(obj_path, "w") as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\nusemtl {mtl_name}\n")
        for v in verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs: f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for fc in faces:
            f.write(f"f {fc[0]+1}/{fc[0]+1} {fc[1]+1}/{fc[1]+1} {fc[2]+1}/{fc[2]+1}\n")


# ══════════════════════════════════════════════════════════════════════════
# PART D — Scene Definitions
# ══════════════════════════════════════════════════════════��═══════════════

def get_luycho_scenes(res=512):
    return [
        {
            "name": "scene1_green_jungle_runner",
            "photo": "Image 4 (green saucer, gold cup)",
            "description": (
                "Green wave saucer — Direct: Jungle scene with pink dinosaur "
                "& castle among foliage. Reflected: Running golden person."
            ),
            "direct_image": make_scene1_direct(res),
            "reflected_image": make_scene1_reflected(res),
            "base_shape": base_wave_concentric(res, n_rings=12, amplitude=0.06),
            "saucer_color": (0.15, 0.60, 0.42),
            "cup_radius": 1.0, "cup_height": 2.0, "saucer_radius": 3.0,
        },
        {
            "name": "scene2_blue_black_cat",
            "photo": "Image 3 (blue saucer, silver cup)",
            "description": (
                "Blue wave saucer — Direct: Distorted black cat on tile floor "
                "with mouse toy. Reflected: Sitting black cat with yellow eyes."
            ),
            "direct_image": make_scene2_direct(res),
            "reflected_image": make_scene2_reflected(res),
            "base_shape": base_wave_concentric(res, n_rings=14, amplitude=0.05),
            "saucer_color": (0.22, 0.42, 0.78),
            "cup_radius": 1.0, "cup_height": 2.0, "saucer_radius": 3.0,
        },
        {
            "name": "scene3_green_castle_runner",
            "photo": "Image 2 (green saucer, gold cup, front+back view)",
            "description": (
                "Green wave saucer — Direct: Greenhouse/castle dome in forest "
                "with pink dinosaur inside. Reflected: Running golden person."
            ),
            "direct_image": make_scene3_direct(res),
            "reflected_image": make_scene3_reflected(res),
            "base_shape": base_wave_concentric(res, n_rings=11, amplitude=0.07),
            "saucer_color": (0.12, 0.55, 0.38),
            "cup_radius": 1.0, "cup_height": 2.0, "saucer_radius": 3.0,
        },
        {
            "name": "scene4_darkblue_sea_turtle",
            "photo": "Image 1 (dark blue saucer, gold cup)",
            "description": (
                "Dark blue wave saucer — Direct: Underwater scene with sea turtle, "
                "coral and seaweed. Reflected: Yellow sea turtle swimming."
            ),
            "direct_image": make_scene4_direct(res),
            "reflected_image": make_scene4_reflected(res),
            "base_shape": base_wave_concentric(res, n_rings=13, amplitude=0.055),
            "saucer_color": (0.12, 0.18, 0.50),
            "cup_radius": 1.0, "cup_height": 2.0, "saucer_radius": 3.0,
        },
    ]


# ══════════════════════════════════════════════════════════════════════════
# PART E — Full Pipeline
# ══════════════════════════════════════════════════════════════════════════

def run_scene(scene, output_root="luycho_demo_outputs", work_res=256, tex_iter=500):
    name = scene["name"]
    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*72}")
    print(f"  🎨 {name}")
    print(f"  📸 Photo: {scene['photo']}")
    print(f"  📝 {scene['description']}")
    print(f"{'═'*72}")

    res = work_res
    cup_R = scene["cup_radius"]
    cup_H = scene["cup_height"]
    sauc_R = scene["saucer_radius"]

    # Resize inputs
    from skimage.transform import resize as sk_resize
    direct_img = sk_resize(scene["direct_image"], (res, res, 3), anti_aliasing=True)
    reflected_img = sk_resize(scene["reflected_image"], (res, res, 3), anti_aliasing=True)
    base_shape = sk_resize(scene["base_shape"], (res, res), anti_aliasing=True)

    # Save inputs
    for arr, fname in [
        (direct_img, "input_direct.png"),
        (reflected_img, "input_reflected.png"),
    ]:
        Image.fromarray((np.clip(arr, 0, 1)*255).astype(np.uint8)).save(
            os.path.join(out_dir, fname))
    bs_norm = (base_shape - base_shape.min()) / (np.ptp(base_shape) + 1e-8)
    Image.fromarray((bs_norm * 255).astype(np.uint8)).save(
        os.path.join(out_dir, "input_base_shape.png"))

    # Mask
    u = np.linspace(-1, 1, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    r = np.sqrt(uu**2 + vv**2)
    mask = ((r <= 1.0) & (r >= cup_R / sauc_R)).astype(float)

    # Step 1: Geometry refinement
    print("  [1/4] Refining saucer geometry...")
    final_hf = refine_geometry(base_shape, direct_img, mask, res, n_iter=80)

    # Step 2: Build mappings
    print("  [2/4] Building reflection mapping...")
    ref_uv, ref_valid = build_reflection_map(res, cup_R, cup_H, sauc_R)
    dir_uv, dir_valid = build_direct_map(res, sauc_R, cup_R)

    # Step 3: Texture optimization
    print(f"  [3/4] Optimizing saucer texture ({tex_iter} iterations)...")
    texture, losses = optimize_texture_adam(
        direct_img, reflected_img,
        dir_uv, dir_valid, ref_uv, ref_valid,
        mask, res, n_iter=tex_iter, lr=0.06
    )

    # Step 4: Render & save
    print("  [4/4] Rendering and saving results...")
    direct_render = bilinear_warp(texture, dir_uv, dir_valid)
    reflected_render = bilinear_warp(texture, ref_uv, ref_valid)
    if direct_render.ndim == 2: direct_render = np.stack([direct_render]*3, -1)
    if reflected_render.ndim == 2: reflected_render = np.stack([reflected_render]*3, -1)

    # Save texture
    Image.fromarray((np.clip(texture, 0, 1)*255).astype(np.uint8)).save(
        os.path.join(out_dir, "saucer_texture.png"))
    np.save(os.path.join(out_dir, "heightfield.npy"), final_hf)

    # OBJ mesh
    gu = np.linspace(-sauc_R, sauc_R, res)
    guu, gvv = np.meshgrid(gu, gu, indexing="xy")
    verts = np.stack([guu, gvv, final_hf], -1).reshape(-1, 3)
    uv_coords = np.stack([
        np.linspace(0,1,res)[None,:].repeat(res,0),
        np.linspace(0,1,res)[:,None].repeat(res,1)
    ], -1).reshape(-1, 2)
    faces = []
    for j in range(res-1):
        for i in range(res-1):
            idx = j*res+i
            faces.append([idx, idx+1, idx+res])
            faces.append([idx+1, idx+res+1, idx+res])
    faces = np.array(faces, dtype=np.int32)
    export_obj(verts, faces, uv_coords, "saucer_texture.png",
               os.path.join(out_dir, "saucer.obj"),
               os.path.join(out_dir, "saucer.mtl"))

    # ── Create comprehensive visualization ��───────────────────────────
    fig = plt.figure(figsize=(28, 16))

    # Title with color bar matching saucer color
    sc = scene["saucer_color"]
    fig.patch.set_facecolor((0.05, 0.05, 0.08))

    gs = fig.add_gridspec(3, 5, hspace=0.35, wspace=0.25)

    def _add_img(gs_pos, img, title, cmap=None):
        ax = fig.add_subplot(gs_pos)
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=11, color="white", fontweight="bold", pad=8)
        ax.axis("off")
        return ax

    # Row 0: INPUTS
    _add_img(gs[0, 0], direct_img, "① INPUT: Direct View Target\n(what you see on saucer)")
    _add_img(gs[0, 1], reflected_img, "② INPUT: Reflected View Target\n(what appears in cup)")
    _add_img(gs[0, 2], bs_norm, "③ INPUT: Base Saucer Shape\n(concentric wave)", cmap="terrain")

    # Dummy "photo reference" label
    ax_ref = fig.add_subplot(gs[0, 3:5])
    ax_ref.text(0.5, 0.5,
        f"📸 Reference: {scene['photo']}\n\n{scene['description']}",
        ha="center", va="center", fontsize=11, color="white",
        transform=ax_ref.transAxes, wrap=True,
        bbox=dict(boxstyle="round,pad=0.8", facecolor=(sc[0]*0.5, sc[1]*0.5, sc[2]*0.5),
                  edgecolor="white", alpha=0.8))
    ax_ref.set_facecolor((0.08, 0.08, 0.12))
    ax_ref.axis("off")

    # Row 1: OUTPUTS
    _add_img(gs[1, 0], direct_render, "④ OUTPUT: Rendered Direct View\n(algorithm result)")
    _add_img(gs[1, 1], reflected_render, "⑤ OUTPUT: Rendered Reflected View\n(algorithm result)")
    _add_img(gs[1, 2], texture, "⑥ OUTPUT: Optimized Saucer Texture\n(the actual pattern on saucer)")
    _add_img(gs[1, 3], final_hf, "⑦ OUTPUT: Refined Height Field\n(saucer surface geometry)", cmap="terrain")
    nrm = heightfield_normals(final_hf)
    _add_img(gs[1, 4], nrm[..., 2], "⑧ OUTPUT: Normal/Shading Map\n(geometry detail)", cmap="gray")

    # Row 2: Analysis
    # Direct comparison: target vs rendered
    ax_cmp_d = fig.add_subplot(gs[2, 0])
    side_by_side_d = np.concatenate([direct_img, np.clip(direct_render, 0, 1)], axis=1)
    ax_cmp_d.imshow(side_by_side_d)
    ax_cmp_d.set_title("Direct: Target (left) vs Rendered (right)", fontsize=10, color="white")
    ax_cmp_d.axis("off")

    ax_cmp_r = fig.add_subplot(gs[2, 1])
    side_by_side_r = np.concatenate([reflected_img, np.clip(reflected_render, 0, 1)], axis=1)
    ax_cmp_r.imshow(side_by_side_r)
    ax_cmp_r.set_title("Reflected: Target (left) vs Rendered (right)", fontsize=10, color="white")
    ax_cmp_r.axis("off")

    # Saucer texture on circular mask
    ax_circ = fig.add_subplot(gs[2, 2])
    circ_tex = texture * mask[..., None]
    # Add saucer color to background
    saucer_bg = np.ones((res, res, 3)) * np.array(sc)
    circ_display = np.where(mask[..., None] > 0.5, circ_tex, saucer_bg)
    ax_circ.imshow(np.clip(circ_display, 0, 1))
    ax_circ.set_title("Texture on Circular Saucer", fontsize=10, color="white")
    ax_circ.axis("off")

    # Heightfield cross section
    ax_prof = fig.add_subplot(gs[2, 3])
    mid = res // 2
    ax_prof.fill_between(np.linspace(-sauc_R, sauc_R, res), final_hf[mid, :],
                          alpha=0.4, color=(sc[0], sc[1], sc[2]))
    ax_prof.plot(np.linspace(-sauc_R, sauc_R, res), final_hf[mid, :],
                 color="white", linewidth=1.5)
    ax_prof.axvline(-cup_R, color="gold", linestyle="--", alpha=0.6, label="Cup edge")
    ax_prof.axvline(cup_R, color="gold", linestyle="--", alpha=0.6)
    ax_prof.set_facecolor((0.08, 0.08, 0.12))
    ax_prof.set_title("Height Profile (cross section)", fontsize=10, color="white")
    ax_prof.tick_params(colors="white")
    ax_prof.legend(fontsize=8, facecolor=(0.1, 0.1, 0.15), edgecolor="white", labelcolor="white")

    # Loss curve
    ax_loss = fig.add_subplot(gs[2, 4])
    ax_loss.plot(losses, color=(sc[0]*1.5, sc[1]*1.5, sc[2]*1.5), linewidth=1.2)
    ax_loss.set_facecolor((0.08, 0.08, 0.12))
    ax_loss.set_title("Optimization Convergence", fontsize=10, color="white")
    ax_loss.set_xlabel("Iteration", fontsize=9, color="white")
    ax_loss.set_ylabel("MSE Loss", fontsize=9, color="white")
    ax_loss.tick_params(colors="white")
    ax_loss.grid(True, alpha=0.15, color="white")

    plt.suptitle(
        f"🎨 {name}  —  Computational Mirror Cup & Saucer Art  (Wu et al. 2022)",
        fontsize=16, color="white", fontweight="bold", y=0.98
    )

    fig_path = os.path.join(out_dir, f"{name}_full_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"\n  ✅ Scene complete! Files in {out_dir}/:")
    for fn in os.listdir(out_dir):
        sz = os.path.getsize(os.path.join(out_dir, fn))
        print(f"     {'📄' if fn.endswith(('.png','.npy')) else '📦'} {fn:40s} ({sz//1024:>5d} KB)")

    return out_dir


# ══════════════════════════════════════════════════════════════════════════
# PART F — 3D Interactive Viewer
# ══════════════════════════════════════════════════════════════════════════

def launch_viewer(scene_dir, sc, cup_R=1.0, cup_H=2.0, sauc_R=3.0):
    hf_path = os.path.join(scene_dir, "heightfield.npy")
    tex_path = os.path.join(scene_dir, "saucer_texture.png")
    try:
        import open3d as o3d
    except ImportError:
        print("⚠️  open3d not installed, using matplotlib fallback.")
        _mpl_viewer(hf_path, tex_path, sauc_R, cup_R, cup_H, sc)
        return

    hf = np.load(hf_path) if os.path.exists(hf_path) else np.zeros((128, 128))
    res = hf.shape[0]
    u = np.linspace(-sauc_R, sauc_R, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    r2 = uu**2 + vv**2
    valid = (r2 <= sauc_R**2) & (r2 >= cup_R**2)
    verts = np.stack([uu, vv, hf], -1).reshape(-1, 3)
    faces = []
    for j in range(res-1):
        for i in range(res-1):
            if valid[j,i] and valid[j,i+1] and valid[j+1,i]:
                idx = j*res+i
                faces.append([idx, idx+1, idx+res])
                if valid[j+1,i+1]:
                    faces.append([idx+1, idx+res+1, idx+res])
    faces_np = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)

    saucer = o3d.geometry.TriangleMesh()
    saucer.vertices = o3d.utility.Vector3dVector(verts)
    saucer.triangles = o3d.utility.Vector3iVector(faces_np)
    if os.path.exists(tex_path):
        tex = np.array(Image.open(tex_path).convert("RGB").resize(
            (res, res), Image.LANCZOS), dtype=float) / 255.0
        saucer.vertex_colors = o3d.utility.Vector3dVector(tex.reshape(-1, 3))
    else:
        saucer.paint_uniform_color(list(sc))
    saucer.compute_vertex_normals()

    cup = o3d.geometry.TriangleMesh.create_cylinder(radius=cup_R, height=cup_H, resolution=64)
    cup.translate([0, 0, cup_H/2])
    cup.paint_uniform_color([0.88, 0.85, 0.60])  # gold cup
    cup.compute_vertex_normals()

    print("\n🎨 3D Mirror Cup & Saucer Viewer")
    print("━"*50)
    print("  🖱️ Left-drag → Rotate  |  Right-drag → Pan  |  Scroll → Zoom")
    print("  ⌨️ 1→Top  2→Side  3→Persp  W→Wireframe  Q→Quit")
    print("━"*50)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Mirror Cup & Saucer — 3D Viewer", 1280, 960)
    vis.add_geometry(saucer); vis.add_geometry(cup)

    ctl = vis.get_view_control()
    ctl.set_zoom(0.45); ctl.set_front([0,-0.5,0.85])
    ctl.set_lookat([0,0,0.3]); ctl.set_up([0,0,1])
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.08, 0.12])

    vis.register_key_callback(ord("1"), lambda v: (
        v.get_view_control().set_front([0,0,1]),
        v.get_view_control().set_up([0,1,0]),
        v.get_view_control().set_zoom(0.4),
        print("📷 Top view")) or False)
    vis.register_key_callback(ord("2"), lambda v: (
        v.get_view_control().set_front([0,-0.7,0.3]),
        v.get_view_control().set_up([0,0,1]),
        v.get_view_control().set_zoom(0.5),
        print("📷 Side view")) or False)
    vis.register_key_callback(ord("3"), lambda v: (
        v.get_view_control().set_front([0.5,-0.5,0.7]),
        v.get_view_control().set_up([0,0,1]),
        v.get_view_control().set_zoom(0.5),
        print("📷 Perspective")) or False)
    vis.register_key_callback(ord("W"), lambda v: (
        setattr(v.get_render_option(), 'mesh_show_wireframe',
                not v.get_render_option().mesh_show_wireframe),
        print("🔲 Wireframe toggled")) or False)
    vis.run()
    vis.destroy_window()


def _mpl_viewer(hf_path, tex_path, sauc_R, cup_R, cup_H, sc):
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    hf = np.load(hf_path) if os.path.exists(hf_path) else np.zeros((64, 64))
    res = hf.shape[0]
    u = np.linspace(-sauc_R, sauc_R, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    r2 = uu**2 + vv**2
    msk = (r2 <= sauc_R**2) & (r2 >= cup_R**2)
    zz = np.where(msk, hf, np.nan)
    tex = None
    if os.path.exists(tex_path):
        tex = np.array(Image.open(tex_path).convert("RGB").resize(
            (res, res), Image.LANCZOS), dtype=float) / 255.0

    theta = np.linspace(0, 2*np.pi, 64)
    zc = np.linspace(0, cup_H, 32)
    tg, zg = np.meshgrid(theta, zc)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    if tex is not None:
        fc = np.where(msk[..., None], tex, np.array(sc)[None, None, :])
        ax.plot_surface(uu, vv, zz, facecolors=fc, rstride=2, cstride=2, alpha=0.9)
    else:
        ax.plot_surface(uu, vv, zz, cmap="terrain", rstride=2, cstride=2, alpha=0.9)
    ax.plot_surface(cup_R*np.cos(tg), cup_R*np.sin(tg), zg, color="gold", alpha=0.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Mirror Cup & Saucer (drag to rotate)", fontsize=14)
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════
# PART G — Gallery & Main
# ══════════════════════════════════════════════════════════════════════════

def make_gallery(scenes, dirs, output_root):
    """4-scene gallery overview."""
    n = len(scenes)
    fig, axes = plt.subplots(n, 5, figsize=(30, 6*n))
    fig.patch.set_facecolor((0.05, 0.05, 0.08))

    cols = ["Direct Target (Input)", "Reflected Target (Input)",
            "Base Shape (Input)", "Optimized Texture (Output)", "Rendered Views (Output)"]

    for i, (sc, d) in enumerate(zip(scenes, dirs)):
        for j, fn in enumerate(["input_direct.png", "input_reflected.png",
                                 "input_base_shape.png", "saucer_texture.png"]):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                axes[i, j].imshow(np.array(Image.open(p)))
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(cols[j], fontsize=11, color="white", fontweight="bold")

        # Full result thumbnail
        fp = os.path.join(d, f"{sc['name']}_full_result.png")
        if os.path.exists(fp):
            axes[i, 4].imshow(np.array(Image.open(fp).resize((600, 350), Image.LANCZOS)))
        axes[i, 4].axis("off")
        if i == 0:
            axes[i, 4].set_title(cols[4], fontsize=11, color="white", fontweight="bold")

        axes[i, 0].set_ylabel(f"Scene {i+1}\n{sc['name'].replace('_', ' ')}",
                               fontsize=10, color="white", rotation=0, labelpad=120, va="center")

    plt.suptitle(
        "🎨 Luycho × oyow — Mirror Cup & Saucer Art: All 4 Scenes\n"
        "Computational reproduction using Wu et al. (ACM TOG 2022) algorithm",
        fontsize=15, color="white", fontweight="bold", y=1.01
    )
    plt.tight_layout(rect=[0.08, 0, 1, 0.97])
    gp = os.path.join(output_root, "LUYCHO_4SCENES_GALLERY.png")
    plt.savefig(gp, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n📊 Gallery: {gp}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: 4 Luycho mirror cup & saucer scenes")
    parser.add_argument("--scene", type=int, default=None, help="Run scene 1-4. Default: all")
    parser.add_argument("--res", type=int, default=256, help="Working resolution")
    parser.add_argument("--tex_iter", type=int, default=500, help="Texture opt iterations")
    parser.add_argument("--viewer", action="store_true", help="Launch 3D viewer")
    parser.add_argument("--output", type=str, default="luycho_demo_outputs")
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🎨 Luycho × oyow Mirror Cup & Saucer Art — Full Demo       ║")
    print("║  Algorithm: Wu et al., ACM TOG 2022 (doi:10.1145/3517120)   ║")
    print("║  Reproducing 4 photographed scenes                          ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    scenes = get_luycho_scenes(res=512)

    if args.scene is not None:
        idx = args.scene - 1
        if 0 <= idx < len(scenes):
            out = run_scene(scenes[idx], args.output, args.res, args.tex_iter)
            if args.viewer:
                launch_viewer(out, scenes[idx]["saucer_color"],
                              scenes[idx]["cup_radius"],
                              scenes[idx]["cup_height"],
                              scenes[idx]["saucer_radius"])
        else:
            print(f"❌ Invalid scene. Choose 1-{len(scenes)}.")
    else:
        all_dirs = []
        for i, sc in enumerate(scenes):
            print(f"\n{'▓'*72}")
            print(f"  [{i+1}/{len(scenes)}]")
            out = run_scene(sc, args.output, args.res, args.tex_iter)
            all_dirs.append(out)
        make_gallery(scenes, all_dirs, args.output)

        if args.viewer:
            print("\n🎮 Launching 3D viewer for Scene 1...")
            launch_viewer(all_dirs[0], scenes[0]["saucer_color"],
                          scenes[0]["cup_radius"], scenes[0]["cup_height"],
                          scenes[0]["saucer_radius"])

    print(f"\n🎉 Demo complete! All outputs in: {args.output}/")


if __name__ == "__main__":
    main()