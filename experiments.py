"""
Experiments: base shape generation and test-image creation.
Supports the shapes used in the paper (plane, saucer, tabula_scalata, random)
and real image pairs from the data/ directory.

Cup types (mirror cup shapes):
  cylinder : circular cylinder (paper default, radius r)
  ellipse  : elliptical cylinder (semi-axes a, b)
  ngon4    : square prism  (n=4)
  ngon6    : hexagonal prism (n=6)  — paper Figure 19
  ngon8    : octagonal prism (n=8)
  ngonN    : regular N-gonal prism (any integer N)
"""
import os
import numpy as np
import cv2


# Paper resolution: 150 x 150 (xy-size 1x1, height 0..1.5)
DEFAULT_RES = (150, 150)


def create_base_shape(shape_type, resolutions=DEFAULT_RES):
    """
    Build (grid_x, grid_y, base_height) for the given shape.

    Paper: the saucer spans x∈[-0.5,0.5], y∈[-1.5,-0.5] in the xOy plane.
    The mirror cylinder sits at the origin with radius 0.4.

    Returns:
        grid_x, grid_y : [H, W] numpy arrays
        Z              : [H, W] numpy array  (height values)
    """
    H, W = resolutions
    # Paper coordinate system: saucer is on the negative y-axis side
    x_coords = np.linspace(-0.5,  0.5, W)
    y_coords = np.linspace(-1.5, -0.5, H)
    X, Y = np.meshgrid(x_coords, y_coords)

    if shape_type == "plane":
        Z = np.zeros_like(X)

    elif shape_type == "random":
        np.random.seed(42)
        # Low-frequency random bump map (same seed as original)
        small = np.random.randn(max(H // 10, 3), max(W // 10, 3))
        Z = cv2.resize(small.astype(np.float32), (W, H),
                       interpolation=cv2.INTER_CUBIC)
        Z = 0.08 * Z   # keep within delta

    elif shape_type == "tabula_scalata":
        # Corrugated saucer (the artist's original wave pattern)
        freq = 12.0
        Z = 0.05 * np.sin(freq * X)

    elif shape_type == "saucer":
        # Radially symmetric bowl centered at (0, -1) — centre of saucer patch
        R = np.sqrt(X ** 2 + (Y + 1.0) ** 2)
        Z = 0.8 * R ** 2   # parabolic bowl, height up to ~0.8*0.5^2=0.2

    elif shape_type == "dish":
        # Realistic dish/plate shape: flat center with gently raised edges
        # This matches the paper's typical saucer reference shape (Figure 1)
        R = np.sqrt(X ** 2 + (Y + 1.0) ** 2)
        R_max = np.sqrt(0.5 ** 2 + 0.5 ** 2)
        r_norm = R / R_max
        # Flat center, raised edges (like a real saucer/dish)
        Z = 0.15 * np.clip(r_norm - 0.5, 0, None) ** 2 / 0.25

    elif shape_type == "shallow_bowl":
        R = np.sqrt(X ** 2 + (Y + 1.0) ** 2)
        Z = 0.4 * R ** 2

    elif shape_type == "cone":
        # Conical saucer – linear radial profile
        R = np.sqrt(X ** 2 + (Y + 1.0) ** 2)
        Z = 0.3 * R

    elif shape_type == "saddle":
        # Hyperbolic saddle surface
        Z = 0.15 * (X ** 2 - (Y + 1.0) ** 2)
        Z = Z - Z.min()   # shift to non-negative

    elif shape_type == "wave":
        # Wave shape similar to the artist's design (paper Figure 2)
        # Sinusoidal waves in both X and Y directions
        Z = 0.04 * (np.sin(8.0 * X) * np.cos(6.0 * (Y + 1.0)) + 1.0)

    elif shape_type == "luycho_concentric":
        # Concentric ring/groove pattern like the Luycho blue saucer
        # Profile: Z = amplitude * sin(2π * n_rings * R / R_max) with raised rim
        R = np.sqrt(X ** 2 + (Y + 1.0) ** 2)
        R_max = np.sqrt(0.5 ** 2 + 0.5 ** 2)
        n_rings = 13
        amplitude = 0.03
        Z = amplitude * np.sin(2.0 * np.pi * n_rings * R / R_max)
        # Add a slightly raised rim
        r_norm = R / R_max
        Z += 0.04 * np.clip(r_norm - 0.8, 0, None) / 0.2
        Z = Z - Z.min()  # shift to non-negative

    elif shape_type == "luycho_radial":
        # Radial stripe/corrugation pattern like the Luycho white saucer
        # Profile: Z = amplitude * sin(n_stripes * theta), theta = polar angle
        theta = np.arctan2(Y + 1.0, X)   # polar angle centred at (0, -1)
        R = np.sqrt(X ** 2 + (Y + 1.0) ** 2)
        R_max = np.sqrt(0.5 ** 2 + 0.5 ** 2)
        n_stripes = 48
        amplitude = 0.025
        # Amplitude fades toward centre (no corrugation right at centre)
        r_norm = np.clip(R / R_max, 0, 1)
        Z = amplitude * r_norm * (np.sin(n_stripes * theta) + 1.0)

    else:
        raise ValueError(f"Unknown shape_type: '{shape_type}'. "
                         "Choose from: plane, random, tabula_scalata, saucer, dish, "
                         "shallow_bowl, cone, saddle, wave, luycho_concentric, luycho_radial")

    return X, Y, Z


def create_reflection_grid(cup_type, grid_x, grid_y, P_2d,
                           r=0.4, a=0.5, b=0.3):
    """
    Factory that computes the reflected XY positions for every vertex in the
    heightfield grid, for the requested mirror cup type.

    cup_type options
    ----------------
    'cylinder'  : circular cylinder (paper default), uses radius r
    'ellipse'   : elliptical cylinder, uses semi-axes a, b
    'ngon4'     : regular square prism    (n=4), uses circumradius r
    'ngon6'     : regular hexagonal prism (n=6), uses circumradius r  — paper Fig. 19
    'ngon8'     : regular octagonal prism (n=8), uses circumradius r
    'ngon<N>'   : regular N-gonal prism (any integer N ≥ 3)

    Returns
    -------
    grid_rx, grid_ry : (H, W) numpy arrays — virtual image XY positions
    """
    from core.geometry import (precompute_reflection_grid,
                           precompute_reflection_ellipse_grid,
                           precompute_reflection_ngon_grid,
                           precompute_reflection_tapered_grid)

    if cup_type == 'cylinder':
        return precompute_reflection_grid(grid_x, grid_y, P_2d, r=r)

    if cup_type == 'ellipse':
        return precompute_reflection_ellipse_grid(grid_x, grid_y, P_2d, a=a, b=b)

    if cup_type == 'luycho_tapered':
        # Tapered/conical cup: bottom radius 0.25, top radius 0.4, height 2.0
        return precompute_reflection_tapered_grid(grid_x, grid_y, P_2d,
                                                  r_bottom=0.25, r_top=0.4, height=2.0)

    if cup_type == 'luycho_straight':
        # Straight cylindrical cup, slightly wider: radius 0.38
        return precompute_reflection_grid(grid_x, grid_y, P_2d, r=0.38)

    if cup_type.startswith('ngon'):
        suffix = cup_type[4:]
        if not suffix:
            raise ValueError("cup_type 'ngon' requires a side count, e.g. 'ngon6'.")
        try:
            n = int(suffix)
        except ValueError:
            raise ValueError(f"Invalid cup_type '{cup_type}'. "
                             "Use 'ngon4', 'ngon6', etc.")
        if n < 3:
            raise ValueError("n-gonal prism requires n ≥ 3.")
        return precompute_reflection_ngon_grid(grid_x, grid_y, P_2d, n=n, r=r)

    raise ValueError(f"Unknown cup_type: '{cup_type}'. "
                     "Choose from: cylinder, ellipse, luycho_tapered, luycho_straight, "
                     "ngon4, ngon6, ngon8, ngon<N>")


def generate_test_images(size=(512, 512), output_dir="data"):
    """
    Create synthetic direct/reflect target images (letter A and B) and
    save them to output_dir.  Returns (path_direct, path_reflect).
    """
    os.makedirs(output_dir, exist_ok=True)

    def make_letter(letter, fg_bgr, size):
        img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        font_scale = max(size[0] / 64, 1.0)
        thickness  = max(int(font_scale * 2), 1)
        # Centre the text
        (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        ox = (size[0] - tw) // 2
        oy = (size[1] + th) // 2
        cv2.putText(img, letter, (ox, oy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg_bgr, thickness, cv2.LINE_AA)
        return img

    img_d = make_letter("A", (0, 0, 200), size)   # red A
    img_r = make_letter("B", (200, 0, 0), size)   # blue B

    path_d = os.path.join(output_dir, "target_direct.png")
    path_r = os.path.join(output_dir, "target_reflect.png")
    cv2.imwrite(path_d, img_d)
    cv2.imwrite(path_r, img_r)
    return path_d, path_r


def generate_synthetic_pair(label_d, label_r, fg_d=(0,0,200), fg_r=(200,0,0),
                            size=(512,512), output_dir="data"):
    """Generate a synthetic image pair with two text labels on white background."""
    os.makedirs(output_dir, exist_ok=True)

    def make_img(text, fg_bgr, size):
        img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        font_scale = max(size[0] / 80, 1.0)
        thickness  = max(int(font_scale * 2), 1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        ox = (size[0] - tw) // 2
        oy = (size[1] + th) // 2
        cv2.putText(img, text, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, fg_bgr, thickness, cv2.LINE_AA)
        return img

    img_d = make_img(label_d, fg_d, size)
    img_r = make_img(label_r, fg_r, size)

    safe_d = label_d.replace(" ", "_")
    safe_r = label_r.replace(" ", "_")
    path_d = os.path.join(output_dir, f"synth_{safe_d}-direct.png")
    path_r = os.path.join(output_dir, f"synth_{safe_r}-reflect.png")
    cv2.imwrite(path_d, img_d)
    cv2.imwrite(path_r, img_r)
    return path_d, path_r


def generate_shape_image(shape_name, fg_color=(0, 0, 200), size=(512, 512),
                         output_dir="data"):
    """Generate an image with a simple geometric shape (circle, star, heart, etc.)."""
    os.makedirs(output_dir, exist_ok=True)
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    cx, cy = size[0] // 2, size[1] // 2
    r = min(size) // 4

    if shape_name == "circle":
        cv2.circle(img, (cx, cy), r, fg_color, -1, cv2.LINE_AA)
    elif shape_name == "star":
        pts = []
        for k in range(5):
            angle = -np.pi / 2 + k * 2 * np.pi / 5
            pts.append([int(cx + r * np.cos(angle)),
                        int(cy + r * np.sin(angle))])
            angle2 = angle + np.pi / 5
            pts.append([int(cx + r * 0.4 * np.cos(angle2)),
                        int(cy + r * 0.4 * np.sin(angle2))])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(img, [pts], fg_color, cv2.LINE_AA)
    elif shape_name == "triangle":
        pts = np.array([
            [cx, cy - r],
            [cx - int(r * 0.87), cy + r // 2],
            [cx + int(r * 0.87), cy + r // 2],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], fg_color, cv2.LINE_AA)
    elif shape_name == "square":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), fg_color, -1)
    else:
        # Default: filled circle
        cv2.circle(img, (cx, cy), r, fg_color, -1, cv2.LINE_AA)

    path = os.path.join(output_dir, f"shape_{shape_name}.png")
    cv2.imwrite(path, img)
    return path


def get_paper_experiment(exp_id=1, output_dir="data"):
    """
    Return (direct_path, reflect_path) for one of the paper's built-in image pairs.
    exp_id=1 : data/1-direct.png + data/1-reflect.png
    exp_id=2 : data/2-direct.png + data/2-reflect.png
    exp_id=0 : generated test images (A / B)
    exp_id=3..7 : additional synthetic pairs for paper experiments
    """
    if exp_id == 0:
        return generate_test_images(output_dir=output_dir)

    # Additional synthetic experiments for demonstrating all paper examples
    synth_pairs = {
        3: ("C", "D", (0, 0, 200), (200, 0, 0)),
        4: ("E", "F", (0, 150, 0), (150, 0, 150)),
        5: ("X", "Y", (0, 0, 0), (0, 0, 0)),
        6: ("M", "W", (200, 0, 0), (0, 0, 200)),
        7: ("H", "K", (0, 100, 200), (200, 100, 0)),
    }
    if exp_id in synth_pairs:
        ld, lr, cd, cr = synth_pairs[exp_id]
        return generate_synthetic_pair(ld, lr, cd, cr, output_dir=output_dir)

    p_d = os.path.join(output_dir, f"{exp_id}-direct.png")
    p_r = os.path.join(output_dir, f"{exp_id}-reflect.png")
    if not os.path.exists(p_d) or not os.path.exists(p_r):
        raise FileNotFoundError(
            f"Images {p_d} / {p_r} not found. "
            "Place the paper's images in the data/ folder."
        )
    return p_d, p_r


# -----------------------------------------------------------------------
# Paper experiment configurations
# -----------------------------------------------------------------------
# Matching the paper's Figures 15-21 experiments.

PAPER_SHAPES = ["plane", "random", "tabula_scalata", "saucer", "shallow_bowl",
                "dish", "wave"]
"""The saucer shapes tested in the paper (Figure 15 and beyond)."""

PAPER_CUP_TYPES = ["cylinder", "ngon4", "ngon6", "ngon8", "ellipse"]
"""Mirror cup types from paper (Section 3.5, Figure 19)."""

# Figure 15: 5 different shapes with one image pair
FIG15_EXPERIMENTS = [
    {"shape": s, "exp_id": 1, "cup": "cylinder"}
    for s in PAPER_SHAPES
]

# Figure 18: Stress tests — plane, random, tabula_scalata with exp_id=1
FIG18_EXPERIMENTS = [
    {"shape": "plane",           "exp_id": 1, "cup": "cylinder"},
    {"shape": "random",          "exp_id": 1, "cup": "cylinder"},
    {"shape": "tabula_scalata",  "exp_id": 1, "cup": "cylinder"},
]

# Figure 19: n-gon prism examples
FIG19_EXPERIMENTS = [
    {"shape": "plane", "exp_id": 1, "cup": "ngon4"},
    {"shape": "plane", "exp_id": 1, "cup": "ngon6"},
    {"shape": "plane", "exp_id": 1, "cup": "ngon8"},
    {"shape": "plane", "exp_id": 1, "cup": "ellipse"},
]

# Combined: all paper example configurations
ALL_PAPER_EXPERIMENTS = (
    FIG15_EXPERIMENTS + FIG18_EXPERIMENTS + FIG19_EXPERIMENTS
)

# Full matrix: 6 saucer types × 3 cup types (18 combinations)
FULL_MATRIX_SAUCERS = [
    "tabula_scalata", "dish", "shallow_bowl", "wave",
    "luycho_concentric", "luycho_radial",
]
FULL_MATRIX_CUPS = ["cylinder", "luycho_tapered", "luycho_straight"]

FULL_MATRIX_EXPERIMENTS = [
    {"shape": s, "exp_id": 1, "cup": c}
    for s in FULL_MATRIX_SAUCERS
    for c in FULL_MATRIX_CUPS
]
