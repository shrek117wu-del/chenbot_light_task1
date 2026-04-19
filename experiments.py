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

    else:
        raise ValueError(f"Unknown shape_type: '{shape_type}'. "
                         "Choose from: plane, random, tabula_scalata, saucer, "
                         "shallow_bowl, cone, saddle")

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
                           precompute_reflection_ngon_grid)

    if cup_type == 'cylinder':
        return precompute_reflection_grid(grid_x, grid_y, P_2d, r=r)

    if cup_type == 'ellipse':
        return precompute_reflection_ellipse_grid(grid_x, grid_y, P_2d, a=a, b=b)

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
                     "Choose from: cylinder, ellipse, ngon4, ngon6, ngon8, ngon<N>")


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


def get_paper_experiment(exp_id=1, output_dir="data"):
    """
    Return (direct_path, reflect_path) for one of the paper's built-in image pairs.
    exp_id=1 : data/1-direct.png + data/1-reflect.png
    exp_id=2 : data/2-direct.png + data/2-reflect.png
    exp_id=0 : generated test images (A / B)
    """
    if exp_id == 0:
        return generate_test_images(output_dir=output_dir)

    p_d = os.path.join(output_dir, f"{exp_id}-direct.png")
    p_r = os.path.join(output_dir, f"{exp_id}-reflect.png")
    if not os.path.exists(p_d) or not os.path.exists(p_r):
        raise FileNotFoundError(
            f"Images {p_d} / {p_r} not found. "
            "Place the paper's images in the data/ folder."
        )
    return p_d, p_r
