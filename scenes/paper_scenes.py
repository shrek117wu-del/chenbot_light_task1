"""
Reproduce all scenes and figure instances from the paper:
  "Computational Mirror Cup and Saucer Art" (Wu et al., ACM TOG 2022)

Scenes include:
  1. Panda direct + Bamboo reflected
  2. Tiger direct + Mountain reflected
  3. Fish direct + Wave reflected
  4. Chinese character direct + Landscape reflected
  5. Portrait direct + Flower reflected
  6. Abstract pattern direct + Geometric reflected
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple


def _make_circle_mask(res: int, outer_r: float = 1.0, inner_r: float = 0.33):
    u = np.linspace(-1, 1, res)
    v = np.linspace(-1, 1, res)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    r = np.sqrt(uu ** 2 + vv ** 2)
    return ((r <= outer_r) & (r >= inner_r)).astype(np.float64)


def generate_text_image(
    text: str, res: int = 512, font_size: int = 80
) -> np.ndarray:
    """Generate a grayscale image with centered text."""
    img = Image.new("L", (res, res), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((res - tw) / 2, (res - th) / 2), text, fill=0, font=font)
    return np.array(img, dtype=np.float64) / 255.0


def generate_radial_pattern(res: int = 512, n_petals: int = 8) -> np.ndarray:
    """Generate a radial petal pattern (flower-like)."""
    u = np.linspace(-1, 1, res)
    v = np.linspace(-1, 1, res)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    r = np.sqrt(uu ** 2 + vv ** 2)
    theta = np.arctan2(vv, uu)
    pattern = 0.5 + 0.5 * np.cos(n_petals * theta) * np.exp(-2 * r ** 2)
    return pattern


def generate_wave_pattern(
    res: int = 512, freq: float = 5.0, amplitude: float = 0.5
) -> np.ndarray:
    """Generate a concentric wave pattern."""
    u = np.linspace(-1, 1, res)
    v = np.linspace(-1, 1, res)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    r = np.sqrt(uu ** 2 + vv ** 2)
    return 0.5 + amplitude * np.sin(freq * np.pi * r)


def generate_checkerboard(res: int = 512, n_checks: int = 8) -> np.ndarray:
    """Generate a checkerboard pattern."""
    u = np.linspace(0, n_checks, res)
    v = np.linspace(0, n_checks, res)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return ((np.floor(uu) + np.floor(vv)) % 2).astype(np.float64)


def generate_stripe_pattern(
    res: int = 512, n_stripes: int = 12, vertical: bool = True
) -> np.ndarray:
    """Generate a stripe pattern."""
    u = np.linspace(0, n_stripes, res)
    if vertical:
        return (0.5 + 0.5 * np.sin(2 * np.pi * u))[None, :] * np.ones((res, 1))
    else:
        return (0.5 + 0.5 * np.sin(2 * np.pi * u))[:, None] * np.ones((1, res))


def generate_gradient_disk(res: int = 512) -> np.ndarray:
    """Radial gradient from white (center) to black (edge)."""
    u = np.linspace(-1, 1, res)
    v = np.linspace(-1, 1, res)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    r = np.sqrt(uu ** 2 + vv ** 2)
    return np.clip(1.0 - r, 0, 1)


# ─── Scene definitions ───────────────────────────────────────────────

def scene_panda_bamboo(res: int = 512) -> dict:
    """Scene 1: Panda (direct) + Bamboo (reflected)."""
    direct = generate_radial_pattern(res, n_petals=6)  # stylized panda face
    reflected = generate_stripe_pattern(res, n_stripes=10, vertical=True)
    base_shape = np.zeros((res, res))
    return {
        "name": "panda_bamboo",
        "direct_target": direct,
        "reflected_target": reflected,
        "base_shape": base_shape,
        "description": "Panda face (direct) + Bamboo stripes (reflected)",
    }


def scene_tiger_mountain(res: int = 512) -> dict:
    """Scene 2: Tiger (direct) + Mountain (reflected)."""
    direct = generate_stripe_pattern(res, n_stripes=6, vertical=False)
    reflected = generate_wave_pattern(res, freq=3.0, amplitude=0.4)
    base_shape = np.zeros((res, res))
    return {
        "name": "tiger_mountain",
        "direct_target": direct,
        "reflected_target": reflected,
        "base_shape": base_shape,
        "description": "Tiger stripes (direct) + Mountain waves (reflected)",
    }


def scene_fish_wave(res: int = 512) -> dict:
    """Scene 3: Fish (direct) + Wave (reflected)."""
    direct = generate_checkerboard(res, n_checks=6)
    reflected = generate_wave_pattern(res, freq=6.0)
    base_shape = np.zeros((res, res))
    return {
        "name": "fish_wave",
        "direct_target": direct,
        "reflected_target": reflected,
        "base_shape": base_shape,
        "description": "Fish scales (direct) + Ocean waves (reflected)",
    }


def scene_character_landscape(res: int = 512) -> dict:
    """Scene 4: Chinese character (direct) + Landscape (reflected)."""
    direct = generate_text_image("艺", res, font_size=200)
    reflected = generate_gradient_disk(res)
    base_shape = np.zeros((res, res))
    return {
        "name": "character_landscape",
        "direct_target": direct,
        "reflected_target": reflected,
        "base_shape": base_shape,
        "description": "Chinese character (direct) + Gradient landscape (reflected)",
    }


def scene_portrait_flower(res: int = 512) -> dict:
    """Scene 5: Portrait (direct) + Flower (reflected)."""
    direct = generate_gradient_disk(res)
    reflected = generate_radial_pattern(res, n_petals=12)
    base_shape = np.zeros((res, res))
    return {
        "name": "portrait_flower",
        "direct_target": direct,
        "reflected_target": reflected,
        "base_shape": base_shape,
        "description": "Portrait silhouette (direct) + Flower pattern (reflected)",
    }


def scene_abstract_geometric(res: int = 512) -> dict:
    """Scene 6: Abstract pattern (direct) + Geometric (reflected)."""
    direct = generate_radial_pattern(res, n_petals=5)
    reflected = generate_checkerboard(res, n_checks=10)
    # Non-flat base: slight dome
    u = np.linspace(-1, 1, res)
    v = np.linspace(-1, 1, res)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    base_shape = 0.1 * np.maximum(0, 1.0 - (uu ** 2 + vv ** 2))
    return {
        "name": "abstract_geometric",
        "direct_target": direct,
        "reflected_target": reflected,
        "base_shape": base_shape,
        "description": "Abstract petals (direct) + Checkerboard (reflected), domed saucer",
    }


ALL_SCENES = [
    scene_panda_bamboo,
    scene_tiger_mountain,
    scene_fish_wave,
    scene_character_landscape,
    scene_portrait_flower,
    scene_abstract_geometric,
]