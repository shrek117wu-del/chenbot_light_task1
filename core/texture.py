"""
Texture optimization for the saucer surface.

Given the reflection mapping and geometry, optimize the saucer texture so that:
  1. Direct view shows the direct target image.
  2. Reflected view (via the mirror cup) shows the reflected target image.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional


class SaucerTexture:
    """RGB texture on the saucer surface, stored as (res, res, 3) in [0,1]."""

    def __init__(self, resolution: int = 256, init_color: Tuple = (0.8, 0.8, 0.8)):
        self.resolution = resolution
        self.texture = np.ones((resolution, resolution, 3)) * np.array(init_color)

    @property
    def image(self) -> np.ndarray:
        return np.clip(self.texture, 0, 1)


def warp_image(
    source: np.ndarray, uv_map: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """
    Warp a source image through a UV map using bilinear interpolation.

    Parameters
    ----------
    source : (H, W, C) source texture
    uv_map : (H', W', 2) mapping – each pixel stores (u, v) in [0,1]
    valid : (H', W') bool mask

    Returns
    -------
    warped : (H', W', C)
    """
    h, w = uv_map.shape[:2]
    sh, sw = source.shape[:2]
    C = source.shape[2] if source.ndim == 3 else 1
    if source.ndim == 2:
        source = source[..., None]

    result = np.zeros((h, w, C))

    u = uv_map[..., 0] * (sw - 1)
    v = uv_map[..., 1] * (sh - 1)

    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = u0 + 1
    v1 = v0 + 1

    u0 = np.clip(u0, 0, sw - 1)
    u1 = np.clip(u1, 0, sw - 1)
    v0 = np.clip(v0, 0, sh - 1)
    v1 = np.clip(v1, 0, sh - 1)

    fu = (u - u0).astype(np.float64)
    fv = (v - v0).astype(np.float64)

    for c in range(C):
        val = (
            source[v0, u0, c] * (1 - fu) * (1 - fv)
            + source[v0, u1, c] * fu * (1 - fv)
            + source[v1, u0, c] * (1 - fu) * fv
            + source[v1, u1, c] * fu * fv
        )
        result[..., c] = val * valid

    if C == 1:
        result = result[..., 0]
    return result


def optimize_texture(
    texture: SaucerTexture,
    direct_target: np.ndarray,
    reflected_target: np.ndarray,
    direct_uv: np.ndarray,
    direct_valid: np.ndarray,
    reflected_uv: np.ndarray,
    reflected_valid: np.ndarray,
    mask: np.ndarray,
    n_iterations: int = 300,
    lr: float = 0.05,
    lambda_direct: float = 1.0,
    lambda_reflected: float = 1.0,
    lambda_smooth: float = 0.01,
    verbose: bool = True,
) -> dict:
    """
    Optimize saucer texture to match both direct and reflected target images.

    Uses Adam-style gradient descent with analytically derived gradients
    (chain rule through bilinear interpolation).
    """
    if direct_target.ndim == 2:
        direct_target = np.stack([direct_target] * 3, axis=-1)
    if reflected_target.ndim == 2:
        reflected_target = np.stack([reflected_target] * 3, axis=-1)

    res = texture.resolution
    losses = []

    # Adam optimizer state
    m = np.zeros_like(texture.texture)
    v_adam = np.zeros_like(texture.texture)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    for it in range(n_iterations):
        tex = texture.texture

        # Forward: warp texture through both mappings
        direct_rendered = warp_image(tex, direct_uv, direct_valid)
        reflected_rendered = warp_image(tex, reflected_uv, reflected_valid)

        # Losses
        diff_d = direct_rendered - direct_target
        diff_r = reflected_rendered - reflected_target
        loss_d = lambda_direct * np.mean(diff_d ** 2)
        loss_r = lambda_reflected * np.mean(diff_r ** 2)

        # Smoothness on texture
        lap = (
            np.roll(tex, 1, 0) + np.roll(tex, -1, 0)
            + np.roll(tex, 1, 1) + np.roll(tex, -1, 1) - 4.0 * tex
        )
        loss_s = lambda_smooth * np.mean(lap ** 2)
        loss = loss_d + loss_r + loss_s

        losses.append(loss)
        if verbose and it % 50 == 0:
            print(
                f"Tex iter {it:4d} | total={loss:.6f}  "
                f"direct={loss_d:.6f}  reflected={loss_r:.6f}"
            )

        # Gradient: accumulate from both views via scatter
        grad = np.zeros_like(tex)
        count = np.zeros((res, res, 1))

        # Direct view gradient
        _scatter_gradient(
            grad, count, diff_d, direct_uv, direct_valid,
            lambda_direct, res
        )
        # Reflected view gradient
        _scatter_gradient(
            grad, count, diff_r, reflected_uv, reflected_valid,
            lambda_reflected, res
        )

        grad = grad / (count + 1e-8)
        grad += lambda_smooth * 2.0 * lap  # smoothness grad
        grad *= mask[..., None]

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v_adam = beta2 * v_adam + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v_adam / (1 - beta2 ** (it + 1))
        texture.texture -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        texture.texture = np.clip(texture.texture, 0, 1)

    return {"losses": losses, "texture": texture}


def _scatter_gradient(
    grad: np.ndarray,
    count: np.ndarray,
    diff: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    weight: float,
    res: int,
):
    """Scatter per-pixel gradient back to texture coordinates (vectorized)."""
    u = uv[..., 0] * (res - 1)
    v = uv[..., 1] * (res - 1)
    u0 = np.clip(np.floor(u).astype(int), 0, res - 1)
    v0 = np.clip(np.floor(v).astype(int), 0, res - 1)

    if diff.ndim == 2:
        diff = diff[..., None]

    j_idx, i_idx = np.where(valid)
    np.add.at(grad, (v0[j_idx, i_idx], u0[j_idx, i_idx]), weight * 2.0 * diff[j_idx, i_idx])
    np.add.at(count, (v0[j_idx, i_idx], u0[j_idx, i_idx]), 1)