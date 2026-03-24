"""
Saucer geometry representation and optimization.

The saucer surface is represented as a base shape plus a height-field
displacement.  Optimization refines this height field so that both the
direct view and the reflected view match their target images.

Key techniques from the paper:
  - Black-white shape enhancement
  - Sparse spike movement for fine geometric detail
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple


class SaucerGeometry:
    """
    Height-field-based saucer geometry.

    Attributes
    ----------
    resolution : int
        Grid resolution.
    saucer_radius : float
    cup_radius : float
    base_shape : (res, res) ndarray
        The user-supplied base shape (height map, 0 for flat).
    displacement : (res, res) ndarray
        Learned/optimized displacement on top of base_shape.
    """

    def __init__(
        self,
        resolution: int = 256,
        saucer_radius: float = 3.0,
        cup_radius: float = 1.0,
        base_shape: Optional[np.ndarray] = None,
        max_displacement: float = 0.3,
    ):
        self.resolution = resolution
        self.saucer_radius = saucer_radius
        self.cup_radius = cup_radius
        self.max_displacement = max_displacement

        if base_shape is not None:
            from skimage.transform import resize

            self.base_shape = resize(
                base_shape, (resolution, resolution), anti_aliasing=True
            )
        else:
            self.base_shape = np.zeros((resolution, resolution))

        self.displacement = np.zeros((resolution, resolution))
        self._build_mask()

    def _build_mask(self):
        """Mask: 1 inside saucer annulus, 0 otherwise."""
        u = np.linspace(-1, 1, self.resolution)
        v = np.linspace(-1, 1, self.resolution)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        r = np.sqrt(uu ** 2 + vv ** 2)
        ratio_outer = self.saucer_radius / self.saucer_radius  # == 1
        ratio_inner = self.cup_radius / self.saucer_radius
        self.mask = ((r <= ratio_outer) & (r >= ratio_inner)).astype(np.float64)

    @property
    def heightfield(self) -> np.ndarray:
        return self.base_shape + self.displacement * self.mask

    def compute_normals(self) -> np.ndarray:
        """Finite-difference surface normals from height field."""
        h = self.heightfield
        dx = np.zeros_like(h)
        dy = np.zeros_like(h)
        dx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / 2.0
        dy[1:-1, :] = (h[2:, :] - h[:-2, :]) / 2.0
        # Normal = (-dh/dx, -dh/dy, 1) normalized
        nz = np.ones_like(h)
        length = np.sqrt(dx ** 2 + dy ** 2 + nz ** 2)
        normals = np.stack([-dx / length, -dy / length, nz / length], axis=-1)
        return normals

    def to_mesh_vertices(self) -> np.ndarray:
        """Generate (res*res, 3) vertex positions."""
        u = np.linspace(-self.saucer_radius, self.saucer_radius, self.resolution)
        v = np.linspace(-self.saucer_radius, self.saucer_radius, self.resolution)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        zz = self.heightfield
        return np.stack([uu, vv, zz], axis=-1).reshape(-1, 3)

    def to_mesh_faces(self) -> np.ndarray:
        """Generate triangle face indices for grid mesh."""
        res = self.resolution
        faces = []
        for j in range(res - 1):
            for i in range(res - 1):
                idx = j * res + i
                faces.append([idx, idx + 1, idx + res])
                faces.append([idx + 1, idx + res + 1, idx + res])
        return np.array(faces, dtype=np.int32)

    def to_mesh_uvs(self) -> np.ndarray:
        """Generate UV coordinates for the mesh."""
        u = np.linspace(0, 1, self.resolution)
        v = np.linspace(0, 1, self.resolution)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        return np.stack([uu, vv], axis=-1).reshape(-1, 2)


def optimize_geometry(
    geometry: SaucerGeometry,
    direct_target: np.ndarray,
    reflected_target: np.ndarray,
    reflection_map_fn,
    n_iterations: int = 200,
    lr: float = 0.01,
    lambda_smooth: float = 0.1,
    lambda_direct: float = 1.0,
    lambda_reflected: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Optimize saucer displacement to simultaneously match direct and
    reflected target images.

    Uses gradient descent with finite-difference gradients.

    Parameters
    ----------
    geometry : SaucerGeometry
    direct_target : (H, W) or (H, W, 3) target for direct view
    reflected_target : (H, W) or (H, W, 3) target for reflected view
    reflection_map_fn : callable
        Given heightfield, returns reflected image.
    n_iterations : int
    lr : float
    lambda_smooth : float
        Smoothness regularization weight.
    lambda_direct, lambda_reflected : float

    Returns
    -------
    info : dict with 'losses', 'geometry'
    """

    def to_gray(img):
        if img.ndim == 3:
            return np.mean(img, axis=-1)
        return img

    direct_gray = to_gray(direct_target)
    reflected_gray = to_gray(reflected_target)

    res = geometry.resolution
    losses = []
    eps = 1e-4

    for it in range(n_iterations):
        # Forward pass
        hf = geometry.heightfield

        # Direct view: the heightfield itself encodes a shading pattern
        # via surface normals (darker where steeper)
        normals = geometry.compute_normals()
        direct_shading = normals[..., 2]  # cos(angle) ≈ brightness

        # Reflected view via the mapping function
        reflected_img = reflection_map_fn(hf)

        # Losses
        loss_direct = lambda_direct * np.mean(
            (direct_shading - direct_gray) ** 2 * geometry.mask
        )
        loss_reflected = lambda_reflected * np.mean(
            (reflected_img - reflected_gray) ** 2
        )

        # Smoothness
        lap = (
            np.roll(geometry.displacement, 1, 0)
            + np.roll(geometry.displacement, -1, 0)
            + np.roll(geometry.displacement, 1, 1)
            + np.roll(geometry.displacement, -1, 1)
            - 4.0 * geometry.displacement
        )
        loss_smooth = lambda_smooth * np.mean(lap ** 2)
        loss = loss_direct + loss_reflected + loss_smooth

        losses.append(loss)
        if verbose and it % 20 == 0:
            print(
                f"Iter {it:4d} | total={loss:.6f}  direct={loss_direct:.6f}"
                f"  reflected={loss_reflected:.6f}  smooth={loss_smooth:.6f}"
            )

        # Gradient via finite differences (per-pixel)
        grad = np.zeros_like(geometry.displacement)
        # Stochastic subset for efficiency
        n_samples = min(res * res, 2000)
        idx = np.random.choice(res * res, n_samples, replace=False)
        rows, cols = np.unravel_index(idx, (res, res))

        for r, c in zip(rows, cols):
            if geometry.mask[r, c] < 0.5:
                continue
            old_val = geometry.displacement[r, c]
            # Plus
            geometry.displacement[r, c] = old_val + eps
            hf_p = geometry.heightfield
            normals_p = geometry.compute_normals()
            ds_p = normals_p[..., 2]
            ri_p = reflection_map_fn(hf_p)
            loss_p = lambda_direct * np.mean(
                (ds_p - direct_gray) ** 2 * geometry.mask
            ) + lambda_reflected * np.mean((ri_p - reflected_gray) ** 2)

            # Minus
            geometry.displacement[r, c] = old_val - eps
            hf_m = geometry.heightfield
            normals_m = geometry.compute_normals()
            ds_m = normals_m[..., 2]
            ri_m = reflection_map_fn(hf_m)
            loss_m = lambda_direct * np.mean(
                (ds_m - direct_gray) ** 2 * geometry.mask
            ) + lambda_reflected * np.mean((ri_m - reflected_gray) ** 2)

            grad[r, c] = (loss_p - loss_m) / (2.0 * eps)
            geometry.displacement[r, c] = old_val

        # Smooth gradient
        grad = gaussian_filter(grad, sigma=1.0)

        # Update
        geometry.displacement -= lr * grad * geometry.mask

        # Clamp
        geometry.displacement = np.clip(
            geometry.displacement,
            -geometry.max_displacement,
            geometry.max_displacement,
        )

    return {"losses": losses, "geometry": geometry}


def black_white_enhancement(
    displacement: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """
    Black-white shape enhancement from the paper: sharpen the height
    field to create crisper shading transitions.
    """
    enhanced = displacement.copy()
    med = np.median(np.abs(displacement[displacement != 0]))
    if med < 1e-8:
        return enhanced
    normalized = displacement / (med + 1e-8)
    # Sigmoid-like sharpening
    enhanced = np.tanh(threshold * normalized) * med
    return enhanced


def sparse_spike_movement(
    displacement: np.ndarray,
    mask: np.ndarray,
    n_spikes: int = 50,
    spike_amplitude: float = 0.1,
) -> np.ndarray:
    """
    Sparse spike movement: add localized geometric features for encoding
    fine detail (as described in the paper).
    """
    result = displacement.copy()
    res = displacement.shape[0]
    valid_idx = np.argwhere(mask > 0.5)
    if len(valid_idx) == 0:
        return result
    chosen = valid_idx[np.random.choice(len(valid_idx), min(n_spikes, len(valid_idx)), replace=False)]
    for r, c in chosen:
        sigma = np.random.uniform(1.0, 3.0)
        amp = np.random.uniform(-spike_amplitude, spike_amplitude)
        y, x = np.ogrid[-r : res - r, -c : res - c]
        gauss = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        result += amp * gauss * mask
    return result