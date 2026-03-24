"""
Simple differentiable-style renderer for mirror cup and saucer scenes.

Renders direct view (top-down orthographic) and reflected view (through
cylindrical mirror) of the saucer.
"""

import numpy as np
from .reflection import build_reflection_map, build_direct_view_map
from .texture import warp_image
from .geometry import SaucerGeometry


class MirrorCupSaucerRenderer:
    """
    Renders both direct and reflected views of a saucer-cup scene.
    """

    def __init__(
        self,
        img_res: int = 512,
        cup_radius: float = 1.0,
        cup_height: float = 2.0,
        saucer_radius: float = 3.0,
        eye_pos: np.ndarray = None,
    ):
        self.img_res = img_res
        self.cup_radius = cup_radius
        self.cup_height = cup_height
        self.saucer_radius = saucer_radius
        self.eye_pos = eye_pos

        # Build direct view map (constant, doesn't depend on geometry)
        self.direct_uv, self.direct_valid = build_direct_view_map(
            img_res, saucer_radius, cup_radius
        )

    def update_reflection_map(
        self, heightfield: np.ndarray = None
    ):
        """(Re-)compute the reflection mapping, optionally with heightfield."""
        self.reflected_uv, self.cup_uv, self.reflected_valid = build_reflection_map(
            self.img_res,
            self.cup_radius,
            self.cup_height,
            self.saucer_radius,
            self.eye_pos,
            heightfield,
        )

    def render_direct(self, texture: np.ndarray) -> np.ndarray:
        """Render the direct (top-down) view."""
        return warp_image(texture, self.direct_uv, self.direct_valid)

    def render_reflected(self, texture: np.ndarray) -> np.ndarray:
        """Render the reflected view through the cylindrical mirror."""
        return warp_image(texture, self.reflected_uv, self.reflected_valid)

    def render_both(self, texture: np.ndarray):
        """Return (direct_view, reflected_view) images."""
        return self.render_direct(texture), self.render_reflected(texture)