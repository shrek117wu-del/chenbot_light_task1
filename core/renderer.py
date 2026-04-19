"""
Differentiable Soft Triangle Rasterizer for Mirror Cup and Saucer Art.
Implements the SoftRas-style renderer from:
  Liu et al., "Soft Rasterizer: A Differentiable Renderer for Image-Based 3D Reasoning", ICCV 2019.

Key fix: correct perspective divide uses w-component (col 3) of clip-space, not z (col 2).
"""
import math
import torch
import torch.nn.functional as F


def get_camera_matrix(camera_pos, look_at, up=(0., 1., 0.),
                      fov_deg=45.0, aspect_ratio=1.0, near=0.1, far=100.0):
    """
    Build a combined View+Projection matrix (4x4, row-major for PyTorch @ convention).

    Paper setup:
      camera_pos = [0, -5.5, 5]
      look_at_direct  = [0, 0, -0.8],  fov = 4.5 deg
      look_at_reflect = [0, 0,  0.1],  fov = 3.0 deg
      up = [0, 0, 1]  (z is vertical in paper coord system)
    """
    camera_pos = torch.tensor(camera_pos, dtype=torch.float32)
    look_at    = torch.tensor(look_at,    dtype=torch.float32)
    up         = torch.tensor(up,         dtype=torch.float32)

    forward = look_at - camera_pos
    forward = forward / torch.norm(forward)

    right = torch.linalg.cross(forward, up)
    right_norm = torch.norm(right)
    if right_norm < 1e-6:
        # degenerate: forward parallel to up – use a fallback
        up = torch.tensor([1., 0., 0.], dtype=torch.float32)
        right = torch.linalg.cross(forward, up)
    right = right / torch.norm(right)

    cam_up = torch.linalg.cross(right, forward)
    cam_up = cam_up / torch.norm(cam_up)

    # View matrix (camera extrinsic): world → camera
    # Row layout for left-multiply: v_cam = M @ v_world
    E = torch.eye(4, dtype=torch.float32)
    E[0, :3] =  right
    E[1, :3] =  cam_up
    E[2, :3] = -forward
    E[0, 3]  = -torch.dot(right,   camera_pos)
    E[1, 3]  = -torch.dot(cam_up,  camera_pos)
    E[2, 3]  =  torch.dot(forward, camera_pos)

    # Perspective projection matrix (OpenGL-style)
    fov_rad = math.radians(fov_deg)
    f = 1.0 / math.tan(fov_rad / 2.0)

    P = torch.zeros(4, 4, dtype=torch.float32)
    P[0, 0] = f / aspect_ratio
    P[1, 1] = f
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = (2.0 * far * near) / (near - far)
    P[3, 2] = -1.0  # places depth in w

    return P @ E  # full MVP


class SoftTriangleRenderer(torch.nn.Module):
    """
    Memory-efficient SoftRas-style differentiable rasterizer.

    For each pixel p we compute:
        D(p, f) = sigmoid( min_edge_dist(p, f) / sigma )
        w_z(f)  = exp( (z_min - z_f) / gamma )
        color(p) = sum_f [ D * w_z * c_f ] / sum_f [ D * w_z ]

    The rendered image is compared to the target via ||mask*(R - T)||^2 / |mask|.
    """

    def __init__(self, img_size=(512, 512), sigma=1e-5, gamma=1e-4, chunk_size=2048):
        super().__init__()
        self.img_size   = img_size
        self.sigma      = sigma
        self.gamma      = gamma
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project(self, vertices, P_matrix):
        """
        Project [N,3] vertices → [N,2] screen NDC + [N] depth.
        Clip coords = P @ [x,y,z,1]^T, then divide by w (col 3).
        """
        N = vertices.shape[0]
        device = vertices.device
        ones = torch.ones(N, 1, device=device, dtype=vertices.dtype)
        pts_h = torch.cat([vertices, ones], dim=1)        # [N, 4]
        clip  = pts_h @ P_matrix.T                        # [N, 4]

        w = clip[:, 3].clamp(min=1e-6)
        u = clip[:, 0] / w   # NDC x  ∈ [-1, 1]
        v = clip[:, 1] / w   # NDC y  ∈ [-1, 1]
        z = clip[:, 2] / w   # NDC z  (depth ∈ [-1, 1])
        return torch.stack([u, v], dim=1), z              # [N,2], [N]

    def _rasterize_chunk(self, p_uv, fv, fc, fz, sigma):
        """
        Rasterize one pixel-chunk against a set of triangles.

        Args:
            p_uv : [P_c, 2] pixel NDC coords
            fv   : [F_k, 3, 2] face vertex NDC coords
            fc   : [F_k, 3] face colors
            fz   : [F_k]   face depth (mean of vertices)
            sigma: float

        Returns:
            c_chunk  : [P_c, 3]
            mask_chunk: [P_c, 1]  (binary; 1 if triangle visible at pixel)
        """
        v0, v1, v2 = fv[:, 0], fv[:, 1], fv[:, 2]   # [F_k, 2]
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        px = p_uv[:, 0:1]   # [P_c, 1]
        py = p_uv[:, 1:2]

        # Signed cross-products (pixel vs each edge)
        # Each is [P_c, F_k]
        cp0 = (px - v0[:, 0]) * e0[:, 1] - (py - v0[:, 1]) * e0[:, 0]
        cp1 = (px - v1[:, 0]) * e1[:, 1] - (py - v1[:, 1]) * e1[:, 0]
        cp2 = (px - v2[:, 0]) * e2[:, 1] - (py - v2[:, 1]) * e2[:, 0]

        # Signed area of triangle (positive = CCW in y-up NDC space)
        area = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) \
             - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
        # Our cross-product formula: cp = (p-v) x e  = -(e x (p-v))
        # For CCW triangles, interior gives cp < 0, so we negate sign
        # to make interior → positive (so sigmoid(d/sigma) > 0.5 = inside)
        sign = -torch.sign(area).unsqueeze(0)  # [1, F_k], negated for correct orientation

        s0 = cp0 * sign
        s1 = cp1 * sign
        s2 = cp2 * sign

        # Soft probability D ∈ (0, 1)
        dist_min = torch.min(torch.min(s0, s1), s2)      # [P_c, F_k]
        D = torch.sigmoid(dist_min / sigma)               # [P_c, F_k]

        # Depth weight (front face wins)
        w_z = torch.exp((fz.min() - fz) / self.gamma)    # [F_k]
        W = D * w_z.unsqueeze(0)                          # [P_c, F_k]

        W_sum = W.sum(dim=1, keepdim=True).clamp(min=1e-6)
        c_chunk = (W @ fc) / W_sum                        # [P_c, 3]
        mask_chunk = (D.max(dim=1)[0] > 0.5).float().unsqueeze(1)

        return c_chunk, mask_chunk

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, vertices, faces, colors, P_matrix, sigma=None):
        """
        Render the mesh and return (rendered_image [H,W,3], mask [H,W,1]).
        No gradient through this call.
        """
        if sigma is None:
            sigma = self.sigma

        device = vertices.device
        H, W   = self.img_size

        uv, z = self._project(vertices, P_matrix)         # [N,2], [N]

        face_uv = uv[faces]                               # [F,3,2]
        face_c  = colors[faces].mean(dim=1)               # [F,3]
        face_z  = z[faces].mean(dim=1)                    # [F]

        # Visibility cull
        tri_min = face_uv.min(dim=1)[0]
        tri_max = face_uv.max(dim=1)[0]
        vis = (tri_max[:, 0] > -1.05) & (tri_min[:, 0] < 1.05) & \
              (tri_max[:, 1] > -1.05) & (tri_min[:, 1] < 1.05) & \
              (face_z < 1.0) & (face_z > -1.0)

        if not vis.any():
            return torch.zeros(H, W, 3, device=device), \
                   torch.zeros(H, W, 1, device=device)

        fv = face_uv[vis]
        fc = face_c[vis]
        fz = face_z[vis]
        tm = tri_min[vis]
        tx = tri_max[vis]

        # Pixel grid
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        yg, xg = torch.meshgrid(ys, xs, indexing='ij')
        grid   = torch.stack([xg, yg], dim=-1).view(-1, 2)   # [P, 2]
        P_count = grid.shape[0]

        buf   = 10 * sigma
        c_list, m_list = [], []

        for i in range(0, P_count, self.chunk_size):
            p_uv = grid[i:i + self.chunk_size]
            pmin = p_uv.min(dim=0)[0]
            pmax = p_uv.max(dim=0)[0]

            # Per-chunk triangle filter
            cmask = (tx[:, 0] > pmin[0] - buf) & (tm[:, 0] < pmax[0] + buf) & \
                    (tx[:, 1] > pmin[1] - buf) & (tm[:, 1] < pmax[1] + buf)

            if not cmask.any():
                c_list.append(torch.zeros(p_uv.shape[0], 3, device=device))
                m_list.append(torch.zeros(p_uv.shape[0], 1, device=device))
                continue

            cc, mc = self._rasterize_chunk(p_uv, fv[cmask], fc[cmask], fz[cmask], sigma)
            c_list.append(cc)
            m_list.append(mc)

        img  = torch.cat(c_list, dim=0).view(H, W, 3)
        mask = torch.cat(m_list, dim=0).view(H, W, 1)
        return img, mask

    def forward(self, vertices, faces, colors, target_bchw, P_matrix, sigma=None):
        """
        Compute visual fidelity loss: mean squared error over visible pixels.

        Args:
            vertices      : [N, 3] float32
            faces         : [F, 3] long
            colors        : [N, 3] float32, in [0,1]
            target_bchw   : [1, 3, H, W] float32
            P_matrix      : [4, 4] float32
            sigma         : override for SoftRas σ

        Returns:
            scalar loss
        """
        if sigma is None:
            sigma = self.sigma

        device = vertices.device
        H, W   = self.img_size

        uv, z = self._project(vertices, P_matrix)
        face_uv = uv[faces]
        face_c  = colors[faces].mean(dim=1)
        face_z  = z[faces].mean(dim=1)

        tri_min = face_uv.min(dim=1)[0]
        tri_max = face_uv.max(dim=1)[0]
        vis = (tri_max[:, 0] > -1.05) & (tri_min[:, 0] < 1.05) & \
              (tri_max[:, 1] > -1.05) & (tri_min[:, 1] < 1.05) & \
              (face_z < 1.0) & (face_z > -1.0)

        if not vis.any():
            return torch.tensor(1.0, device=device, requires_grad=True)

        fv = face_uv[vis]
        fc = face_c[vis]
        fz = face_z[vis]
        tm = tri_min[vis]
        tx = tri_max[vis]

        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        yg, xg = torch.meshgrid(ys, xs, indexing='ij')
        grid   = torch.stack([xg, yg], dim=-1).view(-1, 2)
        P_count = grid.shape[0]

        target_flat = target_bchw.squeeze(0).view(3, H * W).T  # [P, 3]

        buf = 10 * sigma
        loss_sum   = torch.tensor(0.0, device=device)
        mask_count = torch.tensor(0.0, device=device)

        for i in range(0, P_count, self.chunk_size):
            p_uv = grid[i:i + self.chunk_size]
            t_chunk = target_flat[i:i + self.chunk_size]
            pmin = p_uv.min(dim=0)[0]
            pmax = p_uv.max(dim=0)[0]

            cmask = (tx[:, 0] > pmin[0] - buf) & (tm[:, 0] < pmax[0] + buf) & \
                    (tx[:, 1] > pmin[1] - buf) & (tm[:, 1] < pmax[1] + buf)

            if not cmask.any():
                continue

            cc, mc = self._rasterize_chunk(p_uv, fv[cmask], fc[cmask], fz[cmask], sigma)
            diff = (mc * (cc - t_chunk)) ** 2
            loss_sum   = loss_sum + diff.sum()
            mask_count = mask_count + mc.sum()

        return loss_sum / mask_count.clamp(min=1.0)


class SoftTriangleRendererLoss(SoftTriangleRenderer):
    """
    Wrapper that computes loss = SUM of squared pixel differences (matching
    the paper's Frobenius norm definition).  Exposed separately so the
    basic ``render()`` method is unaffected.
    """

    def forward(self, vertices, faces, colors, target_bchw, P_matrix, sigma=None):
        """Same interface as SoftTriangleRenderer.forward but returns SUM loss."""
        if sigma is None:
            sigma = self.sigma

        device = vertices.device
        H, W   = self.img_size

        uv, z = self._project(vertices, P_matrix)
        face_uv = uv[faces]
        face_c  = colors[faces].mean(dim=1)
        face_z  = z[faces].mean(dim=1)

        tri_min = face_uv.min(dim=1)[0]
        tri_max = face_uv.max(dim=1)[0]
        vis = (tri_max[:, 0] > -1.05) & (tri_min[:, 0] < 1.05) & \
              (tri_max[:, 1] > -1.05) & (tri_min[:, 1] < 1.05) & \
              (face_z < 1.0) & (face_z > -1.0)

        if not vis.any():
            return torch.tensor(1.0, device=device, requires_grad=True)

        fv = face_uv[vis]
        fc = face_c[vis]
        fz = face_z[vis]
        tm = tri_min[vis]
        tx = tri_max[vis]

        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        yg, xg = torch.meshgrid(ys, xs, indexing='ij')
        grid   = torch.stack([xg, yg], dim=-1).view(-1, 2)
        P_count = grid.shape[0]

        target_flat = target_bchw.squeeze(0).view(3, H * W).T  # [P, 3]

        buf = 10 * sigma
        loss_sum   = torch.tensor(0.0, device=device)

        for i in range(0, P_count, self.chunk_size):
            p_uv = grid[i:i + self.chunk_size]
            t_chunk = target_flat[i:i + self.chunk_size]
            pmin = p_uv.min(dim=0)[0]
            pmax = p_uv.max(dim=0)[0]

            cmask = (tx[:, 0] > pmin[0] - buf) & (tm[:, 0] < pmax[0] + buf) & \
                    (tx[:, 1] > pmin[1] - buf) & (tm[:, 1] < pmax[1] + buf)

            if not cmask.any():
                continue

            cc, mc = self._rasterize_chunk(p_uv, fv[cmask], fc[cmask], fz[cmask], sigma)
            diff = (mc * (cc - t_chunk)) ** 2
            loss_sum = loss_sum + diff.sum()

        return loss_sum
