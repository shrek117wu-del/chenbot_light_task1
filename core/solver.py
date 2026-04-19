"""
Mirror Cup and Saucer Art — Optimization Solver
Implements Algorithm 1 from Wu et al. 2022:
  1. Black-White Enhancement (two-step σ)
  2. Sparse Spike Strategy (proximal gradient)
  3. Texturing
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2

from .geometry import precompute_reflection_grid, get_grid_triangles
from .renderer  import SoftTriangleRenderer, SoftTriangleRendererLoss, get_camera_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_laplacian_kernel(device):
    k = torch.tensor([[0., 1., 0.],
                       [1.,-4., 1.],
                       [0., 1., 0.]], device=device).view(1, 1, 3, 3)
    return k

def compute_laplacian(h_2d, kernel):
    """h_2d: [1,1,H,W]  →  Laplacian same shape"""
    return F.conv2d(h_2d, kernel, padding=1)


def load_and_enhance(img_path):
    """
    Load an image and produce black-white enhanced version.
    Returns:
        orig      : [H, W, 3] float32 in [0,1]
        enhanced  : [H, W, 3] float32 – fg black (direct) or white (reflect)
        fg_mask   : [H, W] bool
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Detect foreground: anything significantly different from white background
    fg = np.sum(np.abs(rgb - 1.0), axis=-1) > 0.15
    return rgb, fg


def build_enhanced_images(Id_path, Ir_path):
    """
    Build Id, Ir, Ied, Ier arrays as described in Section 3.4.1.
    Ied: fg → black (0,0,0), bg → light-blue (0, 0.5, 1)
    Ier: fg → white (1,1,1), bg → light-blue (0, 0.5, 1)
    """
    Id, fg_d = load_and_enhance(Id_path)
    Ir, fg_r = load_and_enhance(Ir_path)

    LIGHT_BLUE = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    Ied = np.full_like(Id, LIGHT_BLUE)
    Ied[fg_d] = [0., 0., 0.]          # foreground → black

    Ier = np.full_like(Ir, LIGHT_BLUE)
    Ier[fg_r] = [1., 1., 1.]          # foreground → white

    return Id, Ir, fg_d, fg_r, Ied, Ier


def img_to_tensor(arr, device, target_hw=None):
    """[H,W,3] float32 → [1,3,H,W] tensor on device, optionally resized."""
    t = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    if target_hw is not None:
        t = torch.nn.functional.interpolate(t, size=target_hw,
                                            mode='bilinear', align_corners=False)
    return t.to(device)


# ---------------------------------------------------------------------------
# Log-barrier (paper Section 3.1 Eq. 4-5)
# ---------------------------------------------------------------------------
_LOG_BARRIER_EPS = 1e-6

def log_barrier(x):
    """Safe log barrier: -log(x) for x > eps, else large penalty."""
    return -torch.log(x.clamp(min=_LOG_BARRIER_EPS))


def barrier_loss(h, c, z_orig, delta):
    """
    Ebarrier: keeps |hi - zi| < delta  and  0 < cj < 1.
    Uses log barriers as in the paper (Eq. 5).
    Paper uses SUM; scaled by a small coefficient μ for numerical stability.
    """
    mu = 1e-4    # barrier weight coefficient (keeps barrier < visual/deform terms)
    # Height barrier
    diff      = h - z_orig                           # signed
    h_upper   = log_barrier(delta - diff)            # zi - hi < delta
    h_lower   = log_barrier(delta + diff)            # hi - zi < delta
    h_term    = (h_upper + h_lower).sum()

    # Colour barrier
    c_lower   = log_barrier(c)
    c_upper   = log_barrier(1.0 - c)
    c_term    = (c_lower + c_upper).sum()

    return mu * (h_term + c_term)


def _check_convergence(prev_loss, curr_loss, tol):
    """Return True if relative change of loss < tol (paper Section 3.3)."""
    return abs(prev_loss - curr_loss) / (abs(prev_loss) + 1e-10) < tol


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

class MirrorArtSolver:
    """
    Implements the three-stage optimisation from Algorithm 1 (Wu et al. 2022).

    Args:
        Id_path, Ir_path : paths to direct and reflected target images
        base_h_grid      : [H, W] numpy heightfield (initial shape)
        grid_x, grid_y   : [H, W] numpy XY coordinates of saucer vertices
        grid_rx, grid_ry : [H, W] numpy XY coordinates of reflected vertices
        img_render_size  : resolution of rendered images (paper: 512×512)
        device           : 'cuda' or 'cpu'
    """

    def __init__(self, Id_path, Ir_path, base_h_grid,
                 grid_x, grid_y, grid_rx, grid_ry,
                 img_render_size=(512, 512),
                 device='cuda'):

        self.device          = device
        self.img_render_size = img_render_size

        # ---- Target images --------------------------------------------------
        Id_np, Ir_np, fg_d, fg_r, Ied_np, Ier_np = build_enhanced_images(Id_path, Ir_path)

        self.Id  = img_to_tensor(Id_np,  device, target_hw=img_render_size)
        self.Ir  = img_to_tensor(Ir_np,  device, target_hw=img_render_size)
        self.Ied = img_to_tensor(Ied_np, device, target_hw=img_render_size)
        self.Ier = img_to_tensor(Ier_np, device, target_hw=img_render_size)

        # ---- Geometry --------------------------------------------------------
        self.grid_x  = torch.tensor(grid_x,  dtype=torch.float32, device=device)
        self.grid_y  = torch.tensor(grid_y,  dtype=torch.float32, device=device)
        self.grid_rx = torch.tensor(grid_rx, dtype=torch.float32, device=device)
        self.grid_ry = torch.tensor(grid_ry, dtype=torch.float32, device=device)

        self.H, self.W = self.grid_x.shape
        self.N         = self.H * self.W

        self.faces = torch.tensor(
            get_grid_triangles(self.H, self.W), dtype=torch.long, device=device)

        self.z_orig   = torch.tensor(base_h_grid, dtype=torch.float32, device=device)
        self.lap_ker  = create_laplacian_kernel(device)
        self.L_z_orig = compute_laplacian(self.z_orig.view(1,1,self.H,self.W), self.lap_ker)

        # Constraint threshold (paper: δ = 0.05)
        self.delta = 0.05

        # ---- Camera matrices  (Section 4 of paper, exact values) ---------------
        # Paper quote: "The viewer/camera is placed at (0, −5.5, 5).
        #   target viewing positions for the direct view and the reflected view
        #   at (0, 0, −0.8) and (0, 0, 0.1), respectively.
        #   The viewing angles are set to 4.5° and 3° accordingly."
        # Note: z-axis is vertical (up), saucer on negative y-axis.
        cam = [0., -5.5, 5.]
        up  = [0.,  0.,  1.]

        # Direct view: FOV = 4.5°, look-at = (0, 0, -0.8)
        self.Pd = get_camera_matrix(
            cam, [0., 0., -0.8], up, fov_deg=4.5).to(device)

        # Reflected view: FOV = 3.0°, look-at = (0, 0, 0.1)
        self.Pr = get_camera_matrix(
            cam, [0., 0., 0.1], up, fov_deg=3.0).to(device)

        # ---- Renderer -------------------------------------------------------
        # Use SUM-based loss renderer (matching paper's Frobenius norm ||.||²_F)
        # for optimisation.  The render() method (inherited) is unchanged and
        # used for pre-rendering and visualisation.
        self.renderer = SoftTriangleRendererLoss(
            img_size=img_render_size, sigma=1e-5, gamma=1e-4,
            chunk_size=4096
        ).to(device)

        # ---- Adaptive weights (Section 3.5) ---------------------------------
        self.rho = self._compute_rho()
        print(f"[Solver] Compatibility ρ = {self.rho:.4f}")

        # ---- Pre-render blurred target images (Section 3.5) ----------------
        # The paper generates SoftRas-blurred versions of Id/Ir and Ied/Ier
        # at each σ level and uses THOSE as optimisation targets, eliminating
        # the sharpness-mismatch error that arises when comparing sharp inputs
        # to inherently-blurry SoftRas renders.
        print("[Solver] Pre-rendering blurred target images…")
        h0     = self.z_orig.flatten()
        pts_d0 = self._make_pts(h0)
        pts_r0 = self._make_pts(h0, rx=True)

        # Shared UV caches so each camera direction projects only once per
        # (pts, P_matrix) pair, even when called with different images/sigmas.
        uv_cache_d = {}   # keyed by id(Pd) — direct view
        uv_cache_r = {}   # keyed by id(Pr) — reflected view

        # Large σ = 1e-5  (Stage 1a)
        self.blurred_Ied_L = self._prerender_blurred(
            self.Ied, pts_d0, self.Pd, sigma=1e-5, _uv_cache=uv_cache_d)
        self.blurred_Ier_L = self._prerender_blurred(
            self.Ier, pts_r0, self.Pr, sigma=1e-5, _uv_cache=uv_cache_r)
        # Small σ = 1e-7  (Stage 1b + Stage 2)
        self.blurred_Ied_S = self._prerender_blurred(
            self.Ied, pts_d0, self.Pd, sigma=1e-7, _uv_cache=uv_cache_d)
        self.blurred_Ier_S = self._prerender_blurred(
            self.Ier, pts_r0, self.Pr, sigma=1e-7, _uv_cache=uv_cache_r)
        # For Stage 3 texturing: blurred original images
        self.blurred_Id    = self._prerender_blurred(
            self.Id,  pts_d0, self.Pd, sigma=1e-7, _uv_cache=uv_cache_d)
        self.blurred_Ir    = self._prerender_blurred(
            self.Ir,  pts_r0, self.Pr, sigma=1e-7, _uv_cache=uv_cache_r)
        print("[Solver] Pre-rendering done.")

        # ---- Results cache --------------------------------------------------
        self.h_stage1 = None
        self.c_stage1 = None
        self.h_final  = None

    # ------------------------------------------------------------------
    # Vertex assembly helpers
    # ------------------------------------------------------------------

    def _make_pts(self, h, rx=False):
        """Build [N,3] vertex tensor for direct (rx=False) or reflected (rx=True) view."""
        if rx:
            return torch.stack([self.grid_rx.flatten(),
                                 self.grid_ry.flatten(), h.flatten()], dim=1)
        else:
            return torch.stack([self.grid_x.flatten(),
                                 self.grid_y.flatten(), h.flatten()], dim=1)

    # ------------------------------------------------------------------
    # Blurred target pre-rendering  (Section 3.5)
    # ------------------------------------------------------------------

    def _back_project_img_to_verts(self, img_bchw, pts_world, P_matrix,
                                    _uv_cache=None):
        """
        Back-project an image onto mesh vertices by projecting each vertex
        into image space and sampling the pixel colour.

        img_bchw  : [1, 3, H_img, W_img]
        pts_world : [N, 3]
        P_matrix  : [4, 4]
        _uv_cache : optional dict; caches (px, py) keyed by id(P_matrix) so
                    multiple calls with the same camera and base shape avoid
                    redundant projection computations.
        Returns   : [N, 3] vertex colours in [0, 1]
        """
        H_img, W_img = self.img_render_size
        cache_key    = id(P_matrix)

        if _uv_cache is not None and cache_key in _uv_cache:
            px, py = _uv_cache[cache_key]
        else:
            with torch.no_grad():
                uv, _ = self.renderer._project(pts_world, P_matrix)   # [N, 2]
                # NDC [-1,1] → pixel indices [0, W-1] / [0, H-1]
                px = ((uv[:, 0] + 1.0) * 0.5 * (W_img - 1)).long().clamp(0, W_img - 1)
                py = ((1.0 - (uv[:, 1] + 1.0) * 0.5) * (H_img - 1)).long().clamp(0, H_img - 1)
            if _uv_cache is not None:
                _uv_cache[cache_key] = (px, py)

        with torch.no_grad():
            img_hw3 = img_bchw.squeeze(0).permute(1, 2, 0)            # [H_img, W_img, 3]
            return img_hw3[py, px]                                      # [N, 3]

    def _prerender_blurred(self, img_bchw, pts_world, P_matrix, sigma,
                           _uv_cache=None):
        """
        Generate a SoftRas-blurred version of img_bchw (Section 3.5):
          1. Back-project img pixels onto base-shape vertices (cached)
          2. Render the textured base shape through SoftRas at the given σ
        Returns [1, 3, H_img, W_img] blurred image tensor.
        """
        colors = self._back_project_img_to_verts(img_bchw, pts_world, P_matrix,
                                                  _uv_cache=_uv_cache)
        with torch.no_grad():
            rendered, _ = self.renderer.render(
                pts_world, self.faces, colors.clamp(0.0, 1.0),
                P_matrix, sigma=sigma)
        return rendered.permute(2, 0, 1).unsqueeze(0)   # [1, 3, H, W]

    # ------------------------------------------------------------------
    # Adaptive weight ρ  (Section 3.5)
    # ------------------------------------------------------------------

    def _compute_rho(self):
        """
        Paper Section 3.5 compatibility score:
          1. Back-project Ied onto base shape → render reflected view → Icr
          2. Back-project Ier onto reflected base shape → render direct view → Icd
          3. ρ = ||Ied - Icd||²_F + ||Ier - Icr||²_F  (SUM squared, paper's Frobenius norm)

        This correctly measures how conflicting the two target images are
        given the undeformed base shape.
        """
        with torch.no_grad():
            h0     = self.z_orig.flatten()
            pts_d0 = self._make_pts(h0)
            pts_r0 = self._make_pts(h0, rx=True)
            uv_cache = {}   # reuse projections within this function

            # Step 1: texture base shape with Ied → render reflected view → Icr
            c_from_Ied = self._back_project_img_to_verts(self.Ied, pts_d0, self.Pd,
                                                          _uv_cache=uv_cache)
            Icr, _     = self.renderer.render(pts_r0, self.faces,
                                              c_from_Ied.clamp(0., 1.), self.Pr)

            # Step 2: texture reflected base shape with Ier → render direct view → Icd
            c_from_Ier = self._back_project_img_to_verts(self.Ier, pts_r0, self.Pr,
                                                          _uv_cache=uv_cache)
            Icd, _     = self.renderer.render(pts_d0, self.faces,
                                              c_from_Ier.clamp(0., 1.), self.Pd)

            # ρ = SUM squared differences (paper's ||.||²_F Frobenius norm)
            Ied_hw3 = self.Ied.squeeze(0).permute(1, 2, 0)
            Ier_hw3 = self.Ier.squeeze(0).permute(1, 2, 0)
            rho = ((Ied_hw3 - Icd) ** 2).sum().item() + \
                  ((Ier_hw3 - Icr) ** 2).sum().item()

        if rho < 1e-6:
            rho = 1.0
        return rho

    # ------------------------------------------------------------------
    # Barrier loss
    # ------------------------------------------------------------------

    def _barrier(self, h, c):
        return barrier_loss(h, c, self.z_orig, self.delta)

    # ------------------------------------------------------------------
    # Stage 1 – Black-White Enhancement (two-step σ)
    # ------------------------------------------------------------------

    def solve_stage1(self, iters_large_sigma=300, iters_small_sigma=300,
                     convergence_tol=1e-4):
        """
        Minimise Eq. 8:  E_new_visual + w*E_deform + E_barrier
        using Adamax with two-step σ as in Section 3.5.

        Uses pre-rendered blurred target images (Section 3.5) instead of the
        sharp input images, eliminating the sharpness-mismatch error.

        Convergence: terminates early when relative change of E < convergence_tol
        (paper Section 3.3: threshold 10^-4).
        """
        h = self.z_orig.clone().requires_grad_(True)
        c = torch.full((self.N, 3), 0.5, device=self.device, requires_grad=True)

        opt = optim.Adamax([h, c], lr=0.01)

        def one_iter(sigma, w_deform, target_d, target_r):
            opt.zero_grad()
            pts_d = self._make_pts(h)
            pts_r = self._make_pts(h, rx=True)

            loss_vis    = self.renderer(pts_d, self.faces, c, target_d, self.Pd, sigma=sigma) \
                        + self.renderer(pts_r, self.faces, c, target_r, self.Pr, sigma=sigma)

            L_h         = compute_laplacian(h.view(1,1,self.H,self.W), self.lap_ker)
            loss_deform = ((L_h - self.L_z_orig) ** 2).sum()

            loss_bar    = self._barrier(h, c)
            loss        = loss_vis + w_deform * loss_deform + loss_bar
            loss.backward()

            # Backtracking line search: clamp h to satisfy hard constraint |h-z|<δ
            with torch.no_grad():
                opt.step()
                h.clamp_(self.z_orig - self.delta + 1e-6,
                         self.z_orig + self.delta - 1e-6)
                c.clamp_(1e-6, 1.0 - 1e-6)

            return loss.item(), loss_vis.item()

        # Step 1: large σ = 1e-5,  w = 0.08·ρ  — use blurred large-σ targets
        w1 = 0.08 * self.rho
        print(f"[Stage 1a] σ=1e-5  w={w1:.4f}  iters≤{iters_large_sigma}")
        prev_loss = float('inf')
        for i in range(iters_large_sigma):
            total, vis = one_iter(1e-5, w1, self.blurred_Ied_L, self.blurred_Ier_L)
            if i % 50 == 0:
                print(f"  iter {i:4d}  total={total:.4f}  vis={vis:.4f}")
            # Convergence check (paper Section 3.3)
            if i > 20 and _check_convergence(prev_loss, total, convergence_tol):
                print(f"  converged at iter {i} (rel_change < {convergence_tol})")
                break
            prev_loss = total

        # Step 2: small σ = 1e-7,  w = 0.2·ρ  — use blurred small-σ targets
        w2 = 0.2 * self.rho
        print(f"[Stage 1b] σ=1e-7  w={w2:.4f}  iters≤{iters_small_sigma}")
        prev_loss = float('inf')
        for i in range(iters_small_sigma):
            total, vis = one_iter(1e-7, w2, self.blurred_Ied_S, self.blurred_Ier_S)
            if i % 50 == 0:
                print(f"  iter {i:4d}  total={total:.4f}  vis={vis:.4f}")
            if i > 20 and _check_convergence(prev_loss, total, convergence_tol):
                print(f"  converged at iter {i} (rel_change < {convergence_tol})")
                break
            prev_loss = total

        self.h_stage1 = h.detach()
        self.c_stage1 = c.detach().clamp(0., 1.)
        return self.h_stage1, self.c_stage1

    # ------------------------------------------------------------------
    # Stage 2 – Sparse Spike Strategy  (Eq. 10 + proximal gradient)
    # ------------------------------------------------------------------

    def solve_stage2(self, iterations=200, convergence_tol=1e-4):
        """
        Minimise Ẽ = E_new_visual + λ·E_sparse + E_barrier
        using Adamax for c and the proximal gradient method for h.
        λ = 6e-5·ρ  (Section 3.5, small σ = 1e-7)

        Uses pre-rendered blurred targets (small σ) for the visual term.
        Terminates when relative change < convergence_tol (paper: 10^-4).
        """
        assert self.h_stage1 is not None, "Run solve_stage1 first."

        h  = self.h_stage1.clone().requires_grad_(True)
        c  = self.c_stage1.clone().requires_grad_(True)
        he = self.h_stage1.detach()   # reference for L1 term

        lmbda = 6e-5 * self.rho
        alpha = 0.2   # proximal step size (paper: α = 0.2)
        sigma = 1e-7

        opt = optim.Adamax([h, c], lr=0.01)

        print(f"[Stage 2] σ=1e-7  λ={lmbda:.6f}  iters≤{iterations}")
        prev_loss = float('inf')
        for i in range(iterations):
            opt.zero_grad()
            pts_d = self._make_pts(h)
            pts_r = self._make_pts(h, rx=True)

            phi  = self.renderer(pts_d, self.faces, c,
                                 self.blurred_Ied_S, self.Pd, sigma=sigma) \
                 + self.renderer(pts_r, self.faces, c,
                                 self.blurred_Ier_S, self.Pr, sigma=sigma) \
                 + self._barrier(h, c)
            phi.backward()

            with torch.no_grad():
                opt.step()
                # Proximal mapping: soft-threshold h w.r.t. he  (Eq. 15)
                d       = h - he
                thresh  = lmbda * alpha
                h_new   = torch.where(d >  thresh, h - thresh,
                          torch.where(d < -thresh, h + thresh, he))
                h.copy_(h_new)
                # Enforce hard constraints
                h.clamp_(self.z_orig - self.delta + 1e-6,
                         self.z_orig + self.delta - 1e-6)
                c.clamp_(1e-6, 1.0 - 1e-6)

            loss_val = phi.item()
            if i % 50 == 0:
                print(f"  iter {i:4d}  phi={loss_val:.4f}")
            if i > 20 and _check_convergence(prev_loss, loss_val, convergence_tol):
                print(f"  converged at iter {i} (rel_change < {convergence_tol})")
                break
            prev_loss = loss_val

        self.h_final = h.detach()
        self.c_stage2 = c.detach().clamp(0., 1.)
        return self.h_final, self.c_stage2

    # ------------------------------------------------------------------
    # Stage 3 – Texturing  (Section 3.4.2)
    # ------------------------------------------------------------------

    def solve_stage3_texturing(self, iterations=300, convergence_tol=1e-4):
        """
        Fix the geometry h_final; optimise per-face colours c to match the
        original images Id and Ir.

        Uses pre-rendered blurred Id/Ir as targets (Section 3.5) so that
        the optimiser matches the reachable colour range of SoftRas.
        Terminates when relative change < convergence_tol.
        """
        assert self.h_final is not None, "Run solve_stage2 first."

        h  = self.h_final.clone()   # fixed
        c  = self.c_stage2.clone().requires_grad_(True)

        opt = optim.Adamax([c], lr=0.02)
        sigma = 1e-7

        print(f"[Stage 3] Texturing  iters≤{iterations}")
        prev_loss = float('inf')
        for i in range(iterations):
            opt.zero_grad()
            pts_d = self._make_pts(h)
            pts_r = self._make_pts(h, rx=True)

            loss  = self.renderer(pts_d, self.faces, c,
                                  self.blurred_Id, self.Pd, sigma=sigma) \
                  + self.renderer(pts_r, self.faces, c,
                                  self.blurred_Ir, self.Pr, sigma=sigma)
            # Colour-range barrier (SUM-based, with small coefficient)
            loss  = loss + 1e-4 * (torch.log(c.clamp(min=1e-6)) * -1 +
                            torch.log((1.0 - c).clamp(min=1e-6)) * -1).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                c.clamp_(0., 1.)

            loss_val = loss.item()
            if i % 50 == 0:
                print(f"  iter {i:4d}  loss={loss_val:.4f}")
            if i > 20 and _check_convergence(prev_loss, loss_val, convergence_tol):
                print(f"  converged at iter {i} (rel_change < {convergence_tol})")
                break
            prev_loss = loss_val

        c_final = c.detach().clamp(0., 1.)
        H, W = self.H, self.W
        return (h.cpu().numpy().reshape(H, W),
                c_final.cpu().numpy().reshape(H, W, 3))

    # ------------------------------------------------------------------
    # Convenience: render final result images
    # ------------------------------------------------------------------

    def render_views(self, h_np, c_np):
        """Render direct and reflected views from the final result. Returns (img_d, img_r) as uint8 [H,W,3]."""
        h = torch.tensor(h_np, dtype=torch.float32, device=self.device).flatten()
        c = torch.tensor(c_np, dtype=torch.float32, device=self.device).reshape(-1, 3)

        with torch.no_grad():
            img_d, _ = self.renderer.render(self._make_pts(h),        self.faces, c, self.Pd)
            img_r, _ = self.renderer.render(self._make_pts(h, rx=True), self.faces, c, self.Pr)

        to_uint8 = lambda t: (t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return to_uint8(img_d), to_uint8(img_r)
