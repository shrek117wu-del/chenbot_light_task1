import torch
import cv2
import numpy as np
from core.renderer import SoftTriangleRendererLoss, get_camera_matrix
from core.geometry import get_grid_triangles
from experiments import create_base_shape

device = 'cpu' # Use CPU for local inspection if CUDA isn't working
print(f"Using device: {device}")

H, W = 100, 100
grid_x, grid_y, base_h = create_base_shape("saucer", resolutions=(H, W))
vertices = torch.stack([torch.tensor(grid_x).flatten(), 
                        torch.tensor(grid_y).flatten(), 
                        torch.tensor(base_h).flatten()], dim=1).float().to(device)
faces = torch.tensor(get_grid_triangles(H, W), dtype=torch.long, device=device)
colors = torch.full((H*W, 3), 0.5, device=device)

# Simple projection
camera_pos = [0, -5.5, 5.0]
P = get_camera_matrix(camera_pos, [0, 0, -0.8], [0, 0, 1], fov_deg=4.5).to(device)

renderer = SoftTriangleRendererLoss(img_size=(256, 256), sigma=1e-3).to(device)

# We need a dummy target to run forward and get mask/output
target = torch.zeros(1, 3, 256, 256, device=device)

# To get the actual rendered image, we use the return_img=True option
img, mask = renderer(vertices, faces, colors, target, P, return_img=True)

print(f"Rendered image shape: {img.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Mask sum (pixels covered): {mask.sum().item()}")

if mask.sum() > 0:
    print("Saucer is visible!")
    # Save image
    img_np = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite("C:\\Users\\shrek\\.gemini\\antigravity\\brain\\869ac1f2-b56f-4945-a9f6-7be0afb51b65\\scratch\\debug_render.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
else:
    print("Saucer is NOT visible. Check camera or projection.")
    # Check screen coordinates
    # homogeneous to screen
    pts_hom = torch.cat([vertices, torch.ones(vertices.shape[0], 1, device=device)], dim=1)
    screen_pts = pts_hom @ P.T
    z = screen_pts[:, 2]
    u = screen_pts[:, 0] / z
    v = screen_pts[:, 1] / z
    print(f"U min/max: {u.min().item():.4f}, {u.max().item():.4f}")
    print(f"V min/max: {v.min().item():.4f}, {v.max().item():.4f}")
    print(f"Z min/max: {z.min().item():.4f}, {z.max().item():.4f}")
