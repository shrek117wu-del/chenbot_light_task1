import torch
import time
from core.renderer import SoftTriangleRendererLoss, get_camera_matrix
from core.geometry import get_grid_triangles

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

H, W = 100, 100
vertices = torch.randn(H*W, 3, device=device)
faces = torch.tensor(get_grid_triangles(H, W), dtype=torch.long, device=device)
colors = torch.rand(H*W, 3, device=device)
target = torch.rand(1, 3, 256, 256, device=device)
P = torch.eye(4, device=device)

renderer = SoftTriangleRendererLoss(img_size=(256, 256)).to(device)

print("Starting profile...")
start = time.time()
loss = renderer(vertices, faces, colors, target, P)
print(f"Forward pass time: {time.time() - start:.4f}s")
print(f"Loss: {loss.item():.4f}")

start = time.time()
loss.backward()
print(f"Backward pass time: {time.time() - start:.4f}s")
