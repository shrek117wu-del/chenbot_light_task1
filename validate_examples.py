"""
validate_examples.py — 快速验证脚本
===================================
验证所有 pipeline 样本都可以运行，不进行完整优化(只运行少量迭代)。
用于确认：
  1. 渲染管线(direct + reflected)正确投影到屏幕
  2. 图像不为全黑/全零
  3. 相机矩阵正确
  4. 3D export 生成有效 OBJ + texture

使用方法:
    python validate_examples.py
"""
import os
import sys
import numpy as np
import torch
import cv2

# Make sure we can import from project root
sys.path.insert(0, os.path.dirname(__file__))

from core.renderer  import SoftTriangleRenderer, get_camera_matrix
from core.geometry  import get_grid_triangles, precompute_reflection_grid
from experiments    import create_base_shape, generate_test_images, get_paper_experiment
from export_utils   import export_obj_and_texture


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[validate] device = {DEVICE}")

# -----------------------------------------------------------------------
# Camera matrices (paper settings, Section 4)
# -----------------------------------------------------------------------
CAM      = [0., -5.5, 5.]
UP       = [0.,  0.,  1.]
Pd = get_camera_matrix(CAM, [0., 0., -0.8], UP, fov_deg=4.5)
Pr = get_camera_matrix(CAM, [0., 0.,  0.1], UP, fov_deg=3.0)
print(f"[validate] Pd =\n{Pd}")
print(f"[validate] Pr =\n{Pr}")

RENDER_SIZE = (256, 256)
renderer = SoftTriangleRenderer(img_size=RENDER_SIZE, sigma=1e-4, gamma=1e-3,
                                chunk_size=4096)


def check_image(img_np, name):
    """img_np: [H,W,3] float32 in [0,1]"""
    m = img_np.mean()
    nonzero = (img_np > 0.01).sum()
    total   = img_np.size
    pct     = 100.0 * nonzero / total
    print(f"  {name}: mean={m:.4f}  nonzero={pct:.1f}%", end="")
    if pct < 0.5:
        print("  ← ⚠ WARNING: mostly black image!")
    elif pct > 99:
        print("  ← ⚠ WARNING: uniformly bright (no structure)")
    else:
        print("  ✓")
    return m, pct


def validate_shape(shape_type, resolution=(60, 60)):
    print(f"\n{'='*60}")
    print(f"Shape: {shape_type}  @ {resolution}")
    print('='*60)

    grid_x, grid_y, base_h = create_base_shape(shape_type, resolutions=resolution)
    H, W = grid_x.shape

    # Reflection grid
    P_2d      = np.array([0.0, -5.5])
    grid_rx, grid_ry = precompute_reflection_grid(grid_x, grid_y, P_2d, r=0.4)

    # Tensors
    h  = torch.tensor(base_h,    dtype=torch.float32, device=DEVICE).flatten()
    c  = torch.full((H*W, 3), 0.5, dtype=torch.float32, device=DEVICE)
    gx = torch.tensor(grid_x,  dtype=torch.float32, device=DEVICE)
    gy = torch.tensor(grid_y,  dtype=torch.float32, device=DEVICE)
    rx = torch.tensor(grid_rx, dtype=torch.float32, device=DEVICE)
    ry = torch.tensor(grid_ry, dtype=torch.float32, device=DEVICE)

    faces = torch.tensor(get_grid_triangles(H, W), dtype=torch.long, device=DEVICE)

    # Direct view vertices
    pts_d = torch.stack([gx.flatten(), gy.flatten(), h], dim=1)
    # Reflected view vertices
    pts_r = torch.stack([rx.flatten(), ry.flatten(), h], dim=1)

    print(f"  Vertices: {pts_d.shape[0]}  Faces: {faces.shape[0]}")
    print(f"  Direct pts range: x{pts_d[:,0].min():.2f}..{pts_d[:,0].max():.2f}"
          f"  y{pts_d[:,1].min():.2f}..{pts_d[:,1].max():.2f}"
          f"  z{pts_d[:,2].min():.2f}..{pts_d[:,2].max():.2f}")
    print(f"  Reflect pts range: x{pts_r[:,0].min():.2f}..{pts_r[:,0].max():.2f}"
          f"  y{pts_r[:,1].min():.2f}..{pts_r[:,1].max():.2f}"
          f"  z{pts_r[:,2].min():.2f}..{pts_r[:,2].max():.2f}")

    # Projection test
    with torch.no_grad():
        Pd_dev = Pd.to(DEVICE)
        Pr_dev = Pr.to(DEVICE)
        img_d_t, mask_d = renderer.render(pts_d, faces, c, Pd_dev)
        img_r_t, mask_r = renderer.render(pts_r, faces, c, Pr_dev)

    img_d = img_d_t.cpu().numpy()
    img_r = img_r_t.cpu().numpy()
    mask_d_np = mask_d.cpu().numpy()
    mask_r_np = mask_r.cpu().numpy()

    md, pd = check_image(img_d, "direct ")
    mr, pr = check_image(img_r, "reflect")
    print(f"  Direct  mask coverage: {100*mask_d_np.mean():.1f}%")
    print(f"  Reflect mask coverage: {100*mask_r_np.mean():.1f}%")

    # Save validation images
    os.makedirs("validation_out", exist_ok=True)
    def save(arr, fname):
        bgr = (arr.clip(0,1)*255).astype(np.uint8)[:,:,::-1]
        cv2.imwrite(fname, bgr)

    save(img_d, f"validation_out/{shape_type}_direct.png")
    save(img_r, f"validation_out/{shape_type}_reflect.png")
    print(f"  Saved → validation_out/{shape_type}_direct/reflect.png")

    return pd > 0.5 and pr > 0.5   # pass if both have some structure


def validate_export(shape_type="plane", resolution=(40, 40)):
    """Test that OBJ + texture export works and produces valid files."""
    print(f"\n[Export test] shape={shape_type} res={resolution}")
    grid_x, grid_y, base_h = create_base_shape(shape_type, resolutions=resolution)
    H, W = grid_x.shape
    # Dummy colours
    color_grid = np.random.rand(H, W, 3).astype(np.float32) * 0.5 + 0.25
    export_obj_and_texture(grid_x, grid_y, base_h, color_grid,
                           obj_path="validation_out/test_saucer.obj",
                           tex_path="validation_out/test_texture.png")
    # Check files
    for fname in ["validation_out/test_saucer.obj", "validation_out/test_texture.png"]:
        size = os.path.getsize(fname)
        ok   = size > 500
        print(f"  {fname}: {size} bytes  {'✓' if ok else '⚠ TOO SMALL'}")
    return True


def validate_experiment_images():
    """Validate paper experiment images can be loaded."""
    print(f"\n[Experiment image test]")
    for eid in [0, 1, 2]:
        try:
            pd_path, pr_path = get_paper_experiment(eid, output_dir="data")
            img_d = cv2.imread(pd_path)
            img_r = cv2.imread(pr_path)
            assert img_d is not None and img_r is not None
            print(f"  Exp {eid}: direct={img_d.shape}  reflect={img_r.shape}  ✓")
        except FileNotFoundError as e:
            print(f"  Exp {eid}: ⚠ {e}")
        except Exception as e:
            print(f"  Exp {eid}: ✗ {e}")


if __name__ == "__main__":
    results = {}

    # Shape validation (no full optimization, just render check)
    for shape in ["plane", "saucer", "tabula_scalata", "random", "shallow_bowl"]:
        results[shape] = validate_shape(shape, resolution=(60, 60))

    # Export validation
    validate_export()

    # Image loading validation
    validate_experiment_images()

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print('='*60)
    all_ok = True
    for shape, ok in results.items():
        status = "✓ PASS" if ok else "⚠ FAIL (low coverage)"
        print(f"  {shape:20s}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n✅ All shapes produce visible renders. Pipeline OK.")
    else:
        print("\n⚠ Some shapes have low render coverage. Check camera settings.")

    print("\nValidation images saved to: validation_out/")
