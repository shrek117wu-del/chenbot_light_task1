# Walkthrough - Computational Mirror Art Optimization

I have successfully upgraded the rendering pipeline and optimization strategy to match the methodology described in the "Computational Mirror Cup and Saucer Art" paper.

## Key Improvements

### 1. Triangle-based Differentiable Renderer
The system now uses a **Soft Rasterizer (SoftRas)** style triangle-based renderer instead of the previous point-based splatting approach. 
- **Mesh Generation**: Added `get_grid_triangles` to [geometry.py](file:///d:/WorkingProject/Antigravity_workspace/chenbot_light_task1_working/core/geometry.py) to convert the heightfield into a triangle mesh.
- **Differentiable Rasterization**: Implemented `SoftTriangleRendererLoss` in [renderer.py](file:///d:/WorkingProject/Antigravity_workspace/chenbot_light_task1_working/core/renderer.py). It uses signed distance functions to triangles for smooth, differentiable gradients.
- **OOM Fix**: Implemented **Pixel Chunking** to handle high-resolution rendering without exceeding GPU memory.

### 2. Adaptive Weighting Management
Implemented the compatibility score ($\rho$) calculation from Section 3.5 of the paper.
- **Dynamic Parameters**: The weights for deformation ($w$) and sparsity ($\lambda$) are now automatically calculated based on the input images' compatibility, ensuring balanced optimization across different target pairs.

### 3. Two-Step $\sigma$ Procedure
Stage 1 (Black-White Enhancement) now follows the two-step refinement process:
- **Step 1**: Initial optimization with a relatively large $\sigma$ ($10^{-5}$) for global structure.
- **Step 2**: Refinement with a small $\sigma$ ($10^{-7}$) to capture sharp details.

## Verification Results

The pipeline was verified on CPU (as CUDA was unavailable in the current environment) at a reduced resolution ($40 \times 40$). The system successfully executed:
1.  **Precomputation**: Reflection mapping grid generation.
2.  **Stage 1**: Two-step Black-White enhancement.
3.  **Stage 2**: Proximal gradient optimization with adaptive $\lambda$.
4.  **Stage 3**: Surface texturing.

> [!NOTE]
> For production-quality results, it is recommended to run at $(125, 125)$ resolution or higher on a CUDA-enabled GPU.

## Files Modified

- [renderer.py](file:///d:/WorkingProject/Antigravity_workspace/chenbot_light_task1_working/core/renderer.py): New Soft Triangle Renderer.
- [solver.py](file:///d:/WorkingProject/Antigravity_workspace/chenbot_light_task1_working/core/solver.py): Adaptive weights and two-step $\sigma$ logic.
- [geometry.py](file:///d:/WorkingProject/Antigravity_workspace/chenbot_light_task1_working/core/geometry.py): Mesh generation utilities.
- [main.py](file:///d:/WorkingProject/Antigravity_workspace/chenbot_light_task1_working/main.py): Pipeline integration.
