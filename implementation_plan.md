# Goal Description

The goal is to implement the "Computational Mirror Cup and Saucer Art" (ACM TOG 2022) algorithm in Python. This involves creating a robust pipeline that takes a target direct view image, a target reflection view image, and a base saucer shape, and generates an optimized textured saucer mesh. The system will deform the mesh to ensure both views are correctly perceived and will apply the appropriate texture mapping. Additionally, we will reproduce the paper's experiments across various basic shapes and provide a 3D visualization tool to showcase the final physical setup (textured saucer + mirror cylinder).

## User Review Required

> [!IMPORTANT]
> **Differentiable Renderer Choice:** The original paper uses *SoftRasterizer (SoftRas)*, an older PyTorch-based differentiable renderer. Compiling SoftRas or modern alternatives like *PyTorch3D* from source on Windows can be extremely brittle due to C++/CUDA requirements. 
> *Proposal:* I will first attempt to find a pre-compiled wheel for PyTorch3D/SoftRas or construct a simplified pure-PyTorch differentiable splatting/rasterizer tailored specifically for heightfields to guarantee it runs smoothly on your Windows machine without painful environment setups. Do you approve this fallback approach if standard libraries fail to compile?

> [!WARNING]
> **Performance vs. Quality for Experiments:** The paper notes each optimization run takes up to 30 minutes at a $150 \times 150$ resolution. For iterating and running the extensive experiments you requested, I propose starting testing at a lower resolution (e.g., $75 \times 75$) to confirm correctness before running the high-resolution final results.

> [!TIP]
> **3D Viewer Library:** I plan to build the 3D viewer using `pyvista` or by exporting a portable `HTML/WebGL (Three.js)` viewer. `pyvista` is excellent for Python-native interactive 3D, while the HTML approach requires no local 3D drivers. Let me know if you prefer one over the other.

## Proposed Changes

---

### Core Algorithm Implementation (`core/solver.py`)
This will be the mathematical heart of the framework.

#### [NEW] `core/solver.py`
- Setup PyTorch tensors to represent the saucer heightfield $h$ and color $c$.
- Implement the discrete Laplace operator corresponding to Equation (3) for the `Shape-preservation term`.
- Implement `Reflected Shapes` logic (Section 3.2): A PyTorch function converting the saucer vertices `Sd` into the reflected vertices `Sr` mathematically mimicking the cylindrical mirror.
- Implement the Two-stage optimization loop:
  - **Stage 1 (Black-White Enhancement):** Extract foreground masks and perform the high-contrast deformation optimization using `Adamax`.
  - **Stage 2 (Sparse Spike Strategy):** Apply proximal gradient descent with L1 norm penalty to refine the mesh.
- Implement the `Texturing` procedure to reverse-project target images onto the output geometry.

---

### Utilities and Rendering (`core/renderer.py`)

#### [NEW] `core/renderer.py`
- Differentiable rasterization routines.
- Viewport and perspective camera configuration mapping to the paper's physical coordinates `(camera position at (0, -5.5, 5))`.
- Loss functions mapping to equations (2), (5), and (6).

---

### Experiments Reproduction (`experiments.py`)

#### [NEW] `experiments.py`
- Implement the base shape generators:
  - Plane
  - Random Heightfield
  - Tabula Scalata (wave-like)
  - Typical Saucer
- A runner script to loop through these shapes and apply the solver to dummy target images (e.g., character 'A' and 'B').

---

### Main Interface & 3D Viewer (`main.py`, `viewer.py`)

#### [NEW] `main.py`
- An easy-to-use unified python API that accepts:
  - `target_direct_img_path`
  - `target_reflect_img_path`
  - `base_shape_type`
- Orchestrates the solver and exports the `.obj` and `.png` texture maps.

#### [NEW] `viewer.py`
- Loads the generated textured saucer.
- Programmatically injects a perfectly reflective cylinder (the cup).
- Sets up two camera views (one for direct, one aimed at the mirror) so that you can verify the results interactively.

## Open Questions

1. **Target Images:** For the experiment reproduction, I plan to auto-generate some simple geometric shapes or text images (e.g., a Star and a Circle) to validate the algorithm. Should I use specific images you have, or are generated shapes fine for testing?
2. **CUDA Availability:** Do you have an NVIDIA GPU set up with CUDA enabled on this machine that PyTorch can utilize? The optimization will be prohibitively slow on CPU alone.

## Verification Plan

### Automated Tests
- The core reflection coordinate mapping will be unit-tested against the analytic solutions described in the paper's Appendix.

### Manual Verification
- We will run `main.py` on a $75 \times 75$ resolution plane grid and physically visualize the output via `viewer.py` to ensure the anamorphosis is accurately hiding and revealing the proper images.
