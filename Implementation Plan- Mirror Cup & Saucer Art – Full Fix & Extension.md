# Implementation Plan: Mirror Cup & Saucer Art – Full Fix & Extension

## Problem Analysis

After thorough code review and paper reading, I identified **5 critical bugs** and **3 missing features**.

---

## Critical Bugs

### Bug 1: Renderer Produces Zero Loss (Core Rendering Failure)

**Root cause**: The projection matrix `get_camera_matrix` produces **homogeneous clip-space coordinates**, not normalized-device-coordinates (NDC). The division `u = screen_pts[:,0] / z_safe` uses column 2 (`z`) as divisor but the matrix includes a perspective `w` term in column 3. The correct perspective divide is `/ screen_pts[:,3]` (the w-component), not `/ z_safe`.

This means all projected `u,v` values are wildly out of `[-1,1]`, so the triangle bounding-box filter rejects everything or `dist_min/sigma` overflows → `sigmoid → 0` → zero loss.

**Fix**: Correct the perspective divide to use `w = screen_pts[:,3]` and use column 2 for depth `z`.

### Bug 2: Camera Matrix is Incorrect for the Paper's Setup

The paper places the camera at `P=(0, -5.5, 5)`, looks at the saucer center/cylinder. The current `get_camera_matrix` uses `up=[0,0,1]` (z-up) and builds a standard OpenGL look-at. But the cross-product `forward × up` fails when forward is already near [0,0,1]. The result is a degenerate right vector. **Fix**: Use `up=[0,1,0]` for the camera up direction in world space.

### Bug 3: Viewer – OBJ Loads Without Texture (Silent MTL Failure)

The `OBJLoader` in Three.js ships **without** `MTLLoader` integration when used standalone. The code calls `objLoader.load('saucer.obj', ...)` and then replaces the material, but the `map: texture` on `MeshBasicMaterial` requires the UV coordinates to be valid. The OBJ vertex/UV layout has a duplicate vertex write (`v` then `vt` in same loop) but they share indices – correct but the OBJ face format `f v/vt` indices need to be verified.

**Fix**: Use `MTLLoader` + `OBJLoader` in Three.js, or load OBJ and apply UV map correctly. Also fix the export: the OBJ currently writes `v` and `vt` interleaved (fine) but the resulting texture is only `128×128` from the test images (too tiny and not real colors). The **real fix** is to use a proper texture atlas from the two input images as described in the paper.

### Bug 4: Viewer – Mirror Cylinder Shows No Reflection

The `CubeCamera` is positioned **inside** the cylinder mesh (`mirrorCylinder.position=(0,1,0)`, `cubeCamera.position=(0,0.5,0)`). When the cylinder hides itself during update, the saucer is at `y∈[-1.5,-0.5]` in Three.js coords but the saucer OBJ is positioned at the export coord `v_y = z_height` which is near 0, and `v_z = -y` which is around `[0.5, 1.5]`. The mirror cylinder must be re-positioned to `(0, 0, 0)` matching the paper's setup (cylinder at origin), and the saucer placed at `z∈[0.5,1.5]` in Three.js.

**Fix**: Reconcile coordinate systems between paper, Python optimizer, OBJ export, and Three.js scene graph.

### Bug 5: Stage 3 Texturing Uses Wrong Color Space

Stage 3 (`solve_stage3_texturing`) uses a new random `c` initialized to gray `0.5`, discarding the optimized `c_stage1`. The paper says texturing fixes the geometry and **re-optimizes color to match the original images** – this is mostly correct. However the `barrier_loss` with `h` (fixed tensor, no grad) passes a non-grad tensor to `barrier_loss` which only checks `c` penalty – this silently works but `h_penalty` computes `abs(h - z_orig)` but `h = h_final.clone()` (no grad), so that's fine. The real issue is the texturing should use **UV-mapped texture coordinates** as per Section 3.4.2, not per-face flat colors.

---

## Missing Features (Paper Fidelity)

### Missing 1: Proper ρ (Compatibility Score) Computation

Current `compute_rho()` renders with gray `c=0.5`, which gives loss≈0 (sky blue background ≈ saucer average). The paper says: texture S temporarily with `Ied` by reverse projection, render it to get `Icd`, then compute `ρ = ||Ied - Icd||² + ||Ier - Icr||²`. 

**Fix**: Implement proper reverse-projection texturing to set `c` before computing `ρ`.

### Missing 2: Blurred Target Image Pre-rendering

The paper (Section 3.5) says target images should first be rendered through SoftRas at the current σ to generate `Ĩd` and `Ĩr` (blurred versions), then use those as targets. This compensates for the blur introduced by SoftRas.

### Missing 3: Real Experiment Data from PDF

The `data/` folder has `1-direct.png`, `1-reflect.png`, `2-direct.png`, `2-reflect.png` from the paper. The pipeline should be runnable on these real images.

---

## Proposed Changes

### [MODIFY] core/renderer.py
- Fix perspective divide: use `w = pts_hom @ P.T` then `u = col0/col3`, `v = col1/col3`, `depth = col2/col3`
- Remove duplicate `get_camera_matrix` function
- Add `render_image()` standalone function for visualization

### [MODIFY] core/geometry.py  
- Fix camera up vector handling
- Add `compute_texcoords_for_face()` for proper UV assignment in texturing stage

### [MODIFY] core/solver.py
- Fix `compute_rho()` using reverse-projection texturing
- Fix `solve_stage3_texturing()` to use proper per-face UV coordinates (project each face to image space)
- Fix `barrier_loss()` to use **log barrier** (as paper says), not ReLU

### [MODIFY] experiments.py
- Add support for loading real images from `data/` directory
- Add more base shapes: plane, random heightfield, tabula scalata (already there, needs more testing)

### [MODIFY] main.py
- Add CLI argument parsing or experiment selector
- Add proper validation mode that runs all paper experiments

### [REWRITE] export_utils.py
- Fix OBJ UV mapping: use proper texture atlas combining Id and Ir
- Add cylinder OBJ export for the 3D viewer

### [REWRITE] viewer.html
- Fix coordinate system alignment between Python and Three.js
- Fix MTLLoader + texture pipeline
- Fix cylinder position and CubeCamera placement for actual reflections
- Add orbit controls labels and paper result display

### [MODIFY] viewer.py
- Pass more metadata to the viewer (camera position, cylinder radius)

---

## Execution Order

1. Fix `renderer.py` (perspective divide + duplicate function) – **unblocks everything**
2. Fix `solver.py` (rho, barrier, texturing)
3. Fix `experiments.py` (real images)
4. Fix `export_utils.py` (texture atlas + cylinder)
5. Fix `viewer.html` (coordinates + reflection)
6. Test with `data/1-direct.png` + `data/1-reflect.png`
