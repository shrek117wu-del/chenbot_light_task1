# pro/ – Mirror Cup & Saucer Art · Photo-Model Edition

This directory contains an improved implementation of the mirror cup & saucer
art optimisation that uses the **actual photo OBJ models** as the starting
geometry instead of procedurally generated heightfields.

---

## Directory layout

```
pro/
├── __init__.py          Python package marker
├── main.py              Main CLI entry point
├── mesh_io.py           OBJ loading, saucer normalisation, heightfield conversion
├── viewer.html          Three.js interactive 3D viewer
└── README.md            This file
```

---

## Key improvements over the original `main.py`

| Aspect | Original `main.py` | `pro/main.py` |
|---|---|---|
| Saucer geometry | Procedural (parabola, flat, …) | Loaded from `photo_saucer_*.obj` |
| Cup geometry | Procedural cylinder only | Loaded from `photo_cup_*.obj` |
| Cup radius | Hard-coded 0.4 | Measured from the cup OBJ |
| Starting shape | Simple, far from real saucer | Actual concentric-ring profile |
| Viewer | Generic | Side-by-side comparison with targets |

---

## Usage (from the repository root)

```bash
# Default: wavy saucer, straight cup, paper experiment 1
python -m pro.main

# Choose a different saucer model and cup model
python -m pro.main --saucer stepped   --cup conical

# Provide your own target images
python -m pro.main --direct  path/to/direct.png \
                   --reflect path/to/reflect.png

# Quick preview (lower resolution, fewer iterations)
python -m pro.main --res 80 --render_size 256 \
                   --iters1 100 --iters2 50 --iters3 80

# Paper-quality run (slow, best results)
python -m pro.main --res 150 --render_size 512 \
                   --iters1 500 --iters2 300 --iters3 300
```

### Available saucer models

| `--saucer` | File |
|---|---|
| `wavy` *(default)* | `photo_saucer_wavy.obj` |
| `stepped` | `photo_saucer_stepped.obj` |
| `smooth_rim` | `photo_saucer_smooth_rim.obj` |

### Available cup models

| `--cup` | File |
|---|---|
| `straight` *(default)* | `photo_cup_straight.obj` |
| `conical` | `photo_cup_conical.obj` |

---

## Outputs

All outputs go to `pro/out/` (override with `--out_dir`):

| File | Description |
|---|---|
| `out_direct.png` | Rendered direct view after optimisation |
| `out_reflect.png` | Rendered reflected view after optimisation |
| `saucer.obj` | Optimised saucer mesh (fine-tuned geometry + UV) |
| `saucer.mtl` | Material file referencing texture |
| `texture.png` | Learnt per-vertex colour texture |
| `stage1_direct.png` | Direct view after Stage 1 |
| `stage1_reflect.png` | Reflected view after Stage 1 |
| `stage2_direct.png` | Direct view after Stage 2 |
| `stage2_reflect.png` | Reflected view after Stage 2 |
| `viewer.html` | Three.js interactive 3D viewer |

---

## 3D viewer

After the optimisation completes, serve the output directory and open
`viewer.html`:

```bash
cd pro/out
python -m http.server 8080
# then open http://localhost:8080/viewer.html
```

The viewer shows:
* The optimised textured saucer mesh in 3D
* A real-time reflective cup at the centre
* Side panels with the rendered direct / reflected views

**Keyboard shortcuts:**
| Key | Action |
|---|---|
| R | Reset camera |
| P | Paper camera position |
| T | Top-down view |
| W | Wireframe toggle |
| I | Toggle image panels |

---

## Algorithm (identical to the paper)

1. **Stage 1 – Black-White Enhancement** (two-step σ)  
   Optimises height *h* and colour *c* using enhanced binary targets `Ied`/`Ier`.

2. **Stage 2 – Sparse Spike Strategy** (proximal gradient)  
   Adds an L1 sparsity term on height deviations to produce sharp local bumps.

3. **Stage 3 – Texturing**  
   Fixes the geometry and optimises colours to match the original RGB images.

---

## Dependencies

```
torch  ≥ 1.13
numpy
scipy          (for griddata interpolation)
opencv-python
```
