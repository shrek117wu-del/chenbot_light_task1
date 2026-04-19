"""
run_all_experiments.py – Run all paper experiments (Figures 15-21)
================================================================

This script runs the full optimization pipeline for all combinations
of saucer shapes and image pairs described in the paper.

Usage:
    # Run all experiments at paper quality (slow, ~30 min each)
    python run_all_experiments.py

    # Quick smoke test (low resolution, few iterations)
    python run_all_experiments.py --quick

    # Run only Figure 15 experiments (5 shapes, 1 image pair)
    python run_all_experiments.py --figure 15

    # Run a specific single experiment
    python run_all_experiments.py --shape saucer --exp 1 --cup cylinder
"""
import argparse
import os
import shutil
import time

from experiments import (
    PAPER_SHAPES, PAPER_CUP_TYPES,
    FIG15_EXPERIMENTS, FIG18_EXPERIMENTS, FIG19_EXPERIMENTS,
    ALL_PAPER_EXPERIMENTS, get_paper_experiment,
)
from main import run_pipeline
from export_utils import export_obj_and_texture


def run_single(cfg, output_dir, resolution, render_size, iters1, iters2, iters3):
    """Run one experiment configuration and save results to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    shape   = cfg["shape"]
    exp_id  = cfg["exp_id"]
    cup     = cfg["cup"]
    label   = f"{shape}_exp{exp_id}_{cup}"

    print(f"\n{'='*60}")
    print(f"  Experiment: {label}")
    print(f"  Shape={shape}  Images=exp{exp_id}  Cup={cup}")
    print(f"  Res={resolution}  Render={render_size}  Iters={iters1}/{iters2}/{iters3}")
    print(f"{'='*60}")

    Id_path, Ir_path = get_paper_experiment(exp_id)
    X, Y, H, C = run_pipeline(
        Id_path, Ir_path,
        shape_type=shape,
        cup_type=cup,
        resolution=(resolution, resolution),
        img_render_size=(render_size, render_size),
        iters1_large=iters1, iters1_small=iters1,
        iters2=iters2, iters3=iters3,
    )

    # Move outputs to experiment-specific directory
    exp_dir = os.path.join(output_dir, label)
    os.makedirs(exp_dir, exist_ok=True)
    for f in ["out_direct.png", "out_reflect.png",
              "out_stage1_direct.png", "out_stage1_reflect.png",
              "out_stage2_direct.png", "out_stage2_reflect.png",
              "saucer.obj", "texture.png", "cup.obj"]:
        if os.path.exists(f):
            shutil.move(f, os.path.join(exp_dir, f))

    # Also export the cup OBJ
    export_obj_and_texture(X, Y, H, C,
                           obj_path=os.path.join(exp_dir, "saucer.obj"),
                           tex_path=os.path.join(exp_dir, "texture.png"),
                           cup_type=cup)
    return exp_dir


def main():
    p = argparse.ArgumentParser(description="Run all paper experiments")
    p.add_argument("--quick", action="store_true",
                   help="Quick test mode (low res, few iters)")
    p.add_argument("--figure", type=int, default=None,
                   help="Run only experiments for a specific figure (15, 18, 19)")
    p.add_argument("--shape", type=str, default=None,
                   help="Override shape for single experiment")
    p.add_argument("--exp", type=int, default=None,
                   help="Override exp_id for single experiment")
    p.add_argument("--cup", type=str, default=None,
                   help="Override cup type for single experiment")
    p.add_argument("--output", type=str, default="results",
                   help="Output directory for all experiments")
    args = p.parse_args()

    # Select experiments
    if args.shape and args.exp is not None:
        experiments = [{"shape": args.shape,
                        "exp_id": args.exp,
                        "cup": args.cup or "cylinder"}]
    elif args.figure == 15:
        experiments = FIG15_EXPERIMENTS
    elif args.figure == 18:
        experiments = FIG18_EXPERIMENTS
    elif args.figure == 19:
        experiments = FIG19_EXPERIMENTS
    else:
        experiments = ALL_PAPER_EXPERIMENTS

    # Parameters
    if args.quick:
        res, render, i1, i2, i3 = 60, 128, 50, 30, 30
    else:
        res, render, i1, i2, i3 = 150, 512, 500, 300, 300

    print(f"\nRunning {len(experiments)} experiments")
    print(f"  Resolution: {res}×{res}  Render: {render}×{render}")
    print(f"  Iterations: {i1}/{i2}/{i3}")
    print(f"  Output: {args.output}/\n")

    t_all = time.time()
    completed = []
    for i, cfg in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Starting experiment...")
        try:
            exp_dir = run_single(cfg, args.output, res, render, i1, i2, i3)
            completed.append((cfg, exp_dir))
        except Exception as e:
            print(f"  ⚠ FAILED: {e}")

    elapsed = time.time() - t_all
    print(f"\n{'='*60}")
    print(f"  ALL DONE: {len(completed)}/{len(experiments)} experiments")
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"  Results in: {args.output}/")
    print(f"{'='*60}")

    # Summary
    for cfg, exp_dir in completed:
        print(f"  ✓ {cfg['shape']:20s} exp{cfg['exp_id']} {cfg['cup']:10s} → {exp_dir}")


if __name__ == "__main__":
    main()
