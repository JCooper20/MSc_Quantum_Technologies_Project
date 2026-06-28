"""
Critical-region sweep: fine r-grid through the transition, longer window.
r = 0.30..0.50 (0.01 steps), N = 64,128,256,512, t_max = 12N, 300 traj.
Reuses the existing parallel runner machinery; writes a SEPARATE checkpoint
so it doesn't collide with the main sweep's checkpoint.

Run:
    N_WORKERS=24 PYTHONPATH=. python scripts/run_critical_sweep.py
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

from scripts.run_all_to_all_parallel import Config, sweep

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg = Config()
    cfg.N_values     = [64, 128, 256, 512]
    cfg.r_values     = [round(0.30 + 0.01 * k, 2) for k in range(21)]  # 0.30..0.50
    cfg.n_traj       = 300
    cfg.t_max_factor = 12
    cfg.n_log_times  = 30
    cfg.n_lin_times  = 150        # dense late sampling -> smooth gradient for tau
    cfg.ckpt_path    = str(REPO_ROOT / "results" / "checkpoints" / "checkpoint_critical.json")

    print("Critical-region sweep")
    print(f"  N       : {cfg.N_values}")
    print(f"  r       : {cfg.r_values[0]} .. {cfg.r_values[-1]} (0.01 steps, {len(cfg.r_values)} pts)")
    print(f"  n_traj  : {cfg.n_traj}")
    print(f"  t_max   : {cfg.t_max_factor} * N")
    print(f"  ckpt    : {cfg.ckpt_path}\n")
    sweep(cfg)


if __name__ == "__main__":
    main()
