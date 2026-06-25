"""
Parallelised runner for the monitored all-to-all Clifford circuit.

"""

# Imports
import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from src.simulators.all_to_all_stim import (make_sample_times,
                                            run_single_trajectory)

REPO_ROOT = Path(__file__).resolve().parent.parent


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class Config:
    """
    All sweep parameters
    """
    N_values: List[int] = field(default_factory=lambda:
        [8, 16, 32, 64, 128, 256, 512])

    # fine 0.02 grid all the way from 0.00 to 0.90  (46 points)
    r_values: List[float] = field(default_factory=lambda:
        [round(0.02 * k, 2) for k in range(46)])

    n_traj:       int = 500        # trajectories per (N, r)
    t_max_factor: int = 8          # t_max = 8 * N
    n_log_times:  int = 14
    n_lin_times:  int = 60         # dense late-time sampling (stabilises tau)
    seed:         int = 12345

    # parallelism: all cores by default; override with env var N_WORKERS
    n_workers:    int = int(os.environ.get("N_WORKERS", os.cpu_count() or 4))

    ckpt_path:    str = str(REPO_ROOT / "results" / "checkpoints" / "checkpoint.json")


# =====================================================================
# Worker (module-level so it pickles for the pool)
# =====================================================================

def _one_traj(args) -> np.ndarray:
    N, r, sample_times, t_max, seed = args
    return run_single_trajectory(N=N, r=r, sample_times=sample_times,
                                 t_max=t_max, seed=seed)


# =====================================================================
# Checkpoint I/O  (resume-aware, atomic writes)
# =====================================================================

def load_checkpoint(path: str) -> dict:
    p = Path(path)
    return json.load(open(p)) if p.exists() else {}

def save_checkpoint(nested: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(nested, fh, indent=2)
    os.replace(tmp, path)         


# =====================================================================
# Sweep (parallel over trajectories within each cell)
# =====================================================================

def sweep(cfg: Config) -> None:
    nested = load_checkpoint(cfg.ckpt_path)     # resume if checkpoint exists
    print(f"Workers: {cfg.n_workers}   (machine cores: {os.cpu_count()})\n")

    with ProcessPoolExecutor(max_workers=cfg.n_workers) as pool:
        for N in cfg.N_values:
            t_max = cfg.t_max_factor * N
            sample_times = make_sample_times(t_max, cfg.n_log_times,
                                             cfg.n_lin_times)
            N_key = str(N)
            for r in cfg.r_values:
                r_key = f"{r:.4f}"
                if N_key in nested and r_key in nested[N_key]:
                    continue                     # already done — skip (resume)

                t0 = time.time()
                args = [(N, r, sample_times, t_max,
                         cfg.seed + 1009 * N + 7919 * int(r * 1e4) + j)
                        for j in range(cfg.n_traj)]
                traj = np.array(list(pool.map(_one_traj, args, chunksize=4)))

                mean = traj.mean(0)
                sem  = traj.std(0, ddof=1) / np.sqrt(cfg.n_traj)
                nested.setdefault(N_key, {})[r_key] = {
                    "times":  sample_times,
                    "mean":   mean.tolist(),
                    "sem":    sem.tolist(),
                    "n_traj": cfg.n_traj,
                }
                save_checkpoint(nested, cfg.ckpt_path)
                print(f"  N={N:>4}  r={r:.2f}  S_R/N={mean[-1]/N:.3f}   "
                      f"({time.time()-t0:.1f}s)", flush=True)


def main() -> None:
    cfg = Config()
    print("Parallel all-to-all Clifford sweep")
    print(f"  N       : {cfg.N_values}")
    print(f"  r points: {len(cfg.r_values)}  (0.00 .. {max(cfg.r_values):.2f}, step 0.02)")
    print(f"  n_traj  : {cfg.n_traj}")
    print(f"  t_max   : {cfg.t_max_factor} * N")
    print(f"  ckpt    : {cfg.ckpt_path}\n")
    t0 = time.time()
    sweep(cfg)
    print(f"\nDone in {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
