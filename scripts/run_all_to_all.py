"""
Runner for the monitored all-to-all Clifford circuit (purification probe).

Sweeps system sizes N and measurement rates r, averages the reference
entropy S_R over trajectories, and plots the results. 

Drives the simulator in src/simulators/all_to_all.py
"""

import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import stim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.analysis.entropy import stabiliser_entropy_region
from src.simulators.all_to_all import (build_initial_bell_state, make_sample_times, 
                                       run_single_trajectory,)

# repo root = two levels up from scripts/
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
    # Fine grid from r=0 through the transition (~0.45-0.50), coarse above.
    r_values: List[float] = field(default_factory=lambda:
        [round(0.02 * k, 2) for k in range(26)]      # 0.00, 0.02, ..., 0.50
        + [0.60, 0.70, 0.80, 0.90])
    n_traj: int = 300             # trajectories averaged per (N, r)
    t_max_factor: int = 8         # t_max = t_max_factor * N time steps
    n_log_times: int = 14         # log-spaced sample times
    n_lin_times: int = 40         # linearly-spaced sample times (dense, for
                                  # the mid-time log-derivative window)
    seed: int = 12345
    outdir: str = str(REPO_ROOT / "results" / "figures" / "all_to_all")
    ckpt_path: str = str(REPO_ROOT / "results" / "checkpoints" / "checkpoint.json")


# =====================================================================
# Sanity checks
# =====================================================================

def validate() -> None:
    """
    Sanity checks
    """
    print("Validation:")

    sim = build_initial_bell_state(1)
    assert abs(stabiliser_entropy_region(sim, 2, [1]) - 1.0) < 1e-9
    print("  [PASS] N=1 Bell pair        S_R = 1")

    sim = stim.TableauSimulator(); sim.set_num_qubits(2)
    assert abs(stabiliser_entropy_region(sim, 2, [1])) < 1e-9
    print("  [PASS] product state        S_R = 0")

    sim = build_initial_bell_state(4)
    assert abs(stabiliser_entropy_region(sim, 8, list(range(4, 8))) - 4.0) < 1e-9
    print("  [PASS] N=4 Bell pairs       S_R = 4")

    out = run_single_trajectory(N=8, r=0.0, sample_times=[1, 4, 8],t_max=8, seed=1)
    assert np.allclose(out, 8.0, atol=1e-9)
    print("  [PASS] r=0  -> S_R stays 8")

    out = run_single_trajectory(N=8, r=1.0, sample_times=[2, 4, 8],t_max=8, seed=1)
    assert out[-1] < 1.0
    print("  [PASS] r=1  -> S_R decays to 0\n")


# =====================================================================
# Sweep
# =====================================================================

def sweep(cfg: Config) -> Dict[Tuple[int, float], dict]:
    """
    Run every (N, r) cell, averaging S_R(t) over 'n' trajectories.
    Saves the checkpoint after every cell so a long run never loses more
    than one cell on a crash.
    Returns (N, r) -> {'times', 'mean', 'sem', 'n_traj'}
    """
    results: Dict[Tuple[int, float], dict] = {}
    for N in cfg.N_values:
        t_max = cfg.t_max_factor * N
        sample_times = make_sample_times(t_max, cfg.n_log_times, cfg.n_lin_times)
        for r in cfg.r_values:
            t0 = time.time()
            traj = np.array([
                run_single_trajectory(
                    N=N, r=r, sample_times=sample_times, t_max=t_max,
                    seed=cfg.seed + 1009 * N + 7919 * int(r * 1e4) + j)
                for j in range(cfg.n_traj)
            ])
            mean = traj.mean(0)
            sem  = traj.std(0, ddof=1) / np.sqrt(cfg.n_traj)
            results[(N, r)] = {"times": sample_times,
                               "mean": mean.tolist(),
                               "sem": sem.tolist(),
                               "n_traj": cfg.n_traj}
            save_checkpoint(results, cfg.ckpt_path)   # crash-safe incremental save
            print(f"  N={N:>3}  r={r:.2f}  S_R(final)/N = "
                  f"{mean[-1] / N:.3f}   ({time.time() - t0:.1f}s)")
    return results


# =====================================================================
# Checkpoint I/O  (nested {str(N): {f"{r:.4f}": cell}} format)
# =====================================================================

def save_checkpoint(results: Dict[Tuple[int, float], dict], path: str) -> None:
    """Write results to JSON, nested by N then r."""
    nested: dict = {}
    for (N, r), cell in results.items():
        nested.setdefault(str(N), {})[f"{r:.4f}"] = cell
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(nested, fh, indent=2)


# =====================================================================
# Figures
# =====================================================================

def _save(fig, outdir, stem):
    """
    Saves figure.
    """
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/{stem}.{ext}", dpi=200)
    plt.close(fig)


def make_plots(cfg: Config, results: Dict[Tuple[int, float], dict]) -> None:
    """
    fig1/2: S_R/N vs t at small/large r.  fig3: late-time S_R/N vs r."""

    def time_series(r_target, stem, label):
        fig, ax = plt.subplots(figsize=(8, 5))
        for N in cfg.N_values:
            if (N, r_target) not in results:
                continue
            d = results[(N, r_target)]
            t = np.asarray(d["times"], float)
            m = np.asarray(d["mean"]) / N
            ax.plot(t, m, "-o", ms=3, lw=1.4, label=f"$N={N}$")
        ax.set_xlabel(r"time step $t$  (units of $N$ ops)")
        ax.set_ylabel(r"$S_R(t)/N$")
        ax.set_title(fr"All-to-all Clifford MIPT — {label} ($r={r_target:.2f}$)")
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8)
        _save(fig, cfg.outdir, stem)

    time_series(min(cfg.r_values), "fig1_plateau_smallr", "entangling plateau")
    time_series(max(cfg.r_values), "fig2_decay_larger", "disentangling decay")

    fig, ax = plt.subplots(figsize=(8, 5))
    for N in cfg.N_values:
        rs = [r for r in cfg.r_values if (N, r) in results]
        ms = [results[(N, r)]["mean"][-1] / N for r in rs]
        ss = [results[(N, r)]["sem"][-1] / N for r in rs]
        ax.errorbar(rs, ms, yerr=ss, marker="o", ms=4, lw=1.4,
                    capsize=2, label=f"$N={N}$")
    ax.set_xlabel(r"measurement rate $r$")
    ax.set_ylabel(r"$S_R(t_{\rm late})/N$")
    ax.set_title("All-to-all Clifford MIPT — late-time $S_R/N$ vs $r$")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8)
    _save(fig, cfg.outdir, "fig3_late_vs_r")


# =====================================================================
# Main
# =====================================================================

def main(cfg: Config | None = None) -> None:
    cfg = cfg or Config()
    print(f"All-to-all MIPT  |  N={cfg.N_values}  r={cfg.r_values}\n")
    validate()
    print("Sweep:")
    results = sweep(cfg)
    save_checkpoint(results, cfg.ckpt_path)
    print("\nPlotting...")
    make_plots(cfg, results)
    print(f"Figures written to {cfg.outdir}/")


if __name__ == "__main__":
    main()
