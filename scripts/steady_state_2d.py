"""
Steady-state check for the 2D lattice geometry.

Runs the Phase A 2D simulator (run_trajectory_2d) to a deep circuit
and plots S vs depth, averaged over n_traj trajectories. 
"""

# Imports
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.simulators.lattice_2D_stim import run_trajectory_2d

GRIDS = [(4, 16), (6, 36), (8, 64), (11, 121)] # (linear size L, total qubit count N = L*L) for each grid
P_M_VALUES = [0.10, 0.35, 0.50]   # [volume-law, ~p_c, area-law]
N_TRAJ = 50
COLORS = {16: "C0", 36: "C1", 64: "C2", 121: "C3"}


def mean_entropy_history(L, depth, p_m, n_traj):
    """
    Average S at each layer over n_traj trajectories (2D sim takes L)
    """
    runs = [run_trajectory_2d(L, depth, p_m)['entropy_history']
            for _ in range(n_traj)]
    return np.mean(runs, axis=0)


def main():
    os.makedirs("results/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, p_m in zip(axes, P_M_VALUES):
        for L, N in GRIDS:
            depth = 8 * N                     
            S_t = mean_entropy_history(L, depth, p_m, N_TRAJ)
            ax.plot(range(1, depth + 1), S_t, color=COLORS[N],
                    lw=1.4, label=f"$N={N}$")
            ax.axvline(4 * L, color=COLORS[N], ls="--", lw=0.9, alpha=0.7)
        ax.set_xlabel("depth $t$ (layers)")
        ax.set_ylabel(r"$S$ (horizontal cut)")
        ax.set_title(f"$p_m = {p_m:.2f}$")
        ax.set_xscale("log")
        ax.legend(fontsize=8)

    fig.suptitle("Steady-state check — 2D lattice "
                 "(dashed = main-sweep depth 4·sqrt(N))")
    fig.tight_layout()
    fig.savefig("results/figures/steady_state_2d.png", dpi=200)
    plt.close(fig)
    print("Saved results/figures/steady_state_2d.png")


if __name__ == "__main__":
    main()
