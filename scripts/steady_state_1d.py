"""
Steady-state check for the 1D brickwork geometry.

Runs the simulator run_trajectory_stim with a deep circuit (8*L layers) 
and plots the half-chain entropy S(L/2) vs depth, averaged over n_traj 
trajectories. 
"""

# Imports
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.simulators.brickwork_1D_stim import run_trajectory_stim

N_VALUES = [16, 36, 64, 121] # qubit counts (= L for 1D)
P_M_VALUES = [0.10, 0.34, 0.50] # [volume-law, ~p_c, area-law]
N_TRAJ = 50
COLORS = {16: "C0", 36: "C1", 64: "C2", 121: "C3"}


def mean_entropy_history(L, depth, p_m, n_traj):
    """
    Average S(L/2) at each layer over n_traj trajectories
    """
    runs = [run_trajectory_stim(L, depth, p_m)['entropy_history']
            for _ in range(n_traj)]
    return np.mean(runs, axis=0)


def main():
    os.makedirs("results/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, p_m in zip(axes, P_M_VALUES):
        for L in N_VALUES:
            depth = 8 * L                  
            S_t = mean_entropy_history(L, depth, p_m, N_TRAJ)
            ax.plot(range(1, depth + 1), S_t, color=COLORS[L],
                    lw=1.4, label=f"$N={L}$")
            ax.axvline(4 * L, color=COLORS[L], ls="--", lw=0.9, alpha=0.7)
        ax.set_xlabel("depth $t$ (layers)")
        ax.set_ylabel(r"$S(L/2)$")
        ax.set_title(f"$p_m = {p_m:.2f}$")
        ax.set_xscale("log")
        ax.legend(fontsize=8)

    fig.suptitle("Steady-state check — 1D brickwork "
                 "(dashed = main-sweep depth 4N)")
    fig.tight_layout()
    fig.savefig("results/figures/steady_state_1d.png", dpi=200)
    plt.close(fig)
    print("Saved results/figures/steady_state_1d.png")


if __name__ == "__main__":
    main()
