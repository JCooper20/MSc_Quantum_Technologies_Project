"""
scripts/plot_stage6_entropy_trajectory_L24_k12.py
--------------------------------------------------
Loads trained policy for L=24, k=12 from results/checkpoints/policy_L24_k12.npz
if available, otherwise retrains and saves the checkpoint. Then runs 100
episodes for four conditions recording S(L/2) after every circuit layer.

Conditions:
  1. Trained REINFORCE policy (adaptive)
  2. Boundary-avoiding heuristic
  3. Random measurements at rate k/L = 12/24 = 0.5
  4. Uncontrolled circuit (p_m = 0, no measurements)

Plots mean ± SEM trajectories on one figure and saves to
results/figures/stage6_entropy_trajectory_L24_k12.png/.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import stim
import jax.numpy as jnp

from src.training.reinforce import train_policy
from src.models.policy import (
    policy_forward, run_episode_adaptive,
    boundary_scores, _apply_layer,
)
from src.analysis.entropy import stabiliser_entropy


def run_episode_uncontrolled(L: int, depth: int):
    """Free brickwork evolution with no measurements — records S(L/2) per layer."""
    sim = stim.TableauSimulator()
    sim.set_num_qubits(L)
    trajectory = []
    for t in range(depth):
        _apply_layer(sim, L, t)
        trajectory.append(stabiliser_entropy(sim, L, L // 2))
    return trajectory


def load_params(path):
    """Load policy params from .npz, converting arrays back to jnp."""
    data = np.load(path)
    return {k: jnp.array(v) for k, v in data.items()}


def main():
    L, k, depth = 24, 12, 96
    n_episodes   = 100
    ckpt_path    = 'results/checkpoints/policy_L24_k12.npz'

    t0 = time.time()

    # ── Load or retrain ──────────────────────────────────────────────────────
    if os.path.exists(ckpt_path):
        print(f"Loading params from {ckpt_path}")
        params = load_params(ckpt_path)
    else:
        print(f"No checkpoint found. Training: L={L}  k={k}  depth={depth}  n_batches=150")
        params, _ = train_policy(
            L=L, depth=depth, k_per_layer=k,
            n_batches=150, batch_size=48, lr=3e-4,
            pretrain_episodes=500, pretrain_epochs=30,
        )
        np.savez(ckpt_path, **{k_: np.array(v) for k_, v in params.items()})
        print(f"Params saved → {ckpt_path}")
        print(f"Training done in {(time.time()-t0)/60:.1f} min.")

    # ── 100-episode evaluation with per-layer entropy ────────────────────────
    print(f"\nRunning {n_episodes} episodes per condition...")

    adaptive_trajs, boundary_trajs, random_trajs, uncontrolled_trajs = [], [], [], []

    for i in range(n_episodes):
        if (i + 1) % 20 == 0:
            print(f"  Episode {i+1}/{n_episodes}")

        # 1. Adaptive (REINFORCE)
        ep = run_episode_adaptive(L, depth, k, policy_forward, params,
                                   window=4, entropy_interval=1)
        adaptive_trajs.append([s for _, s in ep['entropy_snapshots']])

        # 2. Boundary heuristic with per-layer entropy
        sim = stim.TableauSimulator(); sim.set_num_qubits(L)
        scores = boundary_scores(L)
        traj = []
        for t in range(depth):
            _apply_layer(sim, L, t)
            noisy = scores + np.random.normal(0, 0.01, L)
            for q in np.argsort(noisy)[-k:]:
                sim.measure(q)
            traj.append(stabiliser_entropy(sim, L, L // 2))
        boundary_trajs.append(traj)

        # 3. Random at p_m = k/L with per-layer entropy
        sim = stim.TableauSimulator(); sim.set_num_qubits(L)
        p_m = k / L
        traj = []
        for t in range(depth):
            _apply_layer(sim, L, t)
            for q in range(L):
                if np.random.random() < p_m:
                    sim.measure(q)
            traj.append(stabiliser_entropy(sim, L, L // 2))
        random_trajs.append(traj)

        # 4. Uncontrolled (no measurements)
        uncontrolled_trajs.append(run_episode_uncontrolled(L, depth))

    # ── Compute mean ± SEM ───────────────────────────────────────────────────
    layers = np.arange(1, depth + 1)

    def stats(trajs):
        a = np.array(trajs)
        return a.mean(0), a.std(0) / np.sqrt(n_episodes)

    a_mean, a_sem = stats(adaptive_trajs)
    b_mean, b_sem = stats(boundary_trajs)
    r_mean, r_sem = stats(random_trajs)
    u_mean, u_sem = stats(uncontrolled_trajs)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    for mean, sem, label, color, ls in [
        (a_mean, a_sem, 'Adaptive (REINFORCE)', 'C0', '-'),
        (b_mean, b_sem, 'Boundary heuristic',   'C1', '--'),
        (r_mean, r_sem, r'Random ($p_m = k/L$)', 'C2', '-.'),
        (u_mean, u_sem, 'Uncontrolled',           'C3', ':'),
    ]:
        ax.plot(layers, mean, lw=1.8, label=label, color=color, ls=ls)
        ax.fill_between(layers, mean - sem, mean + sem, alpha=0.2, color=color)

    ax.set_xlabel('Circuit layer')
    ax.set_ylabel(r'$S(L/2)$')
    ax.set_title(r'Entropy trajectory — $L=24$, $k=12$, 100 episodes')
    ax.legend()
    fig.tight_layout()
    fig.savefig('results/figures/stage6_entropy_trajectory_L24_k12.png', dpi=200)
    fig.savefig('results/figures/stage6_entropy_trajectory_L24_k12.pdf')
    print("\nSaved → results/figures/stage6_entropy_trajectory_L24_k12.png/.pdf")
    print(f"Total time: {(time.time()-t0)/60:.1f} min.")


if __name__ == '__main__':
    main()
