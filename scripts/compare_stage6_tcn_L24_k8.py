"""
scripts/compare_stage6_tcn_L24_k8.py
--------------------------------------
Compares MLP and TCN policy backbones for L=24, k=8:

  A. MLP policy:  FC(128) → FC(64) → FC(L)          (episode REINFORCE)
  B. TCN policy:  CausalConv (dilations 1,2,4) → FC(L) (episode REINFORCE)

Both use identical supervised pre-training and the same
n_batches=150, batch_size=48, lr=3e-4 hyperparameters.

Loads existing checkpoint for variant A from
  results/checkpoints/policy_L24_k8.npz        (MLP)
  results/checkpoints/policy_tcn_L24_k8.npz    (TCN)
if available; otherwise trains from scratch.

Produces two figures:
  1. Training curves (entropy vs batch) for A and B overlaid
  2. Entropy trajectory (mean ± SEM over 100 episodes) for A, B,
     boundary heuristic, and uncontrolled — all on one axes

Saves to results/figures/stage6_compare_tcn_L24_k8_*.png/.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import stim
import jax.numpy as jnp

from src.training.reinforce import train_policy, train_policy_tcn
from src.models.policy import (
    policy_forward, tcn_policy_forward,
    run_episode_adaptive, run_episode_adaptive_tcn,
    boundary_scores, _apply_layer,
)
from src.analysis.entropy import stabiliser_entropy


def load_params(path):
    """Load policy params from .npz, converting arrays back to jnp."""
    data = np.load(path)
    return {k: jnp.array(v) for k, v in data.items()}


def run_episode_uncontrolled(L: int, depth: int):
    """Free brickwork with no measurements — records S(L/2) per layer."""
    sim = stim.TableauSimulator()
    sim.set_num_qubits(L)
    traj = []
    for t in range(depth):
        _apply_layer(sim, L, t)
        traj.append(stabiliser_entropy(sim, L, L // 2))
    return traj


def collect_trajectories_mlp(params, L, depth, k, n_episodes=100):
    """Run n_episodes with MLP policy + boundary + uncontrolled."""
    adaptive_trajs, boundary_trajs, uncontrolled_trajs = [], [], []
    scores = boundary_scores(L)

    for i in range(n_episodes):
        if (i + 1) % 20 == 0:
            print(f"  Episode {i+1}/{n_episodes}")

        ep = run_episode_adaptive(L, depth, k, policy_forward, params,
                                   window=4, entropy_interval=1)
        adaptive_trajs.append([s for _, s in ep['entropy_snapshots']])

        sim = stim.TableauSimulator(); sim.set_num_qubits(L)
        traj = []
        for t in range(depth):
            _apply_layer(sim, L, t)
            noisy = scores + np.random.normal(0, 0.01, L)
            for q in np.argsort(noisy)[-k:]:
                sim.measure(q)
            traj.append(stabiliser_entropy(sim, L, L // 2))
        boundary_trajs.append(traj)

        uncontrolled_trajs.append(run_episode_uncontrolled(L, depth))

    def stats(trajs):
        a = np.array(trajs)
        return a.mean(0), a.std(0) / np.sqrt(n_episodes)

    return (stats(adaptive_trajs), stats(boundary_trajs),
            stats(uncontrolled_trajs))


def collect_trajectories_tcn(params, L, depth, k, n_episodes=100):
    """Run n_episodes with TCN policy."""
    adaptive_trajs = []

    for i in range(n_episodes):
        if (i + 1) % 20 == 0:
            print(f"  Episode {i+1}/{n_episodes}")

        ep = run_episode_adaptive_tcn(L, depth, k, params,
                                       window=4, entropy_interval=1)
        adaptive_trajs.append([s for _, s in ep['entropy_snapshots']])

    a = np.array(adaptive_trajs)
    return a.mean(0), a.std(0) / np.sqrt(n_episodes)


def main():
    L, k, depth  = 24, 8, 96
    n_episodes   = 100
    ckpt_a       = 'results/checkpoints/policy_L24_k8.npz'
    ckpt_b       = 'results/checkpoints/policy_tcn_L24_k8.npz'

    t0 = time.time()

    # ── Variant A: MLP policy ────────────────────────────────────────────────
    if os.path.exists(ckpt_a):
        print(f"Variant A: loading MLP params from {ckpt_a}")
        params_a = load_params(ckpt_a)
        print("Variant A: re-training to capture history...")
        params_a, history_a = train_policy(
            L=L, depth=depth, k_per_layer=k,
            n_batches=150, batch_size=48, lr=3e-4,
            pretrain_episodes=500, pretrain_epochs=30,
        )
    else:
        print("Variant A: training (MLP)...")
        params_a, history_a = train_policy(
            L=L, depth=depth, k_per_layer=k,
            n_batches=150, batch_size=48, lr=3e-4,
            pretrain_episodes=500, pretrain_epochs=30,
        )
        np.savez(ckpt_a, **{k_: np.array(v) for k_, v in params_a.items()})
        print(f"Params saved → {ckpt_a}")
    print(f"Variant A done in {(time.time()-t0)/60:.1f} min.")

    # ── Variant B: TCN policy ────────────────────────────────────────────────
    t1 = time.time()
    if os.path.exists(ckpt_b):
        print(f"\nVariant B: loading TCN params from {ckpt_b}")
        params_b = load_params(ckpt_b)
        print("Variant B: re-training to capture history...")
        params_b, history_b = train_policy_tcn(
            L=L, depth=depth, k_per_layer=k,
            n_batches=150, batch_size=48, lr=3e-4,
            pretrain_episodes=500, pretrain_epochs=30,
        )
    else:
        print("\nVariant B: training (TCN)...")
        params_b, history_b = train_policy_tcn(
            L=L, depth=depth, k_per_layer=k,
            n_batches=150, batch_size=48, lr=3e-4,
            pretrain_episodes=500, pretrain_epochs=30,
        )
        np.savez(ckpt_b, **{k_: np.array(v) for k_, v in params_b.items()})
        print(f"Params saved → {ckpt_b}")
    print(f"Variant B done in {(time.time()-t1)/60:.1f} min.")

    # ── Figure 1: training curves ────────────────────────────────────────────
    batches = np.arange(1, 151)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(batches, history_a['entropy'], lw=1.5,
            label='MLP policy (A)', color='C0')
    ax.plot(batches, history_b['entropy'], lw=1.5,
            label='TCN policy (B)', color='C1', ls='--')
    ax.set_xlabel('Batch number')
    ax.set_ylabel(r'$\langle S(L/2) \rangle$')
    ax.set_title(r'Training curves — $L=24$, $k=8$: MLP vs TCN policy')
    ax.legend()
    fig.tight_layout()
    fig.savefig('results/figures/stage6_compare_tcn_L24_k8_training.png', dpi=200)
    fig.savefig('results/figures/stage6_compare_tcn_L24_k8_training.pdf')
    print("\nSaved → stage6_compare_tcn_L24_k8_training.png/.pdf")

    # ── Figure 2: entropy trajectories ───────────────────────────────────────
    print("\nCollecting trajectories for variant A (MLP)...")
    (a_mean, a_sem), (b_mean, b_sem), (u_mean, u_sem) = collect_trajectories_mlp(
        params_a, L, depth, k, n_episodes)

    print("Collecting trajectories for variant B (TCN)...")
    ba_mean, ba_sem = collect_trajectories_tcn(params_b, L, depth, k, n_episodes)

    layers = np.arange(1, depth + 1)
    fig, ax = plt.subplots(figsize=(9, 5))

    for mean, sem, label, color, ls in [
        (a_mean,  a_sem,  'Adaptive A (MLP)',       'C0', '-'),
        (ba_mean, ba_sem, 'Adaptive B (TCN)',        'C1', '--'),
        (b_mean,  b_sem,  'Boundary heuristic',      'C2', '-.'),
        (u_mean,  u_sem,  'Uncontrolled',             'C3', ':'),
    ]:
        ax.plot(layers, mean, lw=1.8, label=label, color=color, ls=ls)
        ax.fill_between(layers, mean - sem, mean + sem, alpha=0.2, color=color)

    ax.set_xlabel('Circuit layer')
    ax.set_ylabel(r'$S(L/2)$')
    ax.set_title(r'Entropy trajectory — $L=24$, $k=8$: MLP vs TCN policy')
    ax.legend()
    fig.tight_layout()
    fig.savefig('results/figures/stage6_compare_tcn_L24_k8_trajectory.png', dpi=200)
    fig.savefig('results/figures/stage6_compare_tcn_L24_k8_trajectory.pdf')
    print("Saved → stage6_compare_tcn_L24_k8_trajectory.png/.pdf")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min.")
    print(f"\nFinal S(L/2):")
    print(f"  Variant A (MLP):        {a_mean[-10:].mean():.3f}")
    print(f"  Variant B (TCN):        {ba_mean[-10:].mean():.3f}")
    print(f"  Boundary heuristic:     {b_mean[-10:].mean():.3f}")


if __name__ == '__main__':
    main()
