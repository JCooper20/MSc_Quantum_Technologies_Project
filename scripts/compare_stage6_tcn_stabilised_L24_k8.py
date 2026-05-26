"""
scripts/compare_stage6_tcn_stabilised_L24_k8.py
-------------------------------------------------
Compares MLP baseline and stabilised TCN policy for L=24, k=8.

Stabilised TCN changes vs original TCN run:
  1. Gradient clipping by global norm (clip to 1.0)
  2. Learning rate reduced from 3e-4 → 1e-4
  3. Temperature scaling T=2.0 on TCN logits before Plackett-Luce:
         p_q ∝ exp(s_q / T),  T=2.0 broadens selection distribution

MLP baseline loaded from results/checkpoints/policy_L24_k8.npz if
available, otherwise retrained.

Produces two figures:
  1. Training curves (entropy vs batch) for MLP and stabilised TCN
  2. Entropy trajectory (mean ± SEM over 100 episodes) for MLP, TCN,
     boundary heuristic, and uncontrolled — all on one axes

Saves to results/figures/stage6_compare_tcn_stabilised_L24_k8_*.png/.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import stim
import jax.numpy as jnp

from src.training.reinforce import train_policy, train_policy_tcn
from src.models.policy import (
    policy_forward, run_episode_adaptive,
    run_episode_adaptive_tcn,
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


def collect_trajectories_tcn(params, L, depth, k, temperature, n_episodes=100):
    """Run n_episodes with stabilised TCN policy."""
    adaptive_trajs = []

    for i in range(n_episodes):
        if (i + 1) % 20 == 0:
            print(f"  Episode {i+1}/{n_episodes}")

        ep = run_episode_adaptive_tcn(L, depth, k, params,
                                       window=4, entropy_interval=1,
                                       temperature=temperature)
        adaptive_trajs.append([s for _, s in ep['entropy_snapshots']])

    a = np.array(adaptive_trajs)
    return a.mean(0), a.std(0) / np.sqrt(n_episodes)


def main():
    L, k, depth    = 24, 8, 96
    n_episodes     = 100
    temperature    = 2.0
    ckpt_mlp       = 'results/checkpoints/policy_L24_k8.npz'
    ckpt_tcn       = 'results/checkpoints/policy_tcn_stabilised_L24_k8.npz'

    t0 = time.time()

    # ── MLP baseline ─────────────────────────────────────────────────────────
    if os.path.exists(ckpt_mlp):
        print(f"MLP: loading params from {ckpt_mlp}")
        params_mlp = load_params(ckpt_mlp)
        print("MLP: re-training to capture history...")
        params_mlp, history_mlp = train_policy(
            L=L, depth=depth, k_per_layer=k,
            n_batches=150, batch_size=48, lr=3e-4,
            pretrain_episodes=500, pretrain_epochs=30,
        )
    else:
        print("MLP: training...")
        params_mlp, history_mlp = train_policy(
            L=L, depth=depth, k_per_layer=k,
            n_batches=150, batch_size=48, lr=3e-4,
            pretrain_episodes=500, pretrain_epochs=30,
        )
        np.savez(ckpt_mlp, **{k_: np.array(v) for k_, v in params_mlp.items()})
        print(f"Params saved → {ckpt_mlp}")
    print(f"MLP done in {(time.time()-t0)/60:.1f} min.")

    # ── Stabilised TCN ───────────────────────────────────────────────────────
    t1 = time.time()
    print(f"\nStabilised TCN: training "
          f"(lr=1e-4, grad_clip=1.0, temperature={temperature})...")
    params_tcn, history_tcn = train_policy_tcn(
        L=L, depth=depth, k_per_layer=k,
        n_batches=150, batch_size=48, lr=1e-4,
        grad_clip=1.0, temperature=temperature,
        pretrain_episodes=500, pretrain_epochs=30,
    )
    np.savez(ckpt_tcn, **{k_: np.array(v) for k_, v in params_tcn.items()})
    print(f"Params saved → {ckpt_tcn}")
    print(f"Stabilised TCN done in {(time.time()-t1)/60:.1f} min.")

    # ── Figure 1: training curves ────────────────────────────────────────────
    batches = np.arange(1, 151)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(batches, history_mlp['entropy'], lw=1.5,
            label='MLP (lr=3e-4)', color='C0')
    ax.plot(batches, history_tcn['entropy'], lw=1.5,
            label=r'TCN stabilised (lr=1e-4, clip=1.0, $T$=2.0)',
            color='C1', ls='--')
    ax.set_xlabel('Batch number')
    ax.set_ylabel(r'$\langle S(L/2) \rangle$')
    ax.set_title(r'Training curves — $L=24$, $k=8$: MLP vs stabilised TCN')
    ax.legend()
    fig.tight_layout()
    fig.savefig('results/figures/stage6_compare_tcn_stabilised_L24_k8_training.png',
                dpi=200)
    fig.savefig('results/figures/stage6_compare_tcn_stabilised_L24_k8_training.pdf')
    print("\nSaved → stage6_compare_tcn_stabilised_L24_k8_training.png/.pdf")

    # ── Figure 2: entropy trajectories ───────────────────────────────────────
    print("\nCollecting trajectories for MLP...")
    (a_mean, a_sem), (b_mean, b_sem), (u_mean, u_sem) = collect_trajectories_mlp(
        params_mlp, L, depth, k, n_episodes)

    print("Collecting trajectories for stabilised TCN...")
    ba_mean, ba_sem = collect_trajectories_tcn(
        params_tcn, L, depth, k, temperature, n_episodes)

    layers = np.arange(1, depth + 1)
    fig, ax = plt.subplots(figsize=(9, 5))

    for mean, sem, label, color, ls in [
        (a_mean,  a_sem,  'Adaptive MLP',                       'C0', '-'),
        (ba_mean, ba_sem, r'Adaptive TCN (clip=1.0, $T$=2.0)',  'C1', '--'),
        (b_mean,  b_sem,  'Boundary heuristic',                  'C2', '-.'),
        (u_mean,  u_sem,  'Uncontrolled',                        'C3', ':'),
    ]:
        ax.plot(layers, mean, lw=1.8, label=label, color=color, ls=ls)
        ax.fill_between(layers, mean - sem, mean + sem, alpha=0.2, color=color)

    ax.set_xlabel('Circuit layer')
    ax.set_ylabel(r'$S(L/2)$')
    ax.set_title(r'Entropy trajectory — $L=24$, $k=8$: MLP vs stabilised TCN')
    ax.legend()
    fig.tight_layout()
    fig.savefig('results/figures/stage6_compare_tcn_stabilised_L24_k8_trajectory.png',
                dpi=200)
    fig.savefig('results/figures/stage6_compare_tcn_stabilised_L24_k8_trajectory.pdf')
    print("Saved → stage6_compare_tcn_stabilised_L24_k8_trajectory.png/.pdf")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min.")
    print(f"\nFinal S(L/2):")
    print(f"  MLP:               {a_mean[-10:].mean():.3f}")
    print(f"  TCN stabilised:    {ba_mean[-10:].mean():.3f}")
    print(f"  Boundary heuristic:{b_mean[-10:].mean():.3f}")


if __name__ == '__main__':
    main()
