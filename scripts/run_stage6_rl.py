"""
Stage 6: Adaptive RL Controller

Trains a REINFORCE policy to adaptively select which qubits to measure
at each circuit layer, targeting high entanglement (volume-law) at
measurement rates that would otherwise drive area-law behaviour.

Running this trains a policy for each (L, k) in RLConfig, evaluates it
against boundary and random baselines, saves the numerical results to
results/rl_controller.json, and writes a training-curve and an
entropy-trajectory figure per configuration into results/figures/.
"""

import os
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import RLConfig
from src.models.policy import (
    run_episode_random, run_episode_boundary, run_episode_adaptive,
    policy_forward, init_policy_params, boundary_scores,
)
from src.training.reinforce import train_policy
from src.analysis.entropy import stabiliser_entropy


def evaluate(params, L: int, depth: int, k_per_layer: int,
             n_eval: int, window: int = 4) -> dict:
    """Compare adaptive vs random vs boundary at matched measurement rate.

    Also retains the full per-layer entropy history of the first adaptive
    episode (sample_trajectory) so the entropy-trajectory figure can be
    drawn without re-running the policy.
    """
    adaptive_S, boundary_S, random_S = [], [], []
    sample_trajectory = None   # S(t) history of one trained episode

    for i in range(n_eval):
        ep = run_episode_adaptive(L, depth, k_per_layer,
                                   policy_forward, params, window=window)
        adaptive_S.append(ep['final_entropy'])
        if i == 0:
            sample_trajectory = ep.get('entropy_history')

        b_S, _ = run_episode_boundary(L, depth, k_per_layer)
        boundary_S.append(b_S)

        r_S, _ = run_episode_random(L, depth, k_per_layer / L)
        random_S.append(r_S)

    S_max  = L / 2   # maximum possible half-chain entropy
    adap   = float(np.mean(adaptive_S))
    bound  = float(np.mean(boundary_S))
    rand   = float(np.mean(random_S))

    return {
        'adaptive':          adap,
        'adaptive_std':      float(np.std(adaptive_S)),
        'boundary':          bound,
        'boundary_std':      float(np.std(boundary_S)),
        'random':            rand,
        'random_std':        float(np.std(random_S)),
        'gain_vs_boundary':  (adap - bound) / S_max,
        'gain_vs_random':    (adap - rand)  / S_max,
        'S_max':             S_max,
        'sample_trajectory': sample_trajectory,
    }


# =====================================================================
# Figures
# =====================================================================

def _save_fig(fig, stem: str, outdir: str = "results/figures") -> None:
    """Save a figure as PNG and PDF into results/figures, then close it."""
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/{stem}.{ext}", dpi=200)
    plt.close(fig)


def make_plots(all_results: list) -> None:
    """
    Produce the Stage 6 figures from the collected results, one pair per
    (L, k) configuration:

        stage6_training_curve_L{L}_k{k} : training entropy vs batch --
            shows the policy learning to hold entanglement over training.
        stage6_entropy_trajectory_L{L}_k{k} : S(t) vs circuit layer for a
            single trained episode -- shows the adaptive controller
            sustaining entropy across the circuit, with the boundary and
            random baselines' final values marked for comparison.

    Figures are written to results/figures/ as PNG + PDF. A configuration
    is skipped silently if the data needed for a panel is unavailable.
    """
    for r in all_results:
        L, k = r['L'], r['k']
        e = r['eval']

        # --- training curve (entropy per training batch) ---
        hist = r.get('training_entropy')
        if hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(hist, lw=1.4)
            ax.set_xlabel("training batch")
            ax.set_ylabel(r"mean episode entropy $S$")
            ax.set_title(fr"Stage 6 training curve  ($L={L}$, $k={k}$)")
            _save_fig(fig, f"stage6_training_curve_L{L}_k{k}")

        # --- entropy trajectory of one trained episode ---
        traj = e.get('sample_trajectory')
        if traj is not None:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(traj, "-o", ms=3, lw=1.4, label="adaptive policy")
            ax.axhline(e['boundary'], ls="--", color="grey",
                       label=f"boundary (final {e['boundary']:.2f})")
            ax.axhline(e['random'], ls=":", color="grey",
                       label=f"random (final {e['random']:.2f})")
            ax.set_xlabel("circuit layer")
            ax.set_ylabel(r"half-chain entropy $S(L/2)$")
            ax.set_title(fr"Stage 6 entropy trajectory  ($L={L}$, $k={k}$)")
            ax.legend(fontsize=8)
            _save_fig(fig, f"stage6_entropy_trajectory_L{L}_k{k}")


# =====================================================================
# Main
# =====================================================================

def main():
    cfg = RLConfig()
    t0  = time.time()
    all_results = []

    for L in cfg.L_values:
        depth = cfg.depth(L)
        S_max = L / 2

        for k_frac in cfg.k_values:
            k = max(1, int(round(k_frac * L)))
            print(f"\n{'='*60}")
            print(f"  L={L}  k={k}  (meas rate = {k/L:.2f})")
            print(f"{'='*60}")

            params, history = train_policy(
                L=L, depth=depth, k_per_layer=k,
                n_batches=cfg.n_batches, batch_size=cfg.batch_size,
                lr=cfg.rl_lr, window=cfg.window,
                baseline_alpha=cfg.baseline_alpha,
                entropy_coeff=cfg.entropy_coeff,
                pretrain_episodes=cfg.n_supervised_eps,
                pretrain_epochs=cfg.pretrain_epochs)

            print(f"\n  Evaluating ({cfg.n_eval} episodes)...")
            eval_res = evaluate(params, L, depth, k, cfg.n_eval, cfg.window)

            print(f"  S adaptive = {eval_res['adaptive']:.3f} ± {eval_res['adaptive_std']:.3f}")
            print(f"  S boundary = {eval_res['boundary']:.3f} ± {eval_res['boundary_std']:.3f}")
            print(f"  ΔS/S_max   = {eval_res['gain_vs_boundary']:+.4f}")

            all_results.append({
                'L': L, 'k': k, 'k_frac': k_frac,
                'eval': eval_res,
                'training_entropy': history['entropy'],
            })

    save = {'results': all_results, 'runtime_min': (time.time()-t0)/60}
    with open('results/rl_controller.json', 'w') as f:
        json.dump(save, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ALL DONE — {(time.time()-t0)/60:.1f} min")
    print(f"{'='*60}")
    print("Saved → results/rl_controller.json")

    print("\nPlotting...")
    make_plots(all_results)
    print("Figures → results/figures/")

    # Summary table
    print(f"\n{'L':>4} {'k':>4} {'S_adap':>8} {'S_bound':>8} {'ΔS/S_max':>10}")
    print('-' * 40)
    for r in all_results:
        e = r['eval']
        print(f"{r['L']:>4} {r['k']:>4} {e['adaptive']:>8.3f} "
              f"{e['boundary']:>8.3f} {e['gain_vs_boundary']:>+10.4f}")


if __name__ == '__main__':
    main()
