"""
scripts/run_stage6_rl.py
------------------------
Stage 6: Adaptive RL Controller

Trains a REINFORCE policy to adaptively select which qubits to measure
at each circuit layer, targeting high entanglement (volume-law) at
measurement rates that would otherwise drive area-law behaviour.

Three baselines:
  - Random measurements at rate k/L
  - Boundary-avoiding heuristic (measures qubits furthest from cut)
  - Sign-Colour Decoder (where tractable)

Key metric: ΔS/S_max = (S_adaptive - S_boundary) / S_max
"""

import numpy as np
import json
import time

from src.config import RLConfig
from src.models.policy import (
    run_episode_random, run_episode_boundary, run_episode_adaptive,
    policy_forward, init_policy_params, boundary_scores,
)
from src.training.reinforce import train_policy
from src.analysis.entropy import stabiliser_entropy


def evaluate(params, L: int, depth: int, k_per_layer: int,
             n_eval: int, window: int = 4) -> dict:
    """Compare adaptive vs random vs boundary at matched measurement rate."""
    adaptive_S, boundary_S, random_S = [], [], []

    for _ in range(n_eval):
        ep = run_episode_adaptive(L, depth, k_per_layer,
                                   policy_forward, params, window=window)
        adaptive_S.append(ep['final_entropy'])

        b_S, _ = run_episode_boundary(L, depth, k_per_layer)
        boundary_S.append(b_S)

        r_S, _ = run_episode_random(L, depth, k_per_layer / L)
        random_S.append(r_S)

    S_max  = L / 2   # maximum possible half-chain entropy
    adap   = float(np.mean(adaptive_S))
    bound  = float(np.mean(boundary_S))
    rand   = float(np.mean(random_S))

    return {
        'adaptive':         adap,
        'adaptive_std':     float(np.std(adaptive_S)),
        'boundary':         bound,
        'boundary_std':     float(np.std(boundary_S)),
        'random':           rand,
        'random_std':       float(np.std(random_S)),
        'gain_vs_boundary': (adap - bound) / S_max,
        'gain_vs_random':   (adap - rand)  / S_max,
        'S_max':            S_max,
    }


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

    # Summary table
    print(f"\n{'L':>4} {'k':>4} {'S_adap':>8} {'S_bound':>8} {'ΔS/S_max':>10}")
    print('-' * 40)
    for r in all_results:
        e = r['eval']
        print(f"{r['L']:>4} {r['k']:>4} {e['adaptive']:>8.3f} "
              f"{e['boundary']:>8.3f} {e['gain_vs_boundary']:>+10.4f}")


if __name__ == '__main__':
    main()
