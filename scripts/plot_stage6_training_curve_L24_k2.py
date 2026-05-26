"""
scripts/plot_stage6_training_curve_L24_k2.py
---------------------------------------------
Single REINFORCE run for L=24, k=2, n_batches=150.
Plots S(L/2) vs batch number to visualise entropy rise during training.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from src.training.reinforce import train_policy


def main():
    L, k, depth = 24, 2, 96
    print(f"Training: L={L}  k={k}  depth={depth}  n_batches=150")
    t0 = time.time()

    params, history = train_policy(
        L=L, depth=depth, k_per_layer=k,
        n_batches=150, batch_size=48, lr=3e-4,
        pretrain_episodes=500, pretrain_epochs=30,
    )

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min.")

    batches   = np.arange(1, len(history['entropy']) + 1)
    entropy   = np.array(history['entropy'])
    baseline  = np.array(history['baseline'])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(batches, entropy,  lw=1.5, label=r'$\langle S(L/2) \rangle$ (adaptive)')
    ax.plot(batches, baseline, lw=1.5, linestyle='--', label='EMA baseline')
    ax.set_xlabel('Batch number')
    ax.set_ylabel(r'$S(L/2)$')
    ax.set_title(r'REINFORCE training curve — $L=24$, $k=2$')
    ax.legend()
    fig.tight_layout()
    fig.savefig('results/figures/stage6_training_curve_L24_k2.png', dpi=200)
    fig.savefig('results/figures/stage6_training_curve_L24_k2.pdf')
    print("Saved → results/figures/stage6_training_curve_L24_k2.png/.pdf")


if __name__ == '__main__':
    main()
