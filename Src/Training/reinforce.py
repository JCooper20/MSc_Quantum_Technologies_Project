"""
training/reinforce.py
---------------------
Supervised pre-training and REINFORCE fine-tuning for the adaptive
measurement policy (Stage 6).

Two-phase training:
  1. Supervised pre-train: MSE loss against boundary-heuristic scores.
     Gives the policy a sensible warm start.
  2. REINFORCE fine-tuning: policy gradient with EMA baseline and
     optional entropy regularisation.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from typing import Dict, List, Tuple

from src.models.policy import (
    policy_forward, boundary_scores,
    run_episode_adaptive, run_episode_boundary,
    _apply_layer
)
from src.training.adam import AdamOptimizer
import stim

SEED = 42


# ============================================================================
# SUPERVISED PRE-TRAINING
# ============================================================================

def generate_supervised_data(L: int, depth: int, k_per_layer: int,
                              n_episodes: int = 500,
                              window: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (state, target_scores) pairs from the boundary-avoiding oracle.
    The target is the static boundary_scores vector for every time step.
    """
    print(f"    Generating {n_episodes} supervised episodes...", flush=True)
    input_dim      = window * 2 * L + 1
    target_scores  = boundary_scores(L)
    X_list, y_list = [], []

    for ep in range(n_episodes):
        sim          = stim.TableauSimulator()
        sim.set_num_qubits(L)
        was_measured = np.zeros((depth, L), dtype=np.float32)
        outcomes     = np.zeros((depth, L), dtype=np.float32)

        for t in range(depth):
            _apply_layer(sim, L, t)
            state = np.zeros(input_dim, dtype=np.float32)
            for w in range(window):
                li = t - window + w
                if li >= 0:
                    base = w * 2 * L
                    state[base:base + L]       = was_measured[li]
                    state[base + L:base + 2*L] = outcomes[li]
            state[-1] = t / depth
            X_list.append(state)
            y_list.append(target_scores.copy())

            # Execute boundary heuristic
            noisy = target_scores + np.random.normal(0, 0.01, L)
            if k_per_layer > 0:
                for q in np.argsort(noisy)[-k_per_layer:]:
                    res                = sim.measure(q)
                    was_measured[t, q] = 1.0
                    outcomes[t, q]     = float(res)

        if (ep + 1) % 100 == 0:
            print(f"      {ep+1}/{n_episodes}", flush=True)

    return np.array(X_list), np.array(y_list)


def supervised_pretrain(params, X: np.ndarray, y: np.ndarray,
                         epochs: int = 30, batch_size: int = 128,
                         lr: float = 1e-3):
    """MSE pre-training to predict boundary distance scores."""
    print(f"    Pre-training: {len(X)} samples, {epochs} epochs", flush=True)
    opt  = AdamOptimizer(params, lr=lr)
    X_j  = jnp.array(X)
    y_j  = jnp.array(y)

    def mse_loss(p, xb, yb):
        return jnp.mean((policy_forward(p, xb) - yb) ** 2)

    grad_fn = jit(grad(mse_loss))
    loss_fn = jit(mse_loss)
    n       = len(X)

    for epoch in range(epochs):
        perm       = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, n, batch_size):
            idx    = perm[start:start + batch_size]
            g      = grad_fn(params, X_j[idx], y_j[idx])
            params = opt.step(params, g)
            epoch_loss += float(loss_fn(params, X_j[idx], y_j[idx]))
            n_batches  += 1
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{epochs}: "
                  f"MSE = {epoch_loss/n_batches:.5f}")

    return params


# ============================================================================
# REINFORCE FINE-TUNING
# ============================================================================

def reinforce_update(params, states: np.ndarray,
                      log_probs: np.ndarray, advantage: float):
    """
    REINFORCE surrogate loss (differentiable w.r.t. policy scores).

        L = -advantage · Σ_t log π(a_t | s_t)

    where log π is the Plackett-Luce log-probability of the chosen
    top-k subset — approximated by re-computing scores and using the
    stored log-probs scaled by the advantage.
    """
    @jit
    def loss_fn(p):
        x       = jnp.array(states)
        scores  = policy_forward(p, x)      # (T, L)
        # Differentiable surrogate: weighted sum of log-softmax scores
        log_pi  = jax.nn.log_softmax(scores, axis=-1)
        # Use stored log_probs as importance weights (stop gradient)
        lp      = jnp.array(log_probs)
        return -advantage * jnp.mean(lp * jnp.sum(log_pi, axis=-1))

    g = grad(loss_fn)(params)
    return g


def train_policy(L: int, depth: int, k_per_layer: int,
                  n_batches: int = 150, batch_size: int = 48,
                  lr: float = 3e-4, window: int = 4,
                  baseline_alpha: float = 0.05,
                  entropy_coeff: float = 0.01,
                  pretrain_episodes: int = 500,
                  pretrain_epochs: int = 30) -> Tuple:
    """
    Full two-phase training: supervised pre-train → REINFORCE.

    Returns (trained_params, training_history_dict)
    """
    import jax.random as jrandom
    input_dim = window * 2 * L + 1

    # Phase 1: supervised warm-start
    key    = jrandom.PRNGKey(SEED)
    from src.models.policy import init_policy_params
    params = init_policy_params(key, input_dim, L, hidden=128)

    X_sup, y_sup = generate_supervised_data(
        L, depth, k_per_layer, n_episodes=pretrain_episodes, window=window)
    params = supervised_pretrain(
        params, X_sup, y_sup,
        epochs=pretrain_epochs, lr=1e-3)

    # Phase 2: REINFORCE
    opt      = AdamOptimizer(params, lr=lr)
    baseline = 0.0
    history  = {'entropy': [], 'baseline': [], 'improvement': []}

    # Reference: boundary heuristic entropy for comparison
    ref_ents = [run_episode_boundary(L, depth, k_per_layer)[0]
                for _ in range(20)]
    ref_mean = float(np.mean(ref_ents))

    for batch_idx in range(n_batches):
        batch_entropies, all_states, all_lp, all_adv = [], [], [], []

        for _ in range(batch_size):
            ep = run_episode_adaptive(
                L, depth, k_per_layer, policy_forward, params,
                window=window)
            batch_entropies.append(ep['final_entropy'])
            all_states.append(ep['states'])
            all_lp.append(ep['log_probs'])

        mean_ent = float(np.mean(batch_entropies))
        baseline = (1 - baseline_alpha) * baseline + baseline_alpha * mean_ent

        # Update with each episode's advantage
        total_grad = jax.tree.map(jnp.zeros_like, params)
        for ent, states, lp in zip(batch_entropies, all_states, all_lp):
            advantage = ent - baseline
            g         = grad(lambda p: reinforce_update(p, states, lp, advantage))(params)
            total_grad = jax.tree.map(lambda tg, gi: tg + gi / batch_size,
                                       total_grad, g)

        params = opt.step(params, total_grad)

        history['entropy'].append(mean_ent)
        history['baseline'].append(baseline)
        history['improvement'].append(mean_ent - ref_mean)

        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx+1:3d} | S = {mean_ent:.3f} | "
                  f"baseline = {baseline:.3f} | "
                  f"vs boundary = {mean_ent - ref_mean:+.3f}")

    return params, history
