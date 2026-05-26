"""
PPO training pipeline for the adaptive measurement policy.

[Phase 1] Supervised pre-training:
 - Identical warm-start to reinforce.py — minimise MSE against
   boundary-heuristic scores to give the policy a physically
   informed initialisation before RL:

        L_supervised = (1/N) Σᵢ ||π(sᵢ) - d(qᵢ)||²

    where d(q) = |q - L/2| / (L/2) is the boundary distance score.

[Phase 2] PPO fine-tuning:
  - Clipped surrogate objective (Schulman et al. 2017):

        L_CLIP(θ) = E_t[ min( r_t(θ) · Â_t,
                              clip(r_t(θ), 1-ε, 1+ε) · Â_t ) ]

    where:
        r_t(θ) = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)   — probability ratio
        Â_t    — GAE advantage estimate (see below)
        ε      — clipping threshold (default 0.2)

  - Value function baseline trained alongside policy:

        L_VF(φ) = (1/T) Σ_t (V_φ(sₜ) - Vₜ_target)²

        Vₜ_target = Â_t + V_φ(sₜ)   (bootstrapped target)

  - Generalised Advantage Estimation (Schulman et al. 2016):

        δ_t  = R_t + γ · V(s_{t+1}) - V(s_t)

        Â_t  = Σ_{l≥0} (γλ)^l · δ_{t+l}

    where:
        R_t = S_t(L/2) - S_{t-1}(L/2)   — per-step entropy gain
        γ   — discount factor (default 0.99)
        λ   — GAE smoothing parameter (default 0.95)

  - Combined loss per PPO epoch:

        L(θ,φ) = -L_CLIP(θ) + c_vf · L_VF(φ) - c_ent · H(π_θ)

    where:
        c_vf  — value function loss coefficient (default 0.5)
        c_ent — entropy bonus coefficient (default 0.01)

  - π_θ(aₜ|sₜ) uses the Plackett-Luce top-k probability:

        log P = Σⱼ log( exp(sⱼ) / Σ_{q∉{q₁,...,qⱼ₋₁}} exp(sq) )
"""
# Imports
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from typing import Dict, List, Tuple
from src.models.policy import (policy_forward, boundary_scores,
    run_episode_adaptive, run_episode_boundary, _apply_layer)
from src.training.adam import AdamOptimizer
from src.training.reinforce import (generate_supervised_data,
    supervised_pretrain)
import stim

SEED = 42


# ============================================================
# Value Function Network
# ============================================================

def init_value_params(key, input_dim: int, hidden: int = 64):
    """
    Initialise a two-layer MLP value function V_φ(s) → scalar.

    Architecture:
        s_t → FC(hidden) → FC(32) → FC(1) → V ∈ ℝ

        Layer 1: (input_dim, hidden) — vw1, vb1
        Layer 2: (hidden, 32)       — vw2, vb2
        Layer 3: (32, 1)            — vw3, vb3

    Weights ~ N(0, 0.02²), biases = 0.
    Kept deliberately smaller than the policy network to
    avoid value overfitting dominating training.

    Parameters:
    - key = JAX PRNG key
    - input_dim = state vector dimension (W·2L + 1)
    - hidden = hidden layer width (default 64)

    Returns:
    - vparams = dict with keys vw1, vb1, vw2, vb2, vw3, vb3
    """
    import jax.random as jrandom
    keys = jrandom.split(key, 6)
    s    = 0.02
    return {
        'vw1': jrandom.normal(keys[0], (input_dim, hidden)) * s,
        'vb1': jnp.zeros(hidden),
        'vw2': jrandom.normal(keys[1], (hidden, 32)) * s,
        'vb2': jnp.zeros(32),
        'vw3': jrandom.normal(keys[2], (32, 1)) * s,
        'vb3': jnp.zeros(1),
    }


def value_forward(vparams, x):
    """
    Forward pass of the value network V_φ(s).

    Architecture:  s → ReLU(FC) → ReLU(FC) → FC → scalar

    Parameters:
    - vparams = value network parameter dict
    - x = state input (N, input_dim)

    Returns:
    - V ∈ ℝ^N  — predicted state values, shape (N,)
    """
    h = jax.nn.relu(x @ vparams['vw1'] + vparams['vb1'])
    h = jax.nn.relu(h @ vparams['vw2'] + vparams['vb2'])
    return (h @ vparams['vw3'] + vparams['vb3']).squeeze(-1)


# ============================================================
# GAE Advantage Estimation
# ============================================================

def compute_gae(rewards: np.ndarray, values: np.ndarray,
                gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalised Advantage Estimates (GAE) for a single episode.

    Temporal difference residuals:
        δ_t = R_t + γ · V(s_{t+1}) - V(s_t)

    with V(s_T) = 0 at the terminal step (no bootstrap beyond episode end).

    GAE advantage:
        Â_t = Σ_{l≥0} (γλ)^l · δ_{t+l}

    Computed backwards in O(T) via the recurrence:
        Â_T     = δ_T
        Â_{t}   = δ_t + γλ · Â_{t+1}

    Value targets (used for L_VF regression):
        Vₜ_target = Â_t + V(s_t)

    Parameters:
    - rewards : per-step rewards R_t = S_t(L/2) - S_{t-1}(L/2),  shape (T,)
    - values  : value network estimates V_φ(s_t),                  shape (T,)
    - gamma   : discount factor γ (default 0.99)
    - lam     : GAE smoothing parameter λ (default 0.95)

    Returns:
    - advantages   : Â_t  shape (T,)
    - value_targets: Vₜ_target shape (T,)
    """
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae        = 0.0

    for t in reversed(range(T)):
        next_val    = values[t + 1] if t + 1 < T else 0.0
        delta       = rewards[t] + gamma * next_val - values[t]
        gae         = delta + gamma * lam * gae
        advantages[t] = gae

    value_targets = advantages + values
    return advantages, value_targets


# ============================================================
# PPO Surrogate Objective
# ============================================================

def ppo_loss(policy_params, vparams, states: np.ndarray,
             log_probs_old: np.ndarray, advantages: np.ndarray,
             value_targets: np.ndarray,
             clip_eps: float = 0.2,
             c_vf: float = 0.5,
             c_ent: float = 0.01):
    """
    Combined PPO loss over a single episode rollout.

    Policy loss — clipped surrogate objective:
        L_CLIP = (1/T) Σ_t min( r_t · Â_t, clip(r_t, 1-ε, 1+ε) · Â_t )

    Value function loss — MSE against bootstrapped targets:
        L_VF = (1/T) Σ_t (V_φ(s_t) - Vₜ_target)²

    Entropy bonus — encourages exploration:
        H = -(1/T) Σ_t Σ_q π_θ(q|s_t) · log π_θ(q|s_t)

    Combined:
        L = -L_CLIP + c_vf · L_VF - c_ent · H

    The probability ratio r_t uses the Plackett-Luce log-prob:
        log π_θ(aₜ|sₜ) = Σⱼ log-softmax scores at selected qubits
    which approximates the full selection log-prob differentiably.

    Parameters:
    - policy_params  : current policy parameter dict
    - vparams        : current value network parameter dict
    - states         : state vectors (T, input_dim)
    - log_probs_old  : old Plackett-Luce log P (stop-gradient),  shape (T,)
    - advantages     : GAE advantages Â_t,                        shape (T,)
    - value_targets  : bootstrapped value targets,                shape (T,)
    - clip_eps       : clipping threshold ε (default 0.2)
    - c_vf           : value function loss coefficient (default 0.5)
    - c_ent          : entropy bonus coefficient (default 0.01)

    Returns:
    - scalar loss to minimise
    """
    x        = jnp.array(states)               # (T, input_dim)
    scores   = policy_forward(policy_params, x) # (T, L)
    log_pi   = jax.nn.log_softmax(scores, axis=-1)  # (T, L)

    # Differentiable log π_θ(aₜ|sₜ) ≈ sum of log-softmax scores per step
    # (same surrogate used in reinforce_update, consistent across files)
    log_probs_new = jnp.sum(log_pi, axis=-1)   # (T,)
    lp_old        = jnp.array(log_probs_old)   # (T,)
    adv           = jnp.array(advantages)      # (T,)

    # Probability ratio r_t = exp(log π_new - log π_old)
    ratio = jnp.exp(log_probs_new - lp_old)

    # Clipped surrogate
    surr_unclipped = ratio * adv
    surr_clipped   = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss    = -jnp.mean(jnp.minimum(surr_unclipped, surr_clipped))

    # Value function loss
    V          = value_forward(vparams, x)          # (T,)
    vf_loss    = jnp.mean((V - jnp.array(value_targets)) ** 2)

    # Entropy bonus H(π) = -Σ_q π log π averaged over timesteps
    pi         = jax.nn.softmax(scores, axis=-1)
    entropy    = -jnp.mean(jnp.sum(pi * log_pi, axis=-1))

    return policy_loss + c_vf * vf_loss - c_ent * entropy


# ============================================================
# PPO Training
# ============================================================

def train_policy_ppo(L: int, depth: int, k_per_layer: int,
                     n_batches: int = 150, batch_size: int = 48,
                     lr: float = 3e-4, window: int = 4,
                     gamma: float = 0.99, lam: float = 0.95,
                     clip_eps: float = 0.2,
                     ppo_epochs: int = 4,
                     c_vf: float = 0.5,
                     c_ent: float = 0.01,
                     pretrain_episodes: int = 500,
                     pretrain_epochs: int = 30) -> Tuple:
    """
    Full two-phase policy training: supervised pre-train → PPO.

    [Phase 1] Supervised warm-start:
        Identical to train_policy in reinforce.py — train policy
        to predict boundary distance scores via MSE.

    [Phase 2] PPO fine-tuning:
        For each batch of rollouts:
            1. Run batch_size episodes under current policy π_θ_old,
               recording (s_t, a_t, log_π_old(aₜ), R_t, S_t) per step
            2. Compute per-step rewards: R_t = S_t(L/2) - S_{t-1}(L/2)
            3. Estimate V_φ(s_t) for all steps
            4. Compute GAE advantages Â_t and value targets Vₜ_target
            5. For ppo_epochs update steps over the collected rollout:
                   g = ∇_{θ,φ} L(θ,φ)   [clipped surrogate + VF + entropy]
                   θ,φ = Adam(θ,φ, g)
            6. Update π_θ_old ← π_θ

    Improvement tracked against boundary heuristic baseline:
        ΔS = ⟨S(L/2)⟩_adaptive - ⟨S(L/2)⟩_boundary

    Parameters:
    - L = number of qubits
    - depth = number of circuit layers (typically 4L)
    - k_per_layer = measurement budget per layer
    - n_batches = number of PPO rollout → update cycles
    - batch_size = episodes per rollout batch B
    - lr = Adam learning rate for both policy and value networks
    - window = state vector history length W
    - gamma = GAE discount factor γ (default 0.99)
    - lam = GAE smoothing parameter λ (default 0.95)
    - clip_eps = PPO clipping threshold ε (default 0.2)
    - ppo_epochs = gradient update steps per batch (default 4)
    - c_vf = value function loss coefficient (default 0.5)
    - c_ent = entropy bonus coefficient (default 0.01)
    - pretrain_episodes = episodes for supervised data generation
    - pretrain_epochs = epochs for supervised pre-training

    Returns:
    - policy_params = trained policy parameter dict
    - vparams = trained value network parameter dict
    - history = 'entropy', 'value_loss', 'improvement' per batch (dict)
    """
    import jax.random as jrandom
    from src.models.policy import init_policy_params
    input_dim = window * 2 * L + 1

    # Phase 1: supervised warm-start (identical to train_policy)
    key           = jrandom.PRNGKey(SEED)
    key, vkey     = jrandom.split(key)
    policy_params = init_policy_params(key, input_dim, L, hidden=128)
    vparams       = init_value_params(vkey, input_dim, hidden=64)

    X_sup, y_sup = generate_supervised_data(
        L, depth, k_per_layer, n_episodes=pretrain_episodes, window=window)
    policy_params = supervised_pretrain(
        policy_params, X_sup, y_sup,
        epochs=pretrain_epochs, lr=1e-3)

    # Phase 2: PPO
    policy_opt = AdamOptimizer(policy_params, lr=lr)
    value_opt  = AdamOptimizer(vparams,       lr=lr)
    history    = {'entropy': [], 'value_loss': [], 'improvement': []}

    # Gradient function over combined (policy_params, vparams) pair
    grad_fn = jit(grad(ppo_loss, argnums=(0, 1)))

    # Reference: boundary heuristic entropy for comparison
    ref_ents = [run_episode_boundary(L, depth, k_per_layer)[0]
                for _ in range(20)]
    ref_mean = float(np.mean(ref_ents))

    for batch_idx in range(n_batches):
        # ── Collect rollout ───────────────────────────────────────────────
        all_states, all_lp_old, all_rewards, all_final = [], [], [], []

        for _ in range(batch_size):
            ep = run_episode_adaptive(
                L, depth, k_per_layer, policy_forward, policy_params,
                window=window, entropy_interval=1)

            entropies = np.array([s for _, s in ep['entropy_snapshots']])
            prev      = np.concatenate([[0.0], entropies[:-1]])
            rewards   = entropies - prev   # R_t = S_t - S_{t-1}

            all_states.append(ep['states'])
            all_lp_old.append(ep['log_probs'])
            all_rewards.append(rewards)
            all_final.append(ep['final_entropy'])

        # ── PPO update epochs ─────────────────────────────────────────────
        total_vf_loss = 0.0

        for _ in range(ppo_epochs):
            total_pg   = jax.tree.map(jnp.zeros_like, policy_params)
            total_vg   = jax.tree.map(jnp.zeros_like, vparams)

            for states, lp_old, rewards in zip(
                    all_states, all_lp_old, all_rewards):

                # Estimate values and compute GAE
                x_ep  = jnp.array(states)
                vals  = np.array(value_forward(vparams, x_ep))
                advantages, value_targets = compute_gae(
                    rewards, vals, gamma=gamma, lam=lam)

                # Normalise advantages per-episode (reduces variance)
                adv_mean = advantages.mean()
                adv_std  = advantages.std() + 1e-8
                advantages = (advantages - adv_mean) / adv_std

                # Compute gradients
                pg, vg = grad_fn(
                    policy_params, vparams, states, lp_old,
                    advantages, value_targets,
                    clip_eps, c_vf, c_ent)

                total_pg = jax.tree.map(
                    lambda tg, gi: tg + gi / batch_size, total_pg, pg)
                total_vg = jax.tree.map(
                    lambda tg, gi: tg + gi / batch_size, total_vg, vg)

                # Track value loss for logging
                x_j  = jnp.array(states)
                V    = value_forward(vparams, x_j)
                total_vf_loss += float(
                    jnp.mean((V - jnp.array(value_targets)) ** 2))

            policy_params = policy_opt.step(policy_params, total_pg)
            vparams       = value_opt.step(vparams,       total_vg)

        mean_ent  = float(np.mean(all_final))
        mean_vloss = total_vf_loss / (ppo_epochs * batch_size)
        history['entropy'].append(mean_ent)
        history['value_loss'].append(mean_vloss)
        history['improvement'].append(mean_ent - ref_mean)

        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx+1:3d} | S = {mean_ent:.3f} | "
                  f"VF loss = {mean_vloss:.4f} | "
                  f"vs boundary = {mean_ent - ref_mean:+.3f}")

    return policy_params, vparams, history
