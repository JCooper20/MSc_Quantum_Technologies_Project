"""
Two-phase training pipeline for the adaptive measurement policy.

[Phase 1] Supervised pre-training:
 - Minimise MSE loss against boundary-heuristic scores to give the
   policy a physically informed warm start before RL:

        L_supervised = (1/N) Σᵢ ||π(sᵢ) - d(qᵢ)||²

    where d(q) = |q - L/2| / (L/2) is the boundary distance score.

[Phase 2] REINFORCE fine-tuning:
  - Policy gradient optimisation maximising expected entanglement:

        ∇_θ J(θ) = E_τ[Σ_t ∇_θ log π_θ(aₜ|sₜ) · (R - b)]

    where:
        R — final half-chain entropy S(L/2) (reward signal)
        b — exponential moving average baseline: b ← (1-α)b + αR
            reduces variance without introducing bias
        π_θ(aₜ|sₜ) — Plackett-Luce probability of top-k selection

    Entropy regularisation encourages exploration:
        L_RL = -∇_θ J(θ) + β · H(π_θ)
"""
# Imports
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from typing import Dict, List, Tuple
from src.models.policy import (policy_forward, boundary_scores,
run_episode_adaptive, run_episode_boundary,_apply_layer)
from src.training.adam import AdamOptimizer
import stim

SEED = 42

# ===================================
# Supervised Learning - Pre Training
# ===================================

def generate_supervised_data(L: int, depth: int, k_per_layer: int,n_episodes: int = 500,
                              window: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (state, target_score) pairs from the boundary-avoiding oracle
    for supervised pre-training of the policy network.

    Each episode runs a full circuit trajectory under the boundary heuristic, 
    measuring the k qubits furthest from the bipartition cut — and records
    the policy input state at every timestep alongside the target score:

        d(q) = |q - L/2| / (L/2) ∈ [0, 1] — boundary distance score

    State vector at timestep t (input_dim = window·2L + 1):

        s_t = [m_{t-W}⁰,...,m_{t-W}^{L-1}, o_{t-W}⁰,...,  ]
              [                    ...                        ] ← W layers
              [m_{t}⁰,  ...,m_{t}^{L-1},   o_{t}⁰,...      ]
              [t/depth] ← time fraction

    where W = window, m = was-measured mask, o = outcomes (Zero-padded for t < W)

    Parameters:
    - L = number of qubits
    - depth = number of circuit layers (typically 4L)
    - k_per_layer = number of qubits measured per layer
    - n_episodes = number of trajectory episodes to generate
    - window = number of past layers in state vector W

    Returns;
    - X = state vectors (n_episodes·depth, window·2L + 1)
    - y = boundary distance scores (n_episodes·depth, L)    
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
    """
    Supervised pre-training of the policy network via MSE regression
    against boundary distance scores.

    Minimises:
        L_MSE = (1/N) Σᵢ ||π_θ(sᵢ) - d(qᵢ)||²

    where:
        π_θ(sᵢ) ∈ ℝ^L  — policy network output (qubit scores)
        d(qᵢ)   ∈ ℝ^L  — boundary distance targets d(q) = |q - L/2| / (L/2)

    Gives the policy a physically informed warm start before REINFORCE —
    the network learns to score qubits by distance from the bipartition
    cut before any RL signal is applied.

    Each epoch:
        1. Shuffle training set
        2. For each mini-batch:
            - g = ∇_θ L_MSE(θ, x_batch, y_batch)
            - θ = Adam(θ, g)

    Parameters:
    - params = initial policy parameter dict
    - X = state vectors (N, window·2L + 1)
    - y = boundary distance score targets (N, L)
    - epochs = number of full passes over training set
    - batch_size = number of samples per gradient update
    - lr = Adam learning rate

    Returns:
    - params = pre-trained policy parameter dict
    """
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


# =======================
#  REINFORCE Fine-Tuning
# =======================

def reinforce_update(params, states: np.ndarray,
                      log_probs: np.ndarray, advantage: float):
    """
    Compute REINFORCE policy gradient for a single episode.

    Surrogate loss (differentiable w.r.t. policy parameters θ):

       - L = -A · (1/T) Σ_t log_probs_t · Σ_q log π_θ(q | s_t)

    where:
      - A = advantage = R - b  (reward minus EMA baseline)
      - log_probs = stored Plackett-Luce log P(top-k selection) at each t
                     treated as importance weights (stop gradient)
      - log π_θ = log-softmax of policy scores, differentiable w.r.t. θ

    The Plackett-Luce probability of selecting top-k qubits {q₁,...,qₖ}:

        log P = Σⱼ log( exp(sⱼ) / Σ_{q∉{q₁,...,qⱼ₋₁}} exp(sq) )

    Gradients flow through log π_θ only — log_probs are fixed constants
    from the episode, making this a valid policy gradient estimator.

    Parameters:
    - params = current policy parameter dict
    - states    : state vectors from episode (T, window·2L + 1
    - log_probs : Plackett-Luce log-probs of actions taken
    - advantage = R - b (scaled reward signal)

    Returns:
    - g = policy gradient (same pytree structure as params)
    """
    @jit
    def loss_fn(p):
        x       = jnp.array(states)
        scores  = policy_forward(p, x) # (T, L)
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
    Full two-phase policy training: supervised pre-train → REINFORCE.

    [Phase 1] Supervised warm-start:
        Train policy to predict boundary distance scores via MSE.
        Provides physically informed initialisation before RL.

    [Phase 2] REINFORCE fine-tuning:
        For each batch of episodes:
            1. Run batch_size episodes under current policy π_θ
            2. Update EMA baseline:
                   b ← (1-α)·b + α·⟨S(L/2)⟩_batch
            3. Compute per-episode advantage:
                   A_i = S_i(L/2) - b
            4. Accumulate policy gradient:
                   g = (1/B) Σᵢ ∇_θ L_REINFORCE(θ, episode_i, A_i)
            5. Update: θ = Adam(θ, g)

    Improvement tracked against boundary heuristic baseline:
        ΔS = ⟨S(L/2)⟩_adaptive - ⟨S(L/2)⟩_boundary

    Parameters:
    - L = number of qubits
    - depth = number of circuit layers (typically 4L)
    - k_per_layer = measurement budget per layer
    - n_batches = number of REINFORCE update steps
    - batch_size = episodes per gradient update B
    - lr = Adam learning rate for REINFORCE
    - window = state vector history length W
    - baseline_alpha = EMA decay α for variance reduction
    - entropy_coeff = entropy regularisation coefficient β
    - pretrain_episodes = episodes for supervised data generation
    - pretrain_epochs = epochs for supervised pre-training

    Returns:
    - params = trained policy parameter dict
    - history = 'entropy', 'baseline', 'improvement' per batch (dict)
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
