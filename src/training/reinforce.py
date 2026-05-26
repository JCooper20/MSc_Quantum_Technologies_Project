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
from src.models.policy import (policy_forward, tcn_policy_forward,
boundary_scores, run_episode_adaptive, run_episode_adaptive_tcn,
run_episode_boundary, _apply_layer)
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
            g         = reinforce_update(params, states, lp, advantage)
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


# ============================================================================
# PER-STEP REINFORCE
# ============================================================================

def reinforce_update_stepwise(params, states: np.ndarray,
                               log_probs: np.ndarray, advantages: np.ndarray):
    """
    Per-step REINFORCE policy gradient where each timestep t carries its
    own advantage signal A_t.

    Surrogate loss (differentiable w.r.t. policy parameters θ):

        L = -(1/T) Σ_t A_t · lp_t · Σ_q log π_θ(q | s_t)

    where:
      - A_t  = R_t - b_t  (per-step reward minus per-step EMA baseline)
      - R_t  = S_t(L/2) - S_{t-1}(L/2)  (incremental entropy gain)
      - lp_t = stored Plackett-Luce log P at step t  (stop gradient)
      - log π_θ = log-softmax of policy scores, differentiable w.r.t. θ

    Parameters:
    - params     : current policy parameter dict
    - states     : state vectors from episode (T, window·2L + 1)
    - log_probs  : Plackett-Luce log-probs of actions taken (T,)
    - advantages : per-step advantages A_t (T,)

    Returns:
    - g = policy gradient (same pytree structure as params)
    """
    @jit
    def loss_fn(p):
        x      = jnp.array(states)                       # (T, input_dim)
        scores = policy_forward(p, x)                     # (T, L)
        log_pi = jax.nn.log_softmax(scores, axis=-1)     # (T, L)
        lp     = jnp.array(log_probs)                    # (T,)
        adv    = jnp.array(advantages)                   # (T,)
        return -jnp.mean(adv * lp * jnp.sum(log_pi, axis=-1))

    return grad(loss_fn)(params)


def train_policy_stepwise(L: int, depth: int, k_per_layer: int,
                           n_batches: int = 150, batch_size: int = 48,
                           lr: float = 3e-4, window: int = 4,
                           baseline_alpha: float = 0.05,
                           pretrain_episodes: int = 500,
                           pretrain_epochs: int = 30) -> Tuple:
    """
    Two-phase policy training using per-layer reward R_t = S_t(L/2) - S_{t-1}(L/2).

    [Phase 1] Identical supervised warm-start to train_policy.

    [Phase 2] Per-step REINFORCE:
        For each batch of episodes:
            1. Run batch_size episodes with entropy_interval=1 to record S_t
            2. Compute per-step rewards: R_t = S_t - S_{t-1}
            3. Update per-step EMA baselines: b_t ← (1-α)b_t + α·⟨R_t⟩_batch
            4. Per-step advantages: A_t = R_t - b_t
            5. Accumulate policy gradient via reinforce_update_stepwise
            6. Update: θ = Adam(θ, g)

    The per-step baseline b_t is a separate EMA for each of the T timesteps,
    reducing variance while keeping the estimator unbiased.

    Parameters mirror train_policy exactly — same hyperparameter interface.

    Returns:
    - params  = trained policy parameter dict
    - history = 'entropy', 'baseline', 'improvement' per batch (dict)
    """
    import jax.random as jrandom
    input_dim = window * 2 * L + 1

    # Phase 1: supervised warm-start (identical to train_policy)
    key    = jrandom.PRNGKey(SEED)
    from src.models.policy import init_policy_params
    params = init_policy_params(key, input_dim, L, hidden=128)

    X_sup, y_sup = generate_supervised_data(
        L, depth, k_per_layer, n_episodes=pretrain_episodes, window=window)
    params = supervised_pretrain(
        params, X_sup, y_sup,
        epochs=pretrain_epochs, lr=1e-3)

    # Phase 2: per-step REINFORCE
    opt        = AdamOptimizer(params, lr=lr)
    baselines  = np.zeros(depth)   # per-step EMA baselines b_t
    history    = {'entropy': [], 'baseline': [], 'improvement': []}

    # Reference: boundary heuristic entropy for comparison
    ref_ents = [run_episode_boundary(L, depth, k_per_layer)[0]
                for _ in range(20)]
    ref_mean = float(np.mean(ref_ents))

    for batch_idx in range(n_batches):
        all_states, all_lp, all_rewards, all_final = [], [], [], []

        for _ in range(batch_size):
            ep = run_episode_adaptive(
                L, depth, k_per_layer, policy_forward, params,
                window=window, entropy_interval=1)

            entropies = np.array([s for _, s in ep['entropy_snapshots']])
            # R_t = S_t - S_{t-1}, with S_{-1} = 0
            prev      = np.concatenate([[0.0], entropies[:-1]])
            rewards   = entropies - prev

            all_states.append(ep['states'])
            all_lp.append(ep['log_probs'])
            all_rewards.append(rewards)
            all_final.append(ep['final_entropy'])

        # Update per-step baselines
        batch_rewards = np.array(all_rewards)    # (B, T)
        mean_rewards  = batch_rewards.mean(0)    # (T,)
        baselines     = ((1 - baseline_alpha) * baselines
                         + baseline_alpha * mean_rewards)

        # Accumulate gradients
        total_grad = jax.tree.map(jnp.zeros_like, params)
        for states, lp, rewards in zip(all_states, all_lp, all_rewards):
            advantages = rewards - baselines     # (T,)
            g = reinforce_update_stepwise(params, states, lp, advantages)
            total_grad = jax.tree.map(lambda tg, gi: tg + gi / batch_size,
                                       total_grad, g)

        params = opt.step(params, total_grad)

        mean_ent = float(np.mean(all_final))
        history['entropy'].append(mean_ent)
        history['baseline'].append(float(baselines.mean()))
        history['improvement'].append(mean_ent - ref_mean)

        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx+1:3d} | S = {mean_ent:.3f} | "
                  f"baseline = {baselines.mean():.3f} | "
                  f"vs boundary = {mean_ent - ref_mean:+.3f}")

    return params, history


# ============================================================================
# TCN POLICY TRAINING — supervised pre-train + REINFORCE with TCN backbone
# ============================================================================

def generate_supervised_data_tcn(L: int, depth: int, k_per_layer: int,
                                  n_episodes: int = 500,
                                  window: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (state_window, target_score) pairs for supervised pre-training
    of the TCN policy network.

    Identical protocol to generate_supervised_data, but the state at each
    timestep is a 2D window tensor rather than a flat vector:

        s_t ∈ ℝ^{W × 2L}  (window of measurement masks + outcomes)

        s_t[w, :L]  = was_measured[t-W+w]   — binary measurement mask
        s_t[w, L:]  = outcomes[t-W+w]        — measurement outcomes
        Earlier rows are zero-padded when t < W.

    Target scores are the same boundary distance oracle used for the MLP:

        d(q) = |q - L/2| / (L/2) ∈ [0, 1]

    Parameters:
    - L           = number of qubits
    - depth       = number of circuit layers (typically 4L)
    - k_per_layer = number of qubits measured per layer
    - n_episodes  = number of trajectory episodes to generate
    - window      = history window length W (input time steps)

    Returns:
    - X = window state tensors  (n_episodes·depth, window, 2L)
    - y = boundary distance scores  (n_episodes·depth, L)
    """
    print(f"    Generating {n_episodes} supervised episodes...", flush=True)
    target_scores  = boundary_scores(L)
    X_list, y_list = [], []

    for ep in range(n_episodes):
        sim          = stim.TableauSimulator()
        sim.set_num_qubits(L)
        was_measured = np.zeros((depth, L), dtype=np.float32)
        outcomes     = np.zeros((depth, L), dtype=np.float32)

        for t in range(depth):
            _apply_layer(sim, L, t)

            # Build window tensor s_t ∈ ℝ^{W × 2L}
            state = np.zeros((window, 2 * L), dtype=np.float32)
            for w in range(window):
                li = t - window + w
                if li >= 0:
                    state[w, :L] = was_measured[li]
                    state[w, L:] = outcomes[li]
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


def supervised_pretrain_tcn(params, X: np.ndarray, y: np.ndarray,
                             epochs: int = 30, batch_size: int = 128,
                             lr: float = 1e-3):
    """
    Supervised pre-training of the TCN policy network via MSE regression
    against boundary distance scores.

    Identical to supervised_pretrain but operates on window-shaped inputs
    and calls tcn_policy_forward instead of policy_forward:

        L_MSE = (1/N) Σᵢ ||tcn_policy_forward(params, sᵢ) - d(qᵢ)||²

    where:
        sᵢ ∈ ℝ^{W × 2L}  — window state tensor
        d(q) = |q - L/2| / (L/2)  — boundary distance targets

    Parameters:
    - params     = initial TCN parameter dict
    - X          = window state tensors (N, window, 2L)
    - y          = boundary distance score targets (N, L)
    - epochs     = number of full passes over the training set
    - batch_size = number of samples per gradient update
    - lr         = Adam learning rate

    Returns:
    - params = pre-trained TCN parameter dict
    """
    print(f"    Pre-training: {len(X)} samples, {epochs} epochs", flush=True)
    opt  = AdamOptimizer(params, lr=lr)
    X_j  = jnp.array(X)
    y_j  = jnp.array(y)

    def mse_loss(p, xb, yb):
        return jnp.mean((tcn_policy_forward(p, xb) - yb) ** 2)

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


def reinforce_update_tcn(params, states: np.ndarray,
                          log_probs: np.ndarray, advantage: float):
    """
    Compute REINFORCE policy gradient for a single episode using the TCN
    policy backbone.

    Identical surrogate loss to reinforce_update, but calls
    tcn_policy_forward with window-shaped states:

        L = -A · (1/T) Σ_t lp_t · Σ_q log π_θ^TCN(q | s_t)

    where:
      - A       = advantage = R - b  (reward minus EMA baseline)
      - lp_t    = stored Plackett-Luce log P at step t  (stop gradient)
      - log π_θ = log-softmax of TCN scores, differentiable w.r.t. θ
      - s_t ∈ ℝ^{W × 2L}  — window state tensor

    Parameters:
    - params    = current TCN parameter dict
    - states    = window state tensors from episode (T, window, 2L)
    - log_probs = Plackett-Luce log-probs of actions taken (T,)
    - advantage = R - b (scaled reward signal)

    Returns:
    - g = policy gradient (same pytree structure as params)
    """
    @jit
    def loss_fn(p):
        x      = jnp.array(states)                        # (T, window, 2L)
        scores = tcn_policy_forward(p, x)                 # (T, L)
        log_pi = jax.nn.log_softmax(scores, axis=-1)      # (T, L)
        lp     = jnp.array(log_probs)                     # (T,)
        return -advantage * jnp.mean(lp * jnp.sum(log_pi, axis=-1))

    return grad(loss_fn)(params)


def clip_grads(grads, max_norm: float = 1.0):
    """
    Clip a gradient pytree by global L2 norm.

    Computes the global norm across all leaves of the gradient pytree
    and rescales uniformly if it exceeds max_norm:

        global_norm = sqrt( Σ_i ||g_i||² )
        scale       = min(1, max_norm / global_norm)
        g_i         ← scale · g_i

    This keeps update magnitudes bounded without changing gradient
    direction, stabilising training when loss curvature is large.

    Parameters:
    - grads    = gradient pytree (same structure as policy params)
    - max_norm = clipping threshold (default 1.0)

    Returns:
    - clipped gradient pytree (same structure as grads)
    """
    leaves      = jax.tree.leaves(grads)
    global_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    scale       = jnp.minimum(1.0, max_norm / (global_norm + 1e-8))
    return jax.tree.map(lambda g: g * scale, grads)


def train_policy_tcn(L: int, depth: int, k_per_layer: int,
                     n_batches: int = 150, batch_size: int = 48,
                     lr: float = 3e-4, window: int = 4,
                     hidden: int = 64, kernel_size: int = 3,
                     baseline_alpha: float = 0.05,
                     entropy_coeff: float = 0.01,
                     grad_clip: float = None,
                     temperature: float = 1.0,
                     pretrain_episodes: int = 500,
                     pretrain_epochs: int = 30) -> Tuple:
    """
    Full two-phase policy training with TCN backbone: supervised pre-train
    → REINFORCE fine-tuning.

    [Phase 1] Supervised warm-start:
        Train TCN to predict boundary distance scores via MSE on window-
        shaped states.  Provides physically informed initialisation.

    [Phase 2] REINFORCE fine-tuning (identical protocol to train_policy):
        For each batch of episodes:
            1. Run batch_size episodes via run_episode_adaptive_tcn
            2. Update EMA baseline:
                   b ← (1-α)·b + α·⟨S(L/2)⟩_batch
            3. Per-episode advantage: A_i = S_i(L/2) - b
            4. Accumulate gradient via reinforce_update_tcn
            5. Update: θ = Adam(θ, g)

    Improvement tracked against boundary heuristic:
        ΔS = ⟨S(L/2)⟩_adaptive - ⟨S(L/2)⟩_boundary

    Parameters:
    - L                 = number of qubits
    - depth             = number of circuit layers (typically 4L)
    - k_per_layer       = measurement budget per layer
    - n_batches         = number of REINFORCE update steps
    - batch_size        = episodes per gradient update B
    - lr                = Adam learning rate for REINFORCE
    - window            = history window length W
    - hidden            = number of TCN channels throughout
    - kernel_size       = causal filter width k
    - baseline_alpha    = EMA decay α for variance reduction
    - entropy_coeff     = entropy regularisation coefficient β
    - grad_clip         = global norm gradient clipping threshold
                          (None = no clipping; 1.0 recommended for TCN)
    - temperature       = Plackett-Luce softmax temperature T:
                              p_q ∝ exp(s_q / T)
                          T > 1 broadens selection, reducing variance.
                          Passed through to run_episode_adaptive_tcn.
    - pretrain_episodes = episodes for supervised data generation
    - pretrain_epochs   = epochs for supervised pre-training

    Returns:
    - params  = trained TCN parameter dict
    - history = 'entropy', 'baseline', 'improvement' per batch (dict)
    """
    import jax.random as jrandom
    from src.models.policy import init_tcn_policy_params

    # Phase 1: supervised warm-start
    key    = jrandom.PRNGKey(SEED)
    params = init_tcn_policy_params(key, L, window,
                                    hidden=hidden, kernel_size=kernel_size)

    X_sup, y_sup = generate_supervised_data_tcn(
        L, depth, k_per_layer,
        n_episodes=pretrain_episodes, window=window)
    params = supervised_pretrain_tcn(
        params, X_sup, y_sup,
        epochs=pretrain_epochs, lr=1e-3)

    # Phase 2: REINFORCE fine-tuning
    opt      = AdamOptimizer(params, lr=lr)
    baseline = 0.0
    history  = {'entropy': [], 'baseline': [], 'improvement': []}

    # Reference: boundary heuristic entropy for comparison
    ref_ents = [run_episode_boundary(L, depth, k_per_layer)[0]
                for _ in range(20)]
    ref_mean = float(np.mean(ref_ents))

    for batch_idx in range(n_batches):
        batch_entropies, all_states, all_lp = [], [], []

        for _ in range(batch_size):
            ep = run_episode_adaptive_tcn(
                L, depth, k_per_layer, params,
                window=window, temperature=temperature)
            batch_entropies.append(ep['final_entropy'])
            all_states.append(ep['states'])
            all_lp.append(ep['log_probs'])

        mean_ent = float(np.mean(batch_entropies))
        baseline = (1 - baseline_alpha) * baseline + baseline_alpha * mean_ent

        # Accumulate gradients across the batch
        total_grad = jax.tree.map(jnp.zeros_like, params)
        for ent, states, lp in zip(batch_entropies, all_states, all_lp):
            advantage  = ent - baseline
            g          = reinforce_update_tcn(params, states, lp, advantage)
            total_grad = jax.tree.map(lambda tg, gi: tg + gi / batch_size,
                                       total_grad, g)

        # Optional gradient clipping by global norm
        if grad_clip is not None:
            total_grad = clip_grads(total_grad, grad_clip)

        params = opt.step(params, total_grad)

        history['entropy'].append(mean_ent)
        history['baseline'].append(baseline)
        history['improvement'].append(mean_ent - ref_mean)

        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx+1:3d} | S = {mean_ent:.3f} | "
                  f"baseline = {baseline:.3f} | "
                  f"vs boundary = {mean_ent - ref_mean:+.3f}")

    return params, history
