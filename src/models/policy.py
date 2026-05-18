"""
Adaptive measurement policy network and episode runners.

The policy network π_θ maps the measurement history to qubit scores,
selecting which qubits to measure at each circuit layer to maximise
half-chain entanglement S(L/2).

Policy architecture:
    s_t ∈ ℝ^{W·2L+1} → FC(hidden) → FC(64) → FC(L) → scores ∈ ℝ^L

Action selection = Plackett-Luce top-k sampling:
    Given scores s ∈ ℝ^L, sample k qubits without replacement:
        - P(q₁) = exp(s_{q₁}) / Σ_q exp(s_q)
        - P(q₂|q₁) = exp(s_{q₂}) / Σ_{q≠q₁} exp(s_q)
        - ...

    During training: stochastic sampling for exploration
    During evaluation: deterministic top-k selection

Reward signal:
    R = S(L/2) — half-chain entanglement entropy at end of trajectory
    Goal: maximise R by selecting measurement locations adaptively

Three episode runners for comparison:
    run_episode_random = uniform random measurements at rate k/L
    run_episode_boundary = boundary-avoiding heuristic (baseline)
    run_episode_adaptive = learned policy π_θ (Plackett-Luce)
"""
# Imports
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrandom
import stim
from typing import Dict, List, Tuple
from src.analysis.entropy import stabiliser_entropy
from src.training.adam import AdamOptimizer


# =================================================
# Boundary Heuristic (used as pre-training oracle)
# =================================================

def boundary_scores(L: int) -> np.ndarray:
    """
    Score each qubit by its distance from the bipartition cut.

        d(q) = |q - L/2| / (L/2)  ∈ [0, 1]

    Qubits far from the cut (near the chain edges) score high — measuring
    them removes entanglement far from the bipartition, minimally reducing S(L/2). 
    
    Qubits near the cut score low — measuring them directly collapses correlations 
    across the bipartition, strongly reducing S(L/2).

    Used as the supervised pre-training oracle — the policy is initialised
    to mimic this heuristic before REINFORCE fine-tuning.
    """
    mid = L / 2.0
    return np.array([abs(q - mid) / mid for q in range(L)])


# ==================================================
# Policy Network - mapping state s_t → qubit scores
# ==================================================

def init_policy_params(key, input_dim: int, L: int, hidden: int = 128):
    """
    Initialise a three-layer MLP policy network.

    Architecture:
        s_t → FC(hidden) → FC(64) → FC(L) → scores ∈ ℝ^L
        
        Layer 1: (input_dim, hidden) — w1, b1
        Layer 2: (hidden, 64) — w2, b2
        Layer 3: (64, L) — w3, b3  ← one score per qubit

    Weights ~ N(0, 0.02²), biases = 0
    Small initialisation scale (0.02) keeps initial scores near uniform
    i.e. no strong qubit preference before training.

    Parameters:
    - key = JAX PRNG key
    - input_dim = state vector (dimension = W·2L + 1)
    - L = number of qubits (output dimension)
    - hidden = hidden layer width (default 128)
    
    Returns:
    - params = dict with keys w1, b1, w2, b2, w3, b3
    """
    keys = jrandom.split(key, 6)
    s    = 0.02
    return {
        'w1': jrandom.normal(keys[0], (input_dim, hidden)) * s,
        'b1': jnp.zeros(hidden),
        'w2': jrandom.normal(keys[1], (hidden, 64)) * s,
        'b2': jnp.zeros(64),
        'w3': jrandom.normal(keys[2], (64, L)) * s,
        'b3': jnp.zeros(L), }

def policy_forward(params, x):
    """
    Forward pass of the policy network.

        h₁ = ReLU(x · w1 + b1) ∈ ℝ^hidden
        h₂ = ReLU(h₁ · w2 + b2) ∈ ℝ^64
        s  = h₂ · w3 + b3 ∈ ℝ^L

    Output scores s are unnormalised — converted to selection
    probabilities via softmax during Plackett-Luce sampling:

        P(q) = exp(s_q) / Σ_{q'} exp(s_{q'})

    Parameters
    - params = {w1, b1, w2, b2, w3, b3} (dict)
    - x = state vectors (batch, input_dim)

    Returns:
    - scores = unnormalised qubit selection scores (batch, L)
    """
    h = jax.nn.relu(x @ params['w1'] + params['b1'])
    h = jax.nn.relu(h @ params['w2'] + params['b2'])
    return h @ params['w3'] + params['b3']


# ==============================================================
# Episode Runners — three measurement strategies for comparison
# ==============================================================

def run_episode_random(L: int, depth: int, p_m: float) -> Tuple[float, float]:
    """
    Baseline episode — uniform random measurements at rate p_m.

    Each qubit measured independently with probability p_m per layer:

        m_t^q ~ Bernoulli(p_m)  ∀ q,t

    No adaptive strategy — measurements are blind to the quantum state.
    Serves as the lower baseline for comparing adaptive controllers.

    Parameters:
    - L = number of qubits
    - depth = number of circuit layers
    - p_m = measurement probability per qubit per layer

    Returns:
    - (S, meas_rate) where: 
        - S = final S(L/2) after full trajectory
        - meas_rate = actual fraction of qubits measured = total / (L·depth)
    """
    sim = stim.TableauSimulator()
    sim.set_num_qubits(L)
    total = 0
    for t in range(depth):
        _apply_layer(sim, L, t)
        for q in range(L):
            if np.random.random() < p_m:
                sim.measure(q)
                total += 1
    return stabiliser_entropy(sim, L, L // 2), total / (L * depth)


def run_episode_boundary(L: int, depth: int,
                          k_per_layer: int) -> Tuple[float, float]:
    """
    Boundary-avoiding heuristic episode — measures k qubits furthest
    from the bipartition cut each layer.

    At each layer selects the k highest-scoring qubits under:

        d(q) = |q - L/2| / (L/2)  ∈ [0,1]

    with small noise for tie-breaking:

        q* = argtop-k( d(q) + ε ),   ε ~ N(0, 0.01²)

    Physical intuition: measuring qubits far from the cut removes
    entanglement away from the bipartition, preserving S(L/2) more
    than measuring qubits near the cut would.

    Serves as the primary baseline against which the learned policy
    is compared (ΔS = S_adaptive - S_boundary measures improvement)

    Parameters:
    - L = number of qubits
    - depth = number of circuit layers
    - k_per_layer = number of qubits measured per layer

    Returns:
    - (S, meas_rate) where:
        - S = final S(L/2) after full trajectory
        - meas_rate = measurement rate = k_per_layer / L
    """
    sim    = stim.TableauSimulator()
    sim.set_num_qubits(L)
    scores = boundary_scores(L)

    for t in range(depth):
        _apply_layer(sim, L, t)
        noisy = scores + np.random.normal(0, 0.01, L)
        if k_per_layer > 0:
            for q in np.argsort(noisy)[-k_per_layer:]:
                sim.measure(q)

    return stabiliser_entropy(sim, L, L // 2), k_per_layer / L


def run_episode_adaptive(L: int, depth: int, k_per_layer: int,
                          policy_fn, policy_params,
                          window: int = 4,
                          entropy_interval: int = 0) -> Dict:
    """
    Adaptive episode — policy π_θ selects k qubits per layer via
    Plackett-Luce sampling to maximise S(L/2).

    At each layer t:
        1. Build state vector s_t from measurement history window W
        2. Score qubits: s = π_θ(s_t) ∈ ℝ^L
        3. Plackett-Luce top-k sampling:
               - P(q₁) = exp(s_{q₁}) / Σ_q exp(s_q)
               - P(q₂|q₁) = exp(s_{q₂}) / Σ_{q≠q₁} exp(s_q)
               - ...
        4. Compute log-prob of selection:
               log P = Σⱼ log P(qⱼ | q₁,...,qⱼ₋₁)
        5. Execute measurements, record outcomes

    Log-probs stored for REINFORCE gradient computation.
    Entropy snapshots optionally recorded every entropy_interval layers.

    Parameters:
    - L = number of qubits
    - depth = number of circuit layers
    - k_per_layer = measurement budget k per layer
    - policy_fn = policy_fn(params, state) → (1, L) scores
    - policy_params = policy network parameter dict
    - window = history window length W
    - entropy_interval = compute S(L/2) every this many layers (0 = end only)

    Returns:
     - dict:
        - states = state vectors (depth, W·2L+1) 
        - chosen = binary measurement mask (depth, L) 
        - log_probs = Plackett-Luce log P per layer (depth,)   
        - final_entropy = S(L/2) at end of trajectory
        - entropy_snapshots = intermediate entropy values [(t, S(t))] 
        - meas_rate = k_per_layer / L
    """
    sim      = stim.TableauSimulator()
    sim.set_num_qubits(L)
    input_dim = window * 2 * L + 1

    was_measured = np.zeros((depth, L), dtype=np.float32)
    outcomes     = np.zeros((depth, L), dtype=np.float32)
    states_list, chosen_list, log_probs_list = [], [], []
    entropy_snapshots = []

    for t in range(depth):
        _apply_layer(sim, L, t)

        # Build state vector
        state = np.zeros(input_dim, dtype=np.float32)
        for w in range(window):
            li = t - window + w
            if li >= 0:
                base = w * 2 * L
                state[base:base + L]       = was_measured[li]
                state[base + L:base + 2*L] = outcomes[li]
        state[-1] = t / depth
        states_list.append(state)

        # Score qubits
        scores = np.array(policy_fn(policy_params, jnp.array(state[None, :]))[0])
        exp_s  = np.exp((scores - scores.max()))
        probs  = exp_s / exp_s.sum()

        # Plackett-Luce top-k sampling
        if 0 < k_per_layer < L:
            chosen = np.random.choice(L, size=k_per_layer,
                                       replace=False, p=probs)
        elif k_per_layer >= L:
            chosen = np.arange(L)
        else:
            chosen = np.array([], dtype=int)

        # Log-prob of selection
        lp = 0.0
        rem = probs.copy()
        for q in chosen:
            lp += np.log(rem[q] + 1e-10)
            rem[q] = 0.0
            s = rem.sum()
            if s > 0:
                rem /= s

        chosen_vec           = np.zeros(L, dtype=np.float32)
        chosen_vec[chosen]   = 1.0
        chosen_list.append(chosen_vec)
        log_probs_list.append(lp)

        # Execute measurements
        for q in chosen:
            result             = sim.measure(q)
            was_measured[t, q] = 1.0
            outcomes[t, q]     = float(result)

        if entropy_interval > 0 and (t + 1) % entropy_interval == 0:
            entropy_snapshots.append(
                (t, stabiliser_entropy(sim, L, L // 2)))

    return {
        'states':            np.array(states_list),
        'chosen':            np.array(chosen_list),
        'log_probs':         np.array(log_probs_list),
        'final_entropy':     stabiliser_entropy(sim, L, L // 2),
        'entropy_snapshots': entropy_snapshots,
        'meas_rate':         k_per_layer / L, }

# ==============================================
# Internal Helper — brickwork layer application
# ==============================================

def _apply_layer(sim: stim.TableauSimulator, L: int, t: int) -> None:
    """
    Apply one brickwork sublayer of random 2-qubit Cliffords in place.

        t even → even bonds: (0,1), (2,3), ..., (L-2, L-1)
        t odd  → odd bonds:  (1,2), (3,4), ..., (L-3, L-2)

    U_{i,i+1} ~ Uniform(C₂),   offset = t mod 2

    Modifies sim in place -> no return value.
    """
    offset = t % 2
    for i in range(offset, L - 1, 2):
        sim.do_tableau(stim.Tableau.random(2), [i, i + 1])
