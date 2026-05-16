"""
models/policy.py
----------------
Adaptive measurement policy network and episode runners for the
REINFORCE controller (Stage 6).

The policy reads a sliding window of past measurement outcomes and
scores each qubit. The top-k scoring qubits are measured each layer
(Plackett-Luce sampling during training for exploration).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrandom
import stim
from typing import Dict, List, Tuple

from src.analysis.entropy import stabiliser_entropy
from src.training.adam import AdamOptimizer


# ============================================================================
# BOUNDARY HEURISTIC (used as pre-training oracle)
# ============================================================================

def boundary_scores(L: int) -> np.ndarray:
    """
    Score each qubit by its distance from the bipartition cut.
    Qubits near the edges (far from centre) get high scores.
    Used as the supervised pre-training oracle.
    """
    mid = L / 2.0
    return np.array([abs(q - mid) / mid for q in range(L)])


# ============================================================================
# POLICY NETWORK
# ============================================================================

def init_policy_params(key, input_dim: int, L: int, hidden: int = 128):
    """
    Initialise a three-layer MLP that maps state → qubit scores.

    input_dim = window * 2 * L + 1   (measurement history + time fraction)
    Output: (batch, L) — one score per qubit.
    """
    keys = jrandom.split(key, 6)
    s    = 0.02
    return {
        'w1': jrandom.normal(keys[0], (input_dim, hidden)) * s,
        'b1': jnp.zeros(hidden),
        'w2': jrandom.normal(keys[1], (hidden, 64)) * s,
        'b2': jnp.zeros(64),
        'w3': jrandom.normal(keys[2], (64, L)) * s,
        'b3': jnp.zeros(L),
    }


def policy_forward(params, x):
    """
    x: (batch, input_dim)
    Returns qubit scores: (batch, L)
    """
    h = jax.nn.relu(x @ params['w1'] + params['b1'])
    h = jax.nn.relu(h @ params['w2'] + params['b2'])
    return h @ params['w3'] + params['b3']


# ============================================================================
# EPISODE RUNNERS
# ============================================================================

def run_episode_random(L: int, depth: int, p_m: float) -> Tuple[float, float]:
    """Baseline: uniform random measurements at rate p_m."""
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
    Boundary-avoiding heuristic: measure the k qubits furthest from
    the bipartition cut each layer.
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
    Adaptive controller with Plackett-Luce top-k measurement selection.

    policy_fn(params, state) → qubit scores (batch=1, L)

    Parameters
    ----------
    entropy_interval : int
        Compute entropy every this many layers (0 = end only).

    Returns dict with keys:
        states, chosen, log_probs, final_entropy,
        entropy_snapshots, meas_rate
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
        'meas_rate':         k_per_layer / L,
    }


# ============================================================================
# INTERNAL HELPER
# ============================================================================

def _apply_layer(sim: stim.TableauSimulator, L: int, t: int) -> None:
    """Apply one brickwork layer (even or odd bonds) of random Cliffords."""
    offset = t % 2
    for i in range(offset, L - 1, 2):
        sim.do_tableau(stim.Tableau.random(2), [i, i + 1])
