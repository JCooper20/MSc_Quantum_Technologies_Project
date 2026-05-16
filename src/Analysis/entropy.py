"""
analysis/entropy.py
-------------------
Canonical entropy utilities used across all Stim-based stages.

These functions were copy-pasted 6+ times in the original monolith.
All other modules should import from here.
"""

import numpy as np
import stim


# ============================================================================
# GF(2) RANK
# ============================================================================

def gf2_rank(matrix: np.ndarray) -> int:
    """
    Compute the rank of a binary matrix over GF(2) via Gaussian elimination.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_rows, n_cols), dtype bool or int

    Returns
    -------
    rank : int
    """
    M = matrix.astype(np.bool_).copy()
    n_rows, n_cols = M.shape
    rank = 0
    for col in range(n_cols):
        pivot = next((row for row in range(rank, n_rows) if M[row, col]), None)
        if pivot is None:
            continue
        if pivot != rank:
            M[[rank, pivot]] = M[[pivot, rank]]
        for row in range(n_rows):
            if row != rank and M[row, col]:
                M[row] ^= M[rank]
        rank += 1
    return rank


# ============================================================================
# STABILISER ENTROPY
# ============================================================================

def stabiliser_entropy(sim: stim.TableauSimulator, L: int, n_A: int) -> float:
    """
    Von Neumann entropy S(A) for subsystem A = {0, ..., n_A-1} of an
    L-qubit stabiliser state.

    Formula:  S(A) = rank(M_B) - n_B

    where M_B is the (L × 2·n_B) binary matrix of Pauli components of
    each stabiliser generator restricted to the complement B = {n_A,...,L-1},
    and rank is computed over GF(2).

    Derivation
    ----------
    The stabiliser group of ρ_A contains those generators acting as
    identity on B. The number of such independent generators is
    L - rank(M_B). Since ρ_A is a stabiliser state on n_A qubits with
    (n_A - S(A)) independent stabilisers:

        n_A - S(A) = L - rank(M_B)  →  S(A) = rank(M_B) - n_B

    Parameters
    ----------
    sim : stim.TableauSimulator
        Simulator whose current state is the stabiliser state of interest.
    L   : int
        Total number of qubits.
    n_A : int
        Size of subsystem A (the first n_A qubits).

    Returns
    -------
    S : float  (in bits, range [0, min(n_A, n_B)])
    """
    n_B = L - n_A
    if n_A == 0 or n_B == 0:
        return 0.0

    stabilisers = sim.canonical_stabilizers()
    M_B = np.zeros((L, 2 * n_B), dtype=np.bool_)

    for i, stab in enumerate(stabilisers):
        xs, zs = stab.to_numpy()
        M_B[i, :n_B] = xs[n_A:L]
        M_B[i, n_B:] = zs[n_A:L]

    rank = gf2_rank(M_B)
    S    = rank - n_B
    return float(max(0, min(S, min(n_A, n_B))))


def compute_half_chain_entropy(sim: stim.TableauSimulator, L: int) -> float:
    """Convenience wrapper: S(L/2) for the current state."""
    return stabiliser_entropy(sim, L, L // 2)
