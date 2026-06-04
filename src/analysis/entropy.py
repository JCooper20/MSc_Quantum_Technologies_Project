"""
Canonical entropy utilities used across all Stim-based stages.
All other modules should import from here.
This script calculated the rank of a binary matrix over GF(2) (mod-2 arithmetic),
the core calculation behind stabiliser entropy S(A) = rank(M_B) - n_B.
"""

#Imports
import numpy as np
import stim

# ===============================================
#  Calculate GF(2) rank via Gaussian elimination
# ===============================================

def gf2_rank(matrix: np.ndarray) -> int:
    """
    Compute the rank of a binary matrix over GF(2) via Gaussian elimination.

    Rank = number of linearly independent rows under mod-2 arithmetic,
    where addition is XOR (a⊕b) and the only scalars a,b∈{0,1}.

    Algorithm for each column:
        1. Find pivot: first row ≥ current rank with a 1 in this column
        2. Swap pivot row into current rank position
        3. Eliminate: XOR pivot row into all other rows with a 1 in this column
        4. Increment rank

    Complexity: O(n_rows · n_cols · min(n_rows, n_cols))

    Parameters:
    - matrix = np.ndarray, shape (n_rows, n_cols)
    
    Returns:
    - rank = Number of linearly independent rows over GF(2)

    *Example*
    M = [[1, 0, 1],   =>  rank = 2 (row 2 = row 0 XOR row 1
         [0, 1, 1],                so only 2 independent rows)
         [1, 1, 0]]
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


# =============================
# Calculate Stabiliser Entropy
# =============================

def stabiliser_entropy(sim: stim.TableauSimulator, L: int, n_A: int) -> float:
    """
    Von Neumann entropy S(A) for subsystem A = {0, ..., n_A-1} of an
    L-qubit stabiliser state.

    Formula: S(A) = rank(M_B) - n_B

    where M_B is the (L × 2·n_B) binary matrix of Pauli components of
    each stabiliser generator restricted to the complement B = {n_A,...,L-1},
    and rank is computed over GF(2) in 'gf2_rank()'.

    *Derivation*
    The stabiliser group of ρ_A contains those generators acting as
    identity on B. The number of such independent generators is
    L - rank(M_B). Since ρ_A is a stabiliser state on n_A qubits with
    (n_A - S(A)) independent stabilisers:

        n_A - S(A) = L - rank(M_B)  →  S(A) = rank(M_B) - n_B

    Parameters:
    - sim = stim.TableauSimulator (stabiliser state of interest)
    - L = Total number of qubits
    - n_A = Size of subsystem A (the first n_A qubits)

    Returns:
    - S = (in bits, range [0, min(n_A, n_B)])
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

# ============================================================
# Tripartite Mutual Information — diagnostic for MIPT order
# ============================================================

def stabiliser_entropy_region(sim: stim.TableauSimulator, L: int,
                               region: list) -> float:
    """
    Von Neumann entropy S(A) for an arbitrary contiguous subsystem A
    specified by a sorted list of qubit indices.

    Extends stabiliser_entropy to non-half-chain cuts by permuting
    the stabiliser tableau so that region qubits appear first, then
    applying the standard rank formula:
        S(A) = rank(M_B) - |B|
    where M_B is the (L × 2|B|) Pauli matrix restricted to the
    complement B = {0,...,L-1} \\ A, and rank is over GF(2).
    """
    region_set = set(region)
    complement = [q for q in range(L) if q not in region_set]
    n_B = len(complement)
    n_A = len(region)

    if n_A == 0 or n_B == 0:
        return 0.0

    stabilisers = sim.canonical_stabilizers()
    M_B = np.zeros((L, 2 * n_B), dtype=np.bool_)

    for i, stab in enumerate(stabilisers):
        xs, zs = stab.to_numpy()
        for j, q in enumerate(complement):
            M_B[i, j]        = xs[q]
            M_B[i, n_B + j]  = zs[q]

    rank = gf2_rank(M_B)
    S    = rank - n_B
    return float(max(0, min(S, min(n_A, n_B))))


def tripartite_mutual_info(sim: stim.TableauSimulator, L: int) -> float:
    """
    Tripartite mutual information I₃(A:B:C) using a 4-region geometry
    where A, B, C are the first three equal quarters and D (the fourth
    quarter) is traced out.

    Partition into four equal quarters of length L/4:
        A = {0,       ..., L/4 - 1}
        B = {L/4,     ..., L/2 - 1}
        C = {L/2,     ..., 3L/4 - 1}
        D = {3L/4,    ..., L - 1}      ← traced out

    Definition:
        I₃(A:B:C) = S(A) + S(B) + S(C)
                  - S(AB) - S(BC) - S(AC)
                  + S(ABC)

    The 4-region geometry is necessary for a non-trivial I₃ — if A, B, C
    exhausted the full chain, the pure-state identity S(AB) = S(C) etc.
    would force I₃ ≡ 0 algebraically. With D traced out:
        S(ABC) = S(D)  ≠ 0  in general

    Physical interpretation:
        I₃ < 0  →  volume-law phase — multipartite entanglement is extensive
        I₃ ≈ 0  →  area-law phase  — entanglement is short-range
        I₃ crosses zero near the critical measurement rate p_c.

    This geometry follows Zabalo et al. (PRB 102, 064305, 2020) and
    Li, Chen, Fisher (PRB 100, 134306, 2019).

    Requires L divisible by 4 — raises ValueError otherwise.
    """
    if L % 4 != 0:
        raise ValueError(
            f"L must be divisible by 4 for the 4-region I₃ geometry; got L={L}")

    q = L // 4
    A  = list(range(0,     q))
    B  = list(range(q,     2 * q))
    C  = list(range(2 * q, 3 * q))
    # D = range(3q, L) — traced out; S(ABC) = S(D) via pure-state complementarity
    AB = A + B
    BC = B + C
    AC = A + C
    ABC = A + B + C

    S_A   = stabiliser_entropy_region(sim, L, A)
    S_B   = stabiliser_entropy_region(sim, L, B)
    S_C   = stabiliser_entropy_region(sim, L, C)
    S_AB  = stabiliser_entropy_region(sim, L, AB)
    S_BC  = stabiliser_entropy_region(sim, L, BC)
    S_AC  = stabiliser_entropy_region(sim, L, AC)
    S_ABC = stabiliser_entropy_region(sim, L, ABC)    # = S(D) for a pure state

    return S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
