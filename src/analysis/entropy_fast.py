"""
Fast EXACT GF(2)-rank entropy utilities and the all-to-all tripartite
mutual information I3.

Role in the v2 sweep: S_R (the published order parameter) is still
computed by the UNCHANGED stabiliser_entropy_region in entropy.py, so
the new S_R curves come from literally the same code as the published
checkpoint. The functions here exist only to make I3 affordable — I3
needs seven entropy evaluations per sample time, which on the old
python path would multiply the entropy slice of the runtime by ~7.
The bit-packed rank below was verified bit-identical to the reference
implementation (200 random matrices + step-by-step on evolving
tableaux) before first use, and the sweep's verification harness
re-checks it at import time of the run.

I3 definition (Zabalo et al., PRB 101, 060301(R) convention):
    I3(A:B:C) = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC
which is <= 0 for stabilizer states in the pure-state literature
setting. Regions: the all-to-all model has no spatial structure, so
A, B, C are an arbitrary equal partition — the first three quarters
of the SYSTEM qubits (A = [0, N/4), B = [N/4, N/2), C = [N/2, 3N/4)),
with the remaining system quarter plus the entire reference register
as environment. N must be divisible by 4 (true for all sweep sizes).

HONEST CAVEATS (verified empirically during preparation):
  - the literature computes I3 for pure initial states; this protocol
    starts the system maximally mixed via Bell pairs. I3 on system
    subregions is still well-defined (the global 2N-qubit state is
    pure), but it does NOT reproduce the pure-state phenomenology:
    it is ~0 in BOTH deep phases (near-maximal subsystem entropies
    cancel) and shows a strong negative dip near the transition —
    usable as a dip locator, not a zero-crossing locator.
  - rare positive samples (I3 = +1) occur (~1% deep in the entangling
    phase); verified identical through the published slow entropy
    path, so this is a genuine property of the mixed-start geometry,
    not a computational bug. Disorder means remain <= 0.

Efficiency: canonical_stabilizers() is extracted ONCE per sample time
and its X/Z bit matrices are reused across all seven entropies.
"""

# Imports
from typing import Sequence

import numpy as np


def gf2_rank_bitpacked(M_bool: np.ndarray) -> int:
    """GF(2) rank of a bool matrix, rows packed into uint64 words.

    Verified bit-identical to src/analysis/entropy.py:gf2_rank on
    random matrices and on real stabilizer check matrices.
    """
    n_rows, n_cols = M_bool.shape
    if n_rows == 0 or n_cols == 0:
        return 0
    words = (n_cols + 63) // 64
    bits = np.packbits(M_bool, axis=1, bitorder="little")
    pad = words * 8 - bits.shape[1]
    if pad:
        bits = np.pad(bits, ((0, 0), (0, pad)))
    packed = np.ascontiguousarray(bits).view(np.uint64).copy()
    rank = 0
    for col in range(n_cols):
        w, b = col // 64, np.uint64(col % 64)
        mask = np.uint64(1) << b
        rows_with_bit = np.nonzero(packed[rank:, w] & mask)[0]
        if len(rows_with_bit) == 0:
            continue
        pivot = rank + rows_with_bit[0]
        if pivot != rank:
            packed[[rank, pivot]] = packed[[pivot, rank]]
        targets = np.nonzero(packed[:, w] & mask)[0]
        targets = targets[targets != rank]
        if len(targets):
            packed[targets] ^= packed[rank]
        rank += 1
        if rank == n_rows:
            break
    return rank


def stab_xz_matrices(stabs, L: int):
    """X and Z bit matrices (L x L bool) of the canonical stabilizers."""
    xs = np.zeros((L, L), dtype=bool)
    zs = np.zeros((L, L), dtype=bool)
    for i, s in enumerate(stabs):
        x, z = s.to_numpy()
        xs[i] = x
        zs[i] = z
    return xs, zs


def entropy_of_region_from_xz(xs: np.ndarray, zs: np.ndarray, L: int,
                              region: np.ndarray) -> float:
    """S(region) = rank(M_B) - |B|, B = complement, from prebuilt X/Z.

    Same formula, clipping and semantics as
    entropy.py:stabiliser_entropy_region — only the matrix build is
    vectorised and the rank bit-packed.
    """
    in_region = np.zeros(L, dtype=bool)
    in_region[region] = True
    comp = np.nonzero(~in_region)[0]
    n_B = len(comp)
    n_A = L - n_B
    if n_A == 0 or n_B == 0:
        return 0.0
    M = np.concatenate([xs[:, comp], zs[:, comp]], axis=1)
    rank = gf2_rank_bitpacked(M)
    S = rank - n_B
    return float(max(0, min(S, min(n_A, n_B))))


def tripartite_I3(sim, N: int) -> float:
    """I3(A:B:C) for the all-to-all state (single tableau extraction).

    A, B, C = first three N/4-sized quarters of the system qubits;
    see module docstring for conventions and the mixed-start caveats.
    """
    if N % 4 != 0:
        raise ValueError(f"N must be divisible by 4, got {N}")
    L = 2 * N
    stabs = sim.canonical_stabilizers()
    xs, zs = stab_xz_matrices(stabs, L)
    q = N // 4
    A = np.arange(0, q)
    B = np.arange(q, 2 * q)
    C = np.arange(2 * q, 3 * q)

    def S(*regions: Sequence[np.ndarray]) -> float:
        return entropy_of_region_from_xz(
            xs, zs, L, np.concatenate(regions))

    return (S(A) + S(B) + S(C)
            - S(A, B) - S(A, C) - S(B, C)
            + S(A, B, C))
