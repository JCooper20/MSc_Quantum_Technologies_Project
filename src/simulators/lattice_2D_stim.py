"""
2D L×L random-Clifford brickwork circuit using the Stim stabiliser-tableau
backend.

Key advantage over Qiskit Statevector: O(n²) time and space instead of
O(2ⁿ), allowing lattices of N = L² qubits within reasonable runtimes.
"""

# Imports
import numpy as np
import stim
from typing import Dict
from src.analysis.entropy import stabiliser_entropy
from src.simulators._stim_common import measure_layer


# ============================================================================
# 2D LATTICE GEOMETRY — L×L brickwork circuit
# ============================================================================

def run_trajectory_2d(L: int, depth: int, p_m: float) -> Dict:
    """
    Run one trajectory on an L×L square lattice of qubits with a
    brickwork gate pattern and random Z-measurements.

    Qubit labelling: site (row i, col j) → index i·L + j,
    so the full register has N = L² qubits.

    Gate schedule uses a 4-phase cycle to cover all bonds:

        t % 4 == 0 → horizontal even:  (i·L+j, i·L+j+1)  j=0,2,...
        t % 4 == 1 → horizontal odd:   (i·L+j, i·L+j+1)  j=1,3,...
        t % 4 == 2 → vertical even:    (i·L+j, (i+1)·L+j) i=0,2,...
        t % 4 == 3 → vertical odd:     (i·L+j, (i+1)·L+j) i=1,3,...

    The 4-phase cycle is necessary to tile all nearest-neighbour bonds —
    a 2-phase (Horizontal/Vervical only) schedule leaves sublattice bonds disconnected
    and produces zero entanglement across the bipartition cut.

        U_{a,b} ~ Uniform(C_2)    for each bond (a,b)

    After gates, each qubit is measured independently:

        m_q ~ Bernoulli(p_m)  where q ∈ {0,...,N-1}

    Entanglement entropy is computed on the horizontal bipartition
    cutting the lattice between rows L//2-1 and L//2:

        S = S( qubits in rows {0,...,L//2 - 1} )

    which gives a cut of size L·(L//2) = L²/2 qubits.
    """
    N   = L * L
    cut = L * (L // 2)    # number of qubits in lower half of lattice

    sim = stim.TableauSimulator()
    sim.set_num_qubits(N)

    entropy_hist = []
    meas_record  = []

    for layer in range(depth):
        phase = layer % 4
        if phase == 0:
            # Horizontal even bonds: j = 0, 2, 4, ...
            for row in range(L):
                base = row * L
                for col in range(0, L - 1, 2):
                    sim.do_tableau(stim.Tableau.random(2),
                                   [base + col, base + col + 1])
        elif phase == 1:
            # Horizontal odd bonds: j = 1, 3, 5, ...
            for row in range(L):
                base = row * L
                for col in range(1, L - 1, 2):
                    sim.do_tableau(stim.Tableau.random(2),
                                   [base + col, base + col + 1])
        elif phase == 2:
            # Vertical even bonds: i = 0, 2, 4, ...
            for col in range(L):
                for row in range(0, L - 1, 2):
                    sim.do_tableau(stim.Tableau.random(2),
                                   [row * L + col, (row + 1) * L + col])
        else:
            # Vertical odd bonds: i = 1, 3, 5, ...
            for col in range(L):
                for row in range(1, L - 1, 2):
                    sim.do_tableau(stim.Tableau.random(2),
                                   [row * L + col, (row + 1) * L + col])

        # Independent measurements at rate p_m
        outcomes = measure_layer(sim, N, p_m, meas_basis='Z')

        meas_record.append(outcomes)
        entropy_hist.append(stabiliser_entropy(sim, N, cut))

    return {
        'entropy_history':    entropy_hist,
        'measurement_record': np.array(meas_record),
        'final_entropy':      entropy_hist[-1] if entropy_hist else 0.0,
    }
