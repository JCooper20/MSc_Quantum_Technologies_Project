"""
Shared helpers for the Stim stabiliser-tableau circuit simulators.

Holds the per-layer projective-measurement routine common to every
geometry (1D brickwork, 2D lattice, fully connected), so the individual
geometry modules differ only in their gate schedule and bipartition cut.
"""

# Imports
import numpy as np
import stim


# =============================================================
# MEASUREMENT LAYER — independent per-qubit projective measure
# =============================================================

def measure_layer(sim: stim.TableauSimulator, n_qubits: int, p_m: float,
                  meas_basis: str = 'Z') -> np.ndarray:
    """
    Apply one layer of independent projective measurements to the
    stabiliser tableau.

    Each qubit is measured with probability p_m in the specified basis:

        m_q ~ Bernoulli(p_m)    ∀ q ∈ {0,...,n_qubits-1}

    X/Y measurements are realised by conjugating a Z measurement with
    the appropriate single-qubit Clifford rotation and undoing it after,
    so the post-measurement basis is unchanged for subsequent layers.

    Parameters:
    sim = Stim TableauSimulator holding the current stabiliser state
    n_qubits = number of qubits in the register
    p_m = measurement probability per qubit
    meas_basis = basis ∈ {X,Y,Z}

    Returns:
    outcomes = (n_qubits,) int8 array; entry is -1 if the qubit was not
               measured this layer, else the 0/1 measurement outcome.
    """
    outcomes = np.full(n_qubits, -1, dtype=np.int8)
    for q in range(n_qubits):
        if np.random.random() < p_m:
            if meas_basis == 'X':
                sim.h(q)
                result = sim.measure(q)
                sim.h(q)
            elif meas_basis == 'Y':
                sim.h(q); sim.s_dag(q)
                result = sim.measure(q)
                sim.s(q); sim.h(q)
            else:
                result = sim.measure(q)
            outcomes[q] = int(result)
    return outcomes
