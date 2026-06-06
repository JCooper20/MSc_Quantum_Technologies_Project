"""
1D random-Clifford brickwork circuit using the Stim stabiliser-tableau
backend.

Key advantage over Qiskit Statevector: O(n²) time and space instead of
O(2ⁿ), allowing system sizes L up to 64+ within reasonable runtimes.
"""

# Imports
import numpy as np
import stim
from typing import Dict
from src.analysis.entropy import stabiliser_entropy
from src.simulators._stim_common import measure_layer


# =======================
# BRICKWORK LAYER HELPER
# =======================

def apply_brickwork_layer(sim: stim.TableauSimulator, L: int, t: int,
                          use_ising: bool = False) -> None:
    """
    Apply one sublayer of random 2-qubit Clifford gates in a brickwork
    pattern to the Stim stabiliser tableau.

    Alternates between even and odd bond sets based on layer index t:
    t even = even bonds: (0,1), (2,3), (4,5), ...
    t odd = odd bonds: (1,2), (3,4), (5,6), ...

    This brick-wall tiling ensures every neighbouring pair is entangled
    on alternating layers, generating long-range correlations throughout
    the chain.

    Each bond receives an independently sampled uniformly random
    2-qubit Clifford gate from the 720-element Clifford group C_2:

    U_{i,i+1} ~ Uniform(C_2)

    Parameters:
    sim = Stim TableauSimulator holding current stabiliser state
    L = number of qubits
    t = layer index — determines even/odd bond offset via t % 2 (t mod 2)
    use_ising : bool (default False)
                Reserved for future use — Ising gate is not in C_2
                so cannot be applied via Stim tableau directly
    """
    offset = t % 2
    for i in range(offset, L - 1, 2):
        cliff = stim.Tableau.random(2)
        sim.do_tableau(cliff, [i, i + 1])


# =======================================================
# 1D BRICKWORK — Single Trajectory
# =======================================================

def run_trajectory_stim(L: int, depth: int, p_m: float,
                        meas_basis: str = 'Z') -> Dict:
    """
    Run one quantum trajectory with Stim's stabiliser-tableau simulator.

    At each bond of each layer a uniformly random 2-qubit Clifford is
    drawn and applied. Measurements are performed in the specified basis
    with probability p_m per qubit per layer.

    Parameters:
    L = number of qubits
    depth = number of circuit layers
    p_m = measurement probability per qubit per layer
    meas_basis = basis ∈ {X,Y,Z}

    Returns:
      dict:
        - 'entropy_history' = S(L/2) after each layer
        - 'measurement_record' = (depth, L) array where (-1 = unmeasured)
        - 'final_entropy' = S(L/2) at end of circuit
    """
    sim = stim.TableauSimulator()
    sim.set_num_qubits(L)

    entropy_hist = []
    meas_record  = []

    for layer in range(depth):
        # Even bonds
        for i in range(0, L - 1, 2):
            sim.do_tableau(stim.Tableau.random(2), [i, i + 1])
        # Odd bonds
        for i in range(1, L - 1, 2):
            sim.do_tableau(stim.Tableau.random(2), [i, i + 1])

        # Measurements
        outcomes = measure_layer(sim, L, p_m, meas_basis)

        meas_record.append(outcomes)
        entropy_hist.append(stabiliser_entropy(sim, L, L // 2))

    return {
        'entropy_history':    entropy_hist,
        'measurement_record': np.array(meas_record),
        'final_entropy':      entropy_hist[-1] if entropy_hist else 0.0,
    }
