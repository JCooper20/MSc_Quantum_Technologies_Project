"""
Fully-connected random-Clifford circuit using the Stim stabiliser-tableau
backend.

This is the "layered" fully-connected construction: at each layer all L
qubits are simultaneously paired into a random perfect matching and a
random 2-qubit Clifford is applied to each of the L/2 pairs, with one
unit of time corresponding to one layer.

"""

# Imports
import numpy as np
import stim
from typing import Dict
from src.analysis.entropy import stabiliser_entropy
from src.simulators._stim_common import measure_layer


# ============================================================================
# FULLY CONNECTED GEOMETRY — random pair Clifford circuit
# ============================================================================

def run_trajectory_fc(L: int, depth: int, p_m: float) -> Dict:
    """
    Run one trajectory on L qubits with a fully-connected gate pattern
    and random Z-measurements.

    At each layer a uniformly random perfect matching of the L qubits
    is drawn and a random 2-qubit Clifford is applied to each pair:

        π_t ~ Uniform(perfect matchings of {0,...,L-1})
        U_{a,b} ~ Uniform(C_2) for each pair (a,b) ∈ q

    When L is odd the unpaired qubit receives no gate that layer.
    Each matching is constructed by shuffling all L qubit indices and
    pairing them sequentially: (q[0],q[1]), (q[2],q[3]), ...

    This geometry has no spatial locality, every qubit can interact
    with every other qubit, so entanglement spreads maximally fast.
    The resulting steady state is expected to be deep in the volume-law
    phase for p_m < p_c.

    After gates, each qubit is measured independently:

        m_q ~ Bernoulli(p_m)  q ∈ {0,...,L-1}

    Entanglement entropy is computed on the standard bipartition:

        S = S( qubits {0,...,L//2-1} )
    """
    sim = stim.TableauSimulator()
    sim.set_num_qubits(L)

    entropy_hist = []
    meas_record  = []

    for _ in range(depth):
        # Random perfect matching: shuffle indices, pair sequentially
        perm = np.random.permutation(L)
        for idx in range(0, L - 1, 2):
            a, b = int(perm[idx]), int(perm[idx + 1])
            sim.do_tableau(stim.Tableau.random(2), [a, b])

        # Independent measurements at rate p_m
        outcomes = measure_layer(sim, L, p_m, meas_basis='Z')

        meas_record.append(outcomes)
        entropy_hist.append(stabiliser_entropy(sim, L, L // 2))

    return {
        'entropy_history':    entropy_hist,
        'measurement_record': np.array(meas_record),
        'final_entropy':      entropy_hist[-1] if entropy_hist else 0.0,
    }
