"""
Monitored all-to-all Clifford circuit with a purification probe.
Stim stabiliser-tableau backend.

There is no spatial structure, it is a group of N qubits where any
qubit can interact with any other. The dynamics is built from single
operations, applied one at a time:
    - probability r = measure one random system qubit (Z basis)
    - probability (1 - r) = apply a random 2-qubit Clifford to one
                            random pair of distinct system qubits

One time steo is defined as N such operations (so each qubit
is 'touched' about once per step on average). The measurement rate r in
[0, 1] is ajustable parameter: 
    - small r = few measurements (entangling phase)
    - large r = many measurements (disentangling phase)

HOW PHASE IS MEASURED (the purification probe)
-------------------------------------------------
All-to-all circuits have no "left half vs right half", so the usual
spatial half-chain entropy is meaningless. Instead we use the
Gullans-Huse purification idea:

    1. Add a REFERENCE register R of N extra qubits and Bell-pair each
       reference qubit to one system qubit. This makes the system start
       MAXIMALLY MIXED
    2. Run the circuit on the SYSTEM qubits ONLY
    3. Track S_R, the entropy of the reference register, over time.

HOW S_R IS COMPUTED
-------------------
S_R is the entanglement entropy, so we use the sake shared routine
as every other circuit in this project: stabiliser_entropy_region from
src/analysis/entropy.py. The only difference from the 1D/2D circuits is
WHICH qubits we take the entropy of the reference register R.

*Key reference*
The all-to-all measurement-induced phase transition (MIPT) of
Nahum, Roy, Skinner & Ruhman, PRX Quantum 2, 010352 (2021).
"""

# Imports
import numpy as np
import stim
from typing import List
from src.analysis.entropy import stabiliser_entropy_region


# =====================================================================
# Initial state: system + reference Bell pairs
# =====================================================================

def build_initial_bell_state(N: int) -> stim.TableauSimulator:
    """
    Create the starting state: N Bell pairs linking each system qubit to
    its reference partner.

    Qubit layout in the tableau:
        system (S) = {0,    ..., N-1}
        reference (R) = {N,    ..., 2N-1}

    For each i we apply H to R_i then CNOT(R_i -> S_i), giving the Bell
    pair (|00> + |11>)/sqrt(2). Once all N pairs are made:
        - the system on its own is maximally mixed
        - the reference entropy is exactly S_R = N bits.

    This is the maximally-mixed starting point whose purification we
    track.
    """
    sim = stim.TableauSimulator()
    sim.set_num_qubits(2 * N)
    for i in range(N):
        sim.h(N + i)
        sim.cnot(N + i, i)
    return sim


# =====================================================================
# Sample-time schedule
# =====================================================================

def make_sample_times(t_max: int, n_log: int, n_lin: int) -> List[int]:
    """
    Build the list of time steps at which to record S_R.

    Combines logarithmically-spaced steps (to resolve the fast early-time
    behaviour) with linearly-spaced steps (to cover the long plateau),
    then de-duplicates and clips to [1, t_max].
    """
    log_t = np.geomspace(1, t_max, max(n_log, 2))
    lin_t = np.linspace(1, t_max,  max(n_lin, 2))
    times = np.unique(np.round(np.concatenate([log_t, lin_t])).astype(int))
    times = times[(times >= 1) & (times <= t_max)]
    return [int(t) for t in times]


# =====================================================================
# Single trajectory of the monitored all-to-all circuit
# =====================================================================

def run_single_trajectory(N: int, r: float, sample_times: List[int],
                          t_max: int, seed: int) -> np.ndarray:
    """
    Run one full trajectory and return S_R at each sample time.

    The reference qubits R = {N, ..., 2N-1} are prepared once and then
    left alone, only the system qubits S = {0, ..., N-1} get gates and
    measurements. At each requested time step we read off S_R as the
    entanglement entropy of the reference register.

    Parameters:
    N            = number of system qubits
    r            = measurement rate in [0, 1]
    sample_times = time steps at which to record S_R
    t_max        = total number of time steps to run
    seed         = RNG seed (makes this trajectory reproducible)

    Returns:
    out = array of S_R values (in bits), one per entry of sample_times.
    """
    rng = np.random.default_rng(seed)
    sim = build_initial_bell_state(N)

    reference = list(range(N, 2 * N))     # the qubits whose entropy = S_R
    sample_set = set(sample_times)
    idx_of = {t: i for i, t in enumerate(sample_times)}
    out = np.full(len(sample_times), np.nan, dtype=float)

    # Each time step is N elementary operations on the system.
    for step in range(1, t_max + 1):
        for _ in range(N):
            if rng.random() < r:
                # Born-rule Z-measurement on a random system qubit 
                q = int(rng.integers(0, N))
                sim.measure(q)
            else:
                # random 2-qubit Clifford on a random distinct pair 
                a = int(rng.integers(0, N))
                b = int(rng.integers(0, N - 1))
                if b >= a:           # ensures b != a (no qubit is paired with itself)
                    b += 1
                sim.do_tableau(stim.Tableau.random(2), [a, b])

        # Record S_R = entropy of the reference register at this step.
        if step in sample_set:
            out[idx_of[step]] = stabiliser_entropy_region(
                sim, 2 * N, reference)

    return out
