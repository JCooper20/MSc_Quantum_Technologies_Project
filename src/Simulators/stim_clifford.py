"""
simulators/stim_clifford.py
---------------------------
Random Clifford brickwork circuit using the Stim stabiliser-tableau
backend (Stages 4–6).

Key advantage over Qiskit Statevector: O(n²) time and space instead of
O(2ⁿ), allowing system sizes L up to 64+ within reasonable runtimes.

All stages that previously copy-pasted gf2_rank / stabiliser_entropy /
run_trajectory_stim should import from here instead.
"""

import numpy as np
import stim
from typing import Dict, List

from src.analysis.entropy import gf2_rank, stabiliser_entropy


# ============================================================================
# BRICKWORK LAYER HELPER
# ============================================================================

def apply_brickwork_layer(sim: stim.TableauSimulator, L: int, t: int,
                          use_ising: bool = False) -> None:
    """
    Apply one brickwork sublayer of random 2-qubit Clifford gates.

    Even layers (t even): bonds (0,1), (2,3), ...
    Odd layers  (t odd):  bonds (1,2), (3,4), ...

    Parameters
    ----------
    use_ising : bool
        If True, apply the fixed Ising-type gate instead of a random
        Clifford (used for the Ising-type ensemble in Stage 3).
        Note: the Ising gate is not in the Clifford group — this flag
        is reserved for future use with approximate Clifford proxies.
    """
    offset = t % 2
    for i in range(offset, L - 1, 2):
        cliff = stim.Tableau.random(2)
        sim.do_tableau(cliff, [i, i + 1])


# ============================================================================
# SINGLE TRAJECTORY
# ============================================================================

def run_trajectory_stim(L: int, depth: int, p_m: float,
                        meas_basis: str = 'Z') -> Dict:
    """
    Run one quantum trajectory with Stim's stabiliser-tableau simulator.

    At each bond of each layer a uniformly random 2-qubit Clifford is
    drawn and applied. Measurements are performed in the specified basis
    with probability p_m per qubit per layer.

    Parameters
    ----------
    L          : number of qubits
    depth      : number of circuit layers
    p_m        : measurement probability per qubit per layer
    meas_basis : 'Z', 'X', or 'Y'

    Returns
    -------
    dict:
        'entropy_history'    : S(L/2) after each layer
        'measurement_record' : (depth, L) int8 array (-1 = unmeasured)
        'final_entropy'      : S(L/2) at end of circuit
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
        outcomes = np.full(L, -1, dtype=np.int8)
        for q in range(L):
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

        meas_record.append(outcomes)
        entropy_hist.append(stabiliser_entropy(sim, L, L // 2))

    return {
        'entropy_history':    entropy_hist,
        'measurement_record': np.array(meas_record),
        'final_entropy':      entropy_hist[-1] if entropy_hist else 0.0,
    }


# ============================================================================
# MIPT SWEEP
# ============================================================================

def run_mipt_sweep(L: int, depth: int, pm_values: List[float],
                   n_traj: int, meas_basis: str = 'Z') -> Dict:
    """
    Sweep over measurement rates and compute trajectory-averaged
    half-chain entropy S(L/2).

    Returns dict with keys: 'p_m', 'mean_S', 'std_S', 'sem_S',
                             'mean_history'
    """
    results = {'p_m': [], 'mean_S': [], 'std_S': [], 'sem_S': [],
               'mean_history': []}

    for p_m in pm_values:
        ents, hists = [], []
        for _ in range(n_traj):
            t = run_trajectory_stim(L, depth, p_m, meas_basis)
            ents.append(t['final_entropy'])
            hists.append(t['entropy_history'])

        m, s = np.mean(ents), np.std(ents)
        sem  = s / np.sqrt(n_traj)
        results['p_m'].append(p_m)
        results['mean_S'].append(m)
        results['std_S'].append(s)
        results['sem_S'].append(sem)
        results['mean_history'].append(np.mean(hists, axis=0).tolist())
        print(f"    p_m={p_m:.2f}: S = {m:.3f} ± {sem:.3f}")

    return results


# ============================================================================
# MULTI-SIZE SCALING
# ============================================================================

def run_scaling(L_values: List[int], pm_values: List[float],
                n_traj: int, meas_basis: str = 'Z') -> Dict:
    """
    Run MIPT sweeps for multiple system sizes.

    Returns {L: mipt_dict} where each mipt_dict is the output of
    run_mipt_sweep.
    """
    import time
    scaling = {}
    for L in L_values:
        depth = 4 * L
        print(f"\n  L={L} (depth={depth})")
        t0 = time.time()
        scaling[L] = run_mipt_sweep(L, depth, pm_values, n_traj, meas_basis)
        print(f"    Done in {time.time() - t0:.1f}s")
    return scaling
