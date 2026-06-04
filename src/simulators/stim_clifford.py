"""
Random Clifford brickwork circuit using the Stim stabiliser-tableau
backend (Stages 4–6).

Key advantage over Qiskit Statevector: O(n²) time and space instead of
O(2ⁿ), allowing system sizes L up to 64+ within reasonable runtimes.

All stages that previously copy-pasted gf2_rank / stabiliser_entropy /
run_trajectory_stim should import from here instead.
"""
# Imports
import numpy as np
import stim
from typing import Dict, List
from src.analysis.entropy import gf2_rank, stabiliser_entropy


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
# Single Trajectory — Stim stabiliser tableau simulation
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


# =====================================================================
# MIPT Sweep — trajectory-averaged S(L/2) across measurement rates p_m
# =====================================================================

def run_mipt_sweep(L: int, depth: int, pm_values: List[float],
                   n_traj: int, meas_basis: str = 'Z') -> Dict:
    """
    Sweep over measurement rates p_m and compute trajectory-averaged
    half-chain entanglement entropy S(L/2).
    For each p_m runs n_traj independent trajectories and computes:

    ⟨S(L/2)⟩ = (1/N) Σ_i S_i(L/2)

    Averaging over trajectories is necessary because each trajectory
    is a stochastic quantum process — individual runs fluctuate
    significantly near the critical point p_c.

    Parameters:
    - L = number of qubits
    - depth = number of circuit layers (typically 4L)
    - pm_values = list of measurement rates p_m ∈ [0, 1] to sweep
    - n_traj = number of trajectories per p_m
    - meas_basis = measurement basis ∈ {X,Y,Z}

    Returns:
     dict:
     - p_m = list of measurement rates swept
     - mean_S = ⟨S(L/2)⟩ at each p_m
     - std_S = standard deviation across trajectories
     - sem_S = standard error = std_S / sqrt(n_traj)
     - mean_history = mean S(L/2) vs layer averaged across trajectories
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
    Run MIPT sweeps across multiple system sizes to perform
    finite-size scaling analysis.

    For each L runs a full p_m sweep via 'run_mipt_sweep()', enabling:
        - Crossing analysis of S(L/2)/L vs p_m to extract p_c
        - Finite-size scaling collapse via ansatz:
              S(L/2) = f((p_m - p_c) · L^{1/ν})
        - Estimation of critical exponent ν

    Circuit depth scales as 4L for each system size, ensuring
    the circuit is deep enough to reach the steady state before
    measurements are applied.

    Parameters:
    - L_values = system sizes to sweep e.g. [8, 16, 32, 64]
    - pm_values = measurement rates p_m ∈ [0, 1]
    - n_traj = number of trajectories per (L, p_m) point
    - meas_basis = measurement basis ∈ {X,Y,Z}

    Returns:
     dict : {L: mipt_dict} where each mipt_dict contains
           p_m, mean_S, std_S, sem_S, mean_history
           as returned by run_mipt_sweep
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

"""
Random Clifford brickwork circuit using the Stim stabiliser-tableau
backend (Stages 4–6).

Key advantage over Qiskit Statevector: O(n²) time and space instead of
O(2ⁿ), allowing system sizes L up to 64+ within reasonable runtimes.

All stages that previously copy-pasted gf2_rank / stabiliser_entropy /
run_trajectory_stim should import from here instead.
"""
# Imports
import numpy as np
import stim
from typing import Dict, List
from src.analysis.entropy import gf2_rank, stabiliser_entropy


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
    a 2-phase (H/V only) schedule leaves sublattice bonds disconnected
    and produces zero entanglement across the bipartition cut.

        U_{a,b} ~ Uniform(C_2)    for each bond (a,b)

    After gates, each qubit is measured independently:

        m_q ~ Bernoulli(p_m)    ∀ q ∈ {0,...,N-1}

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
        outcomes = np.full(N, -1, dtype=np.int8)
        for q in range(N):
            if np.random.random() < p_m:
                outcomes[q] = int(sim.measure(q))

        meas_record.append(outcomes)
        entropy_hist.append(stabiliser_entropy(sim, N, cut))

    return {
        'entropy_history':    entropy_hist,
        'measurement_record': np.array(meas_record),
        'final_entropy':      entropy_hist[-1] if entropy_hist else 0.0,
    }


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
        U_{a,b} ~ Uniform(C_2)    for each pair (a,b) ∈ π_t

    When L is odd the unpaired qubit receives no gate that layer.
    Each matching is constructed by shuffling all L qubit indices and
    pairing them sequentially: (π[0],π[1]), (π[2],π[3]), ...

    This geometry has no spatial locality — every qubit can interact
    with every other qubit, so entanglement spreads maximally fast.
    The resulting steady state is expected to be deep in the volume-law
    phase for all p_m < p_c, with p_c shifted to higher values compared
    to the 1D brickwork circuit.

    After gates, each qubit is measured independently:

        m_q ~ Bernoulli(p_m)    ∀ q ∈ {0,...,L-1}

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
        outcomes = np.full(L, -1, dtype=np.int8)
        for q in range(L):
            if np.random.random() < p_m:
                outcomes[q] = int(sim.measure(q))

        meas_record.append(outcomes)
        entropy_hist.append(stabiliser_entropy(sim, L, L // 2))

    return {
        'entropy_history':    entropy_hist,
        'measurement_record': np.array(meas_record),
        'final_entropy':      entropy_hist[-1] if entropy_hist else 0.0,
    }
