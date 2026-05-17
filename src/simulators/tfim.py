"""
Transverse-Field Ising Model (TFIM) quantum circuit simulator using Qiskit.

    H = -J Σ ZᵢZᵢ₊₁ - h Σ Xᵢ

Evolves a GHZ target state under Trotter steps, stochastic measurements,
and bit-flip errors. Designed to feed the TFIMNeuralController.
"""
#Imports
import numpy as np
from enum import IntEnum
from typing import Dict, List, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity, Operator
from src.config import TFIMConfig

# Actions Class
class Action(IntEnum):
    NOTHING   = 0
    APPLY_X   = 1
    APPLY_Z   = 2
    REMEASURE = 3

# Simulator Class
class QiskitTFIMSimulator:
    """
    Simulate the TFIM under Trotter evolution, stochastic measurements,
    and bit-flip errors using Qiskit Statevector.
    """

    def __init__(self, config: TFIMConfig):
        self.config = config
        self.n = config.num_qubits

        self.ghz_statevector = self._create_ghz_statevector()
        self.trotter_op      = self._create_trotter_operator()

        self.X_ops = [self._create_single_qubit_op('X', q) for q in range(self.n)]
        self.Z_ops = [self._create_single_qubit_op('Z', q) for q in range(self.n)]
        self.H_ops = [self._create_single_qubit_op('H', q) for q in range(self.n)]

    # Create GHZ State (1/√2[|000...0> +|111...1>])
    def _create_ghz_statevector(self) -> Statevector:
        qc = QuantumCircuit(self.n)
        qc.h(0)
        for i in range(self.n - 1):
            qc.cx(i, i + 1)
        return Statevector.from_instruction(qc)

    # Trotterization of unitary evolution (e^{−iHdt} -> e^{iJ∑ZZ⋅dt} x e^{ih∑X⋅dt})
    def _create_trotter_operator(self) -> Operator:
        """
        Trotterize e^{-iHdt} ≈ ∏ e^{iJZ_iZ_{i+1}dt} · ∏ e^{ihX_idt}
        - RZZ layers (Ising coupling) then RX layers (transverse field).
        - First-order Trotter error = O(dt²) (valid for small dt)
        """
        qc = QuantumCircuit(self.n)
        dt, J, h = self.config.dt, self.config.J, self.config.h
        for i in range(self.n - 1):
            qc.rzz(-2 * J * dt, i, i + 1)
        for i in range(self.n):
            qc.rx(-2 * h * dt, i)
        return Operator.from_circuit(qc) 
        
    # Embed single-qubit gate (X, Z, H) on target qubit into full 2^n × 2^n operator matirx: 
    # e.g. I⊗I⊗G⊗I⊗I = 2^5 x 2^5 = 32 x 32
    def _create_single_qubit_op(self, gate: str, qubit: int) -> Operator:
        qc = QuantumCircuit(self.n)
        getattr(qc, gate.lower())(qubit)
        return Operator.from_circuit(qc)
    
    # -------------------------------------------------------
    # Physics — observables, measurement, Born-rule collapse
    # -------------------------------------------------------

    # Computes fidelity of current state |ψ⟩ vs target GHZ state |ψ_GHZ⟩ 
    def compute_fidelity(self, state: Statevector) -> float:
         """
        Compute fidelity between current state and target GHZ state.
        - F = |⟨ψ_GHZ|ψ⟩|² ∈ [0, 1]
        - F = 1.0 → state is exactly GHZ (perfect preservation)
        - F = 0.0 → state is orthogonal to GHZ (completely destroyed)
        Used as the primary metric for controller performance and
        as a loss signal during training via L = (1 - F).
        """
        return state_fidelity(state, self.ghz_statevector)

    # Preform  Projective measurement on single qubit
    def measure_qubit(self, state: Statevector, qubit: int) -> Tuple[Statevector, int]:
         """
        Projective measurement of a single qubit with Born-rule collapse.
        Steps:
          1. Compute P(outcome=0) and P(outcome=1) from state amplitudes
          2. Sample outcome stochastically: P(0) = |⟨0|ψ⟩|²
          3. Project: Keep only the amplitudes where the measured qubit matches 
          the observed outcome, set everything else to zero (a_i -> a_i if bit q of i == outcome, else 0)
          4. Renormalise collapsed state to unit norm (|ψ'> = |ψ_projected> / sqrt(sum_i |a_i|^2)
    
        Parameters:
        - state = current n-qubit statevector
        - qubit = index of qubit to measure (0 to n-1)
    
        Returns:
        - (collapsed_state, outcome) = post-measurement state and result (0 or 1)
        """
        probs = state.probabilities([qubit])
        outcome = 0 if np.random.random() < probs[0] else 1

        arr = np.array(state.data)
        new = np.zeros(2 ** self.n, dtype=complex)
        for i in range(2 ** self.n):
            if ((i >> qubit) & 1) == outcome:
                new[i] = arr[i]

        norm = np.linalg.norm(new)
        if norm > 1e-10:
            new /= norm
        return Statevector(new), outcome

    # Compute local Pauli expectation values <X_i> and <Z_i> for each qubit i
    def get_local_observables(self, state: Statevector) -> Dict[str, np.ndarray]:
         """
        Compute local Pauli expectation values <X_i> and <Z_i> for each qubit:
         - <Z_i> = P(0) - P(1) (direct from Z-basis probabilities):
         - <X_i> = <psi'|Z_i|psi'>,  |psi'> = H_i|psi>
           <X_i> = P'(0) - P'(1) (rotate to X basis via Hadamard then measure in Z)
      
        For a perfect GHZ state: <Z_i> = <X_i> = 0 for all i, 
        since entanglement is non-local.
        
        Parameters
        - state = current n-qubit statevector ∣ψ⟩
       
        Returns:
        dict with keys 'X' and 'Z' e.g [<Z_0>, <Z_1>, <Z_2>] and [<X_0>, <X_1>, <X_2>]
        containing per-qubit expectation values in [-1, 1]
        """
        arr = np.array(state.data)
        X_exp = np.zeros(self.n)
        Z_exp = np.zeros(self.n)
        for i in range(self.n):
            probs = state.probabilities([i])
            Z_exp[i] = probs[0] - probs[1]
            rotated = Statevector(self.H_ops[i].data @ arr)
            px = rotated.probabilities([i])
            X_exp[i] = px[0] - px[1]
        return {'X': X_exp, 'Z': Z_exp}

    # -------------------------------------------------------
    # Trajectory — full simulation loop with optional control
    # -------------------------------------------------------

    def generate_trajectory(self, apply_errors: bool = True,controlled: bool = False,
                            controller=None) -> Dict:
        """
        Run a full TFIM trajectory under Trotter evolution, stochastic
        measurements, and optional neural control.

        Each timestep t applies five operations in order:
        1. Unitary evolution: ∣ψ(t+dt)⟩=U_trotter​∣ψ(t)⟩
        2. Bit-flip errors: ∣ψ⟩→ X_q​∣ψ⟩  with prob. p_error per qubit
        3. Measurements: |psi⟩ -> Πm|psi⟩ / ||Πm|psi⟩||  with prob. p_measure
        4. Observables: record ⟨X_i⟩, ⟨Z_i⟩ ∀ i
        5. Neural corrections:  C_q ​∈ {I, X, Z, remeasure} per qubit 
        (active only after t >= history_length)

        Fidelity, F(t) = |⟨ψ_GHZ|ψ(t)⟩|^2 is recorded after every step.

        Parameters:
        - apply_errors : bool
        If True apply stochastic bit-flip errors at rate p_error
        - controlled   : bool
        If True query controller for corrections each timestep
        - controller   : TFIMNeuralController or None
        Neural controller — must implement get_corrections(features)

        Returns
        dict:
        fidelities = F(t) at every timestep, shape (T+1,)
        measurement_outcomes = list of (t, qubit, outcome) tuples
        measurement_mask = binary array shape (T, n), 1 = measured
        errors_applied = list of (t, qubit) error locations
        corrections_applied = list of (t, qubit, action) corrections
        observables_history = [{'X': array(n,), 'Z': array(n,)}] x T
        final_fidelity = F(T) scalar summary of trajectory
        """
        state = self.ghz_statevector.copy()
        arr   = np.array(state.data)

        measurement_outcomes = []
        measurement_mask     = np.zeros((self.config.num_timesteps, self.n))
        fidelities           = [self.compute_fidelity(state)]
        errors_applied       = []
        corrections_applied  = []
        observables_history  = []

        for t in range(self.config.num_timesteps):
            # Trotter step
            arr   = self.trotter_op.data @ arr
            state = Statevector(arr)

            # Bit-flip errors
            if apply_errors:
                for q in range(self.n):
                    if np.random.random() < self.config.p_error:
                        arr   = self.X_ops[q].data @ arr
                        state = Statevector(arr)
                        errors_applied.append((t, q))

            # Stochastic measurements
            outcomes_this_step = np.full(self.n, -1)
            for q in range(self.n):
                if np.random.random() < self.config.p_measure:
                    state, outcome = self.measure_qubit(state, q)
                    arr = np.array(state.data)
                    outcomes_this_step[q] = outcome
                    measurement_mask[t, q] = 1
                    measurement_outcomes.append((t, q, outcome))

            # Observables
            obs = self.get_local_observables(state)
            observables_history.append(obs)

            # Neural control (optional)
            if controlled and controller is not None and \
                    t >= self.config.history_length - 1:
                features = self._prepare_features(
                    t, measurement_outcomes, measurement_mask,
                    observables_history, outcomes_this_step)
                corrections = controller.get_corrections(features)

                for q, action in enumerate(corrections):
                    if action == Action.APPLY_X:
                        arr = self.X_ops[q].data @ arr
                        corrections_applied.append((t, q, action))
                    elif action == Action.APPLY_Z:
                        arr = self.Z_ops[q].data @ arr
                        corrections_applied.append((t, q, action))
                    elif action == Action.REMEASURE:
                        state = Statevector(arr)
                        state, _ = self.measure_qubit(state, q)
                        arr = np.array(state.data)
                        corrections_applied.append((t, q, action))

                state = Statevector(arr)

            fidelities.append(self.compute_fidelity(Statevector(arr)))

        return {
            'fidelities':          np.array(fidelities),
            'measurement_outcomes': measurement_outcomes,
            'measurement_mask':    measurement_mask,
            'errors_applied':      errors_applied,
            'corrections_applied': corrections_applied,
            'observables_history': observables_history,
            'final_fidelity':      fidelities[-1], }

        # Prepare raw trajectoy history into fixed sized tensors for nearal network
        def _prepare_features(self, t: int, measurement_outcomes: List,measurement_mask: np.ndarray,
                              observables_history: List,current_outcomes: np.ndarray) -> Dict:
         """
        Encode the last T timesteps of trajectory history into fixed-size
        input tensors for the neural controller.
        Builds a sliding window [t-T+1, ..., t] of length T = history_length,
        zero-padded at the start when t < T.

        Arrays constructed:
        outcomes  : (T, n)  measurement outcomes — 0/1 where measured, 0 elsewhere
        mask      : (T, n)  binary — 1 where measurement occurred, 0 otherwise
                          (distinguishes 'measured=0' from 'not measured')
        X_obs     : (T, n)  <X_i> at each timestep in window
        Z_obs     : (T, n)  <Z_i> at each timestep in window
        params    : (3,)    [p_measure, J, h] (physical parameters)

    The final row of outcomes is overwritten with current_outcomes
    to ensure the most recent measurements are always included.

    Parameters:
    t = current timestep
    measurement_outcomes = list of (t, qubit, outcome) tuples — full history
    measurement_mask = binary array shape (T_total, n)
    observables_history = list of {'X': array(n,), 'Z': array(n,)} dicts
    current_outcomes = array(n,) — this timestep's outcomes, -1 if unmeasured

    Returns
    -------
    dict with keys 'outcomes', 'mask', 'X_obs', 'Z_obs', 'params'
    all as float32 arrays ready for neural network input
    """
        T, n = self.config.history_length, self.n
        outcomes_array = np.zeros((T, n))
        mask_array     = np.zeros((T, n))
        X_obs          = np.zeros((T, n))
        Z_obs          = np.zeros((T, n))

        start_t = max(0, t - T + 1)
        for tau in range(start_t, t + 1):
            idx = tau - start_t
            if idx < T:
                mask_array[idx] = measurement_mask[tau]
                if tau < len(observables_history):
                    X_obs[idx] = observables_history[tau]['X']
                    Z_obs[idx] = observables_history[tau]['Z']

        for (time, qubit, outcome) in measurement_outcomes:
            if start_t <= time <= t:
                idx = time - start_t
                if idx < T:
                    outcomes_array[idx, qubit] = outcome

        outcomes_array[-1] = np.where(current_outcomes >= 0, current_outcomes, 0)
        params = np.array([self.config.p_measure, self.config.J, self.config.h])

        return {
            'outcomes': outcomes_array.astype(np.float32),
            'mask':     mask_array.astype(np.float32),
            'X_obs':    X_obs.astype(np.float32),
            'Z_obs':    Z_obs.astype(np.float32),
            'params':   params.astype(np.float32),
        }
