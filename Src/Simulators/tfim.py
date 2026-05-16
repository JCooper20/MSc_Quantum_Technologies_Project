"""
simulators/tfim.py
------------------
Transverse-Field Ising Model (TFIM) quantum circuit simulator using Qiskit.

    H = -J Σ ZᵢZᵢ₊₁ - h Σ Xᵢ

Evolves a GHZ target state under Trotter steps, stochastic measurements,
and bit-flip errors. Designed to feed the TFIMNeuralController (Stage 1).
"""

import numpy as np
from enum import IntEnum
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity, Operator

from src.config import TFIMConfig


class Action(IntEnum):
    NOTHING   = 0
    APPLY_X   = 1
    APPLY_Z   = 2
    REMEASURE = 3


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

    # ------------------------------------------------------------------ build

    def _create_ghz_statevector(self) -> Statevector:
        qc = QuantumCircuit(self.n)
        qc.h(0)
        for i in range(self.n - 1):
            qc.cx(i, i + 1)
        return Statevector.from_instruction(qc)

    def _create_trotter_operator(self) -> Operator:
        qc = QuantumCircuit(self.n)
        dt, J, h = self.config.dt, self.config.J, self.config.h
        for i in range(self.n - 1):
            qc.rzz(-2 * J * dt, i, i + 1)
        for i in range(self.n):
            qc.rx(-2 * h * dt, i)
        return Operator.from_circuit(qc)

    def _create_single_qubit_op(self, gate: str, qubit: int) -> Operator:
        qc = QuantumCircuit(self.n)
        getattr(qc, gate.lower())(qubit)
        return Operator.from_circuit(qc)

    # ---------------------------------------------------------------- physics

    def compute_fidelity(self, state: Statevector) -> float:
        return state_fidelity(state, self.ghz_statevector)

    def measure_qubit(self, state: Statevector, qubit: int) -> Tuple[Statevector, int]:
        """Projective measurement with Born-rule collapse."""
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

    def get_local_observables(self, state: Statevector) -> Dict[str, np.ndarray]:
        """Compute ⟨Xᵢ⟩ and ⟨Zᵢ⟩ for each qubit."""
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

    # --------------------------------------------------------------- trajectory

    def generate_trajectory(self, apply_errors: bool = True,
                            controlled: bool = False,
                            controller=None) -> Dict:
        """Run a full TFIM trajectory, optionally with neural control."""
        state = self.ghz_statevector.copy()
        arr   = np.array(state.data)

        measurement_outcomes = []
        measurement_mask     = np.zeros((self.config.num_timesteps, self.n))
        fidelities           = [self.compute_fidelity(state)]
        errors_applied       = []
        corrections_applied  = []
        observables_history  = []

        for t in range(self.config.num_timesteps):
            # 1. Trotter step
            arr   = self.trotter_op.data @ arr
            state = Statevector(arr)

            # 2. Bit-flip errors
            if apply_errors:
                for q in range(self.n):
                    if np.random.random() < self.config.p_error:
                        arr   = self.X_ops[q].data @ arr
                        state = Statevector(arr)
                        errors_applied.append((t, q))

            # 3. Stochastic measurements
            outcomes_this_step = np.full(self.n, -1)
            for q in range(self.n):
                if np.random.random() < self.config.p_measure:
                    state, outcome = self.measure_qubit(state, q)
                    arr = np.array(state.data)
                    outcomes_this_step[q] = outcome
                    measurement_mask[t, q] = 1
                    measurement_outcomes.append((t, q, outcome))

            # 4. Observables
            obs = self.get_local_observables(state)
            observables_history.append(obs)

            # 5. Neural control
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
            'final_fidelity':      fidelities[-1],
        }

    def _prepare_features(self, t: int, measurement_outcomes: List,
                          measurement_mask: np.ndarray,
                          observables_history: List,
                          current_outcomes: np.ndarray) -> Dict:
        """Encode the measurement history into neural network input tensors."""
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
