"""
simulators/statevector.py
-------------------------
1D monitored brickwork circuit using Qiskit Statevector.
Handles Ising-type Clifford gates and projective measurements with
explicit Born-rule collapse. Limited to L ≲ 12 due to exponential cost.
(For larger systems use simulators.stim_clifford instead!)
"""
#Imports
import numpy as np
from typing import Dict, List, Optional, Tuple
from qiskit.quantum_info import Statevector, Operator, partial_trace
from qiskit.quantum_info import entropy
from src.config import BrickworkConfig

# Build Ising entangling gate
def build_ising_gate() -> Operator:
    """
    Build the two-qubit Ising entangling gate:

        U = exp(-i π/4 (X⊗X + Z⊗Z))

    XX and ZZ commute, so U = exp(-i π/4 XX) · exp(-i π/4 ZZ).
    """
    X  = np.array([[0, 1], [1, 0]], dtype=complex)
    Z  = np.array([[1, 0], [0, -1]], dtype=complex)
    I4 = np.eye(4, dtype=complex)

    theta  = np.pi / 4
    exp_XX = np.cos(theta) * I4 - 1j * np.sin(theta) * np.kron(X, X)
    exp_ZZ = np.cos(theta) * I4 - 1j * np.sin(theta) * np.kron(Z, Z)

    op = Operator(exp_XX @ exp_ZZ)
    assert op.is_unitary(), "Ising gate is not unitary!"
    return op


ISING_OP = build_ising_gate()


# ============================================================================
# ENTROPY HELPER
# ============================================================================

def compute_entropy(sv: Statevector, L: int, subsystem: List[int]) -> float:
    """
    Von Neumann entropy S(A) for a subsystem A of a Qiskit Statevector.
    Uses partial_trace and the Qiskit entropy function (base-2 bits).
    """
    trace_out = [q for q in range(L) if q not in subsystem]
    if len(trace_out) == 0 or len(trace_out) == L:
        return 0.0
    rho_A = partial_trace(sv, trace_out)
    return float(entropy(rho_A, base=2))


# ============================================================================
# SINGLE-QUBIT MEASUREMENT
# ============================================================================

def measure_qubit_statevector(sv: Statevector, qubit: int,
                               L: int, basis: str = 'X'
                               ) -> Tuple[int, Statevector]:
    """
    Projective single-qubit measurement with Born-rule collapse.

    Steps:
      1. Rotate to measurement basis
      2. Compute P(outcome=0)
      3. Sample outcome
      4. Project and renormalise
      5. Rotate back

    Returns (outcome, collapsed_statevector).
    """
    data = np.array(sv.data, dtype=complex)
    n    = int(np.log2(len(data)))

    H     = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    Y_rot = np.array([[1, 1j], [1, -1j]], dtype=complex) / np.sqrt(2)
    rot   = {'X': H, 'Y': Y_rot}.get(basis, None)

    def _apply_single(state_flat, rot_mat, qubit):
        s = state_flat.reshape([2] * n)
        s = np.moveaxis(s, qubit, 0)
        s = np.tensordot(rot_mat, s, axes=([1], [0]))
        return np.moveaxis(s, 0, qubit).reshape(len(data))

    if rot is not None:
        data = _apply_single(data, rot, qubit)

    state = data.reshape([2] * n)
    state = np.moveaxis(state, qubit, 0)
    prob_0 = float(np.clip(np.sum(np.abs(state[0]) ** 2), 0.0, 1.0))
    outcome = 0 if np.random.random() < prob_0 else 1

    if outcome == 0:
        state[1] = 0.0
        norm = np.sqrt(prob_0) if prob_0 > 1e-15 else 1.0
    else:
        state[0] = 0.0
        norm = np.sqrt(1.0 - prob_0) if (1.0 - prob_0) > 1e-15 else 1.0
    state /= norm

    state = np.moveaxis(state, 0, qubit)
    data  = state.reshape(len(data))

    if rot is not None:
        data = _apply_single(data, rot.conj().T, qubit)

    return outcome, Statevector(data)


# ============================================================================
# MONITORED BRICKWORK CIRCUIT
# ============================================================================

class MonitoredBrickworkCircuit:
    """
    1D brickwork circuit with Ising entangling gates and stochastic
    projective measurements via Qiskit Statevector.

    Each layer:
      1. Even bonds: U_Ising on (0,1), (2,3), ...
      2. Odd bonds:  U_Ising on (1,2), (3,4), ...
      3. Measurements: each qubit measured with prob p_m in meas_basis

    Use for L ≲ 12. For larger L use stim_clifford.run_trajectory_stim.
    """

    def __init__(self, L: int, p_u: float = 1.0,
                 depth: int = 32, meas_basis: str = 'X'):
        self.L         = L
        self.p_u       = p_u
        self.depth     = depth
        self.meas_basis = meas_basis

    def run_trajectory(self, p_m: float) -> Dict:
        """
        Simulate one quantum trajectory.

        Returns dict with:
          'entropy_history'    : S(L/2) after each layer
          'measurement_record' : (depth, L) int8 array (-1 = unmeasured)
          'final_entropy'      : S(L/2) of the final state
          'measurement_count'  : total measurements performed
        """
        L  = self.L
        sv = Statevector.from_label('0' * L)
        entropy_history    = []
        measurement_record = []
        total_measurements = 0

        for layer in range(self.depth):

            # Even bonds
            for i in range(0, L - 1, 2):
                if np.random.random() < self.p_u:
                    sv = sv.evolve(ISING_OP, qargs=[i, i + 1])

            # Odd bonds
            for i in range(1, L - 1, 2):
                if np.random.random() < self.p_u:
                    sv = sv.evolve(ISING_OP, qargs=[i, i + 1])

            # Stochastic measurements
            layer_outcomes = np.full(L, -1, dtype=np.int8)
            for q in range(L):
                if np.random.random() < p_m:
                    outcome, sv = measure_qubit_statevector(
                        sv, q, L, basis=self.meas_basis)
                    layer_outcomes[q] = outcome
                    total_measurements += 1

            measurement_record.append(layer_outcomes)
            entropy_history.append(
                compute_entropy(sv, L, list(range(L // 2))))

        return {
            'entropy_history':    entropy_history,
            'measurement_record': np.array(measurement_record),
            'final_entropy':      entropy_history[-1] if entropy_history else 0.0,
            'measurement_count':  total_measurements,
        }
