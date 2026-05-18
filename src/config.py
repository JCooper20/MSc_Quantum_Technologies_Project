"""
All configuration dataclasses for every stage of the project.
"""

# Imports
from dataclasses import dataclass, field
from typing import List
import numpy as np


# ========================
# TFIM NEURAL CONTROLLER
# ========================

@dataclass
class TFIMConfig:
    """
    Configuration for the TFIM neural controller 
    """
    # System parameters
    num_qubits: int = 6
    J: float = 1.0           # Ising coupling
    h: float = 0.2           # Transverse field (weaker → slower dynamics)
    dt: float = 0.1          # Trotter step

    # Measurement
    p_measure: float = 0.3   # Higher → more information for controller

    # Simulation
    num_timesteps: int = 10
    history_length: int = 8

    # Error model
    p_error: float = 0.01    # Lower → easier correction task

    # Training
    num_train_trajectories: int = 4000
    num_val_trajectories: int = 800
    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Loss weights
    lambda_fidelity: float = 1.0
    lambda_sparse: float = 0.001

    # Network architecture (Conv + LSTM)
    conv_channels: int = 64
    conv_kernel: int = 3
    hidden_size: int = 128
    num_lstm_layers: int = 2
    dropout: float = 0.1

    # Evaluation
    num_eval_trajectories: int = 200


# ===================
# TORIC CODE DECODER
# ===================

@dataclass
class ToricConfig:
    # Code geometry
    L: int = 6
    p_error: float = 0.08

    # Architecture
    base_channels: int = 48
    num_res_blocks: int = 3
    dropout: float = 0.1

    # Training
    num_train_samples: int = 50_000
    num_val_samples: int = 5_000
    batch_size: int = 512
    num_epochs: int = 30
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4

    # Evaluation
    eval_samples_training: int = 300
    eval_samples_final: int = 2000

    @property
    def n_qubits(self) -> int:
        return 2 * self.L * self.L

    @property
    def n_plaquettes(self) -> int:
        return self.L * self.L


# ==========================================
# MONITORED BRICKWORK CIRCUIT (STATEVECTOR)
# ==========================================

@dataclass
class BrickworkConfig:
    """
    Configuration for the Ising-type / random Clifford monitored circuit
    simulated via Qiskit statevector (L ≲ 12)
    """
    # System
    L: int = 8
    p_u: float = 1.0 # Gate application probability (1.0 = all gates)
    depth: int = 32
    meas_basis: str = 'X' # 'X', 'Y', or 'Z'

    # MIPT sweep
    p_m_values: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20,
                                  0.25, 0.30, 0.40, 0.50, 0.60])
    n_trajectories: int = 30

    # Multi-L scaling
    L_values: List[int] = field(default_factory=lambda: [6, 8, 10, 12])

    # Phase classifier training
    p_low: float = 0.05   # Training measurement rate for volume-law class
    p_high: float = 0.30  # Training measurement rate for area-law class
    n_train_samples: int = 400
    clf_epochs: int = 30


# ========================
# STIM STABILISER BACKEND
# ========================

@dataclass
class StimConfig:
    """
    Configuration for the random Clifford circuit with Stim stabiliser
    backend (L>12)
    """
    # MIPT sweep
    L_main: int = 16
    meas_basis: str = 'Z'

    # Scaling study
    L_values: List[int] = field(default_factory=lambda: [8, 12, 16, 24, 32])

    # Measurement rate grids
    pm_fine: List[float] = field(default_factory=lambda: [
        0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14,
        0.16, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00
    ])
    scaling_pm: List[float] = field(default_factory=lambda: [
        0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15,
        0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00
    ])

    # Trajectory counts
    n_traj_mipt: int = 80
    n_traj_scaling: int = 50
    n_traj_data: int = 150  # per phase per difficulty

    # Classifier
    clf_epochs: int = 40
    clf_batch: int = 32
    clf_lr: float = 1e-3
    sweep_n: int = 40  # trajectories per p_m in learnability sweep


# ====================================================
# TEMPORAL ARCHITECTURES (TCN / GRU / CNN comparison)
# ====================================================

@dataclass
class ArchConfig:
    """
    Configuration for the CNN / TCN / GRU architecture comparison
    """

    # Stim simulation (same defaults as StimConfig)
    L_main: int = 16
    meas_basis: str = 'Z'
    L_values: List[int] = field(default_factory=lambda: [8, 16, 32, 64])

    # Architecture hyperparameters (shared across JAX models)
    hidden: int = 64
    kernel_size: int = 3       # TCN kernel
    gru_hidden: int = 64

    # Training
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3

    # Delta sweep (learnability gap as a function of |p_m - p_c|)
    p_c_estimate: float = 0.16
    delta_values: List[float] = field(
        default_factory=lambda: [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20])
    n_traj_delta: int = 200


# =======================
# ADAPTIVE RL CONTROLLER
# =======================

@dataclass
class RLConfig:
    """
    Configuration for the REINFORCE adaptive feedback controller
    """
    # System sizes to train and evaluate
    L_values: List[int] = field(default_factory=lambda: [8, 12, 16, 24])
    depth_fn: str = '4L' # circuit depth as a multiple of L

    # Measurement budget
    k_values: List[float] = field(
    default_factory=lambda: [0.1, 0.2, 0.3])  # k_per_layer / L fractions

    # Policy network
    window: int = 4  # number of past layers in state
    hidden: int = 128

    # Supervised pre-training
    n_supervised_eps: int = 500
    pretrain_epochs: int = 30
    pretrain_lr: float = 1e-3

    # REINFORCE fine-tuning
    n_batches: int = 150
    batch_size: int = 48
    rl_lr: float = 3e-4
    baseline_alpha: float = 0.05   # EMA baseline decay
    entropy_coeff: float = 0.01    # Entropy regularisation

    # Evaluation
    n_eval: int = 200
    n_eval_high: int = 500 # High-stat evaluation at critical p_m
    high_stats_pm: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])

    def depth(self, L: int) -> int:
        return 4 * L
