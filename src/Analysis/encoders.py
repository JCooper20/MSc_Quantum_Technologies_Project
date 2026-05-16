"""
analysis/encoders.py
--------------------
Measurement record encodings for the phase classifiers.

Four encodings are used across Stages 3–5:
  encode_flat     → MLP input (1D feature vector)
  encode_2d       → CNN input (2-channel spatiotemporal image)
  encode_temporal → TCN / GRU input (time-step sequence)

generate_data and generate_data_all build labelled datasets from
Stim trajectories for the phase classifiers.
"""

import numpy as np
from typing import Dict

from src.simulators.stim_clifford import run_trajectory_stim


# ============================================================================
# ENCODING FUNCTIONS
# ============================================================================

def encode_flat(rec: np.ndarray) -> np.ndarray:
    """
    Flatten the measurement record for MLP input.

    Features (concatenated):
      - was-measured mask : shape (depth × L,)
      - outcomes          : shape (depth × L,)
      - measurement density per layer : shape (depth,)
      - mean outcome per layer        : shape (depth,)

    Shape: (2·depth·L + 2·depth,)
    """
    depth, L = rec.shape
    wm = (rec >= 0).astype(np.float64)
    oc = np.clip(rec, 0, 1).astype(np.float64)
    md = wm.mean(axis=1)
    of = np.zeros(depth)
    for t in range(depth):
        mask = rec[t] >= 0
        if mask.any():
            of[t] = rec[t][mask].mean()
    return np.concatenate([wm.ravel(), oc.ravel(), md, of])


def encode_2d(rec: np.ndarray) -> np.ndarray:
    """
    Two-channel spatiotemporal image for CNN input.
    Channel 0: was-measured mask.
    Channel 1: outcomes (0 if not measured).

    Shape: (2, depth, L)
    """
    wm = (rec >= 0).astype(np.float32)
    oc = np.clip(rec, 0, 1).astype(np.float32)
    return np.stack([wm, oc], axis=0)


def encode_temporal(rec: np.ndarray) -> np.ndarray:
    """
    Causal sequence encoding for TCN / GRU input.

    At each time step t the input is a 2L-dimensional vector:
        [was_measured(q0..qL-1), outcome(q0..qL-1)]

    Shape: (depth, 2·L)
    """
    wm = (rec >= 0).astype(np.float32)
    oc = np.clip(rec, 0, 1).astype(np.float32)
    return np.concatenate([wm, oc], axis=1)   # (depth, 2L)


# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_data(L: int, depth: int, p_low: float, p_high: float,
                  n_samples: int, meas_basis: str = 'Z') -> Dict:
    """
    Generate labelled trajectories for CNN and MLP classifiers.

    n_samples // 2 trajectories at p_low (label 0, volume-law),
    n_samples // 2 trajectories at p_high (label 1, area-law).

    Returns dict with keys:
        flat_train, flat_val  → MLP input arrays
        img_train,  img_val   → CNN input arrays
        y_train,    y_val     → labels
    """
    flat_X, img_X, labels = [], [], []
    n_each = n_samples // 2

    for phase, p_m, label in [('Volume-law', p_low, 0),
                               ('Area-law',   p_high, 1)]:
        print(f"    {phase} (p_m={p_m}): ", end='', flush=True)
        for i in range(n_each):
            t = run_trajectory_stim(L, depth, p_m, meas_basis)
            flat_X.append(encode_flat(t['measurement_record']))
            img_X.append(encode_2d(t['measurement_record']))
            labels.append(label)
            if (i + 1) % 50 == 0:
                print(f"{i+1} ", end='', flush=True)
        print()

    flat_X  = np.array(flat_X)
    img_X   = np.array(img_X)
    labels  = np.array(labels)
    idx     = np.random.permutation(len(labels))
    flat_X, img_X, labels = flat_X[idx], img_X[idx], labels[idx]

    n_val = len(labels) // 5
    return {
        'flat_train': flat_X[n_val:], 'flat_val': flat_X[:n_val],
        'img_train':  img_X[n_val:],  'img_val':  img_X[:n_val],
        'y_train':    labels[n_val:], 'y_val':    labels[:n_val],
    }


def generate_data_all(L: int, depth: int, p_low: float, p_high: float,
                      n_samples: int, meas_basis: str = 'Z') -> Dict:
    """
    Generate labelled trajectories with all four encodings
    (flat, 2d, temporal) for CNN / MLP / TCN / GRU comparison.

    Returns dict with keys:
        flat_train/val, img_train/val, seq_train/val, y_train/val
    """
    flat_X, img_X, seq_X, labels = [], [], [], []
    n_each = n_samples // 2

    for phase, p_m, label in [('Volume-law', p_low, 0),
                               ('Area-law',   p_high, 1)]:
        print(f"    {phase} (p_m={p_m}): ", end='', flush=True)
        for i in range(n_each):
            t   = run_trajectory_stim(L, depth, p_m, meas_basis)
            rec = t['measurement_record']
            flat_X.append(encode_flat(rec))
            img_X.append(encode_2d(rec))
            seq_X.append(encode_temporal(rec))
            labels.append(label)
            if (i + 1) % 50 == 0:
                print(f"{i+1} ", end='', flush=True)
        print()

    flat_X  = np.array(flat_X)
    img_X   = np.array(img_X)
    seq_X   = np.array(seq_X)
    labels  = np.array(labels)
    idx     = np.random.permutation(len(labels))
    flat_X, img_X, seq_X, labels = (flat_X[idx], img_X[idx],
                                      seq_X[idx], labels[idx])
    n_val = len(labels) // 5
    return {
        'flat_train': flat_X[n_val:], 'flat_val': flat_X[:n_val],
        'img_train':  img_X[n_val:],  'img_val':  img_X[:n_val],
        'seq_train':  seq_X[n_val:],  'seq_val':  seq_X[:n_val],
        'y_train':    labels[n_val:], 'y_val':    labels[:n_val],
    }
