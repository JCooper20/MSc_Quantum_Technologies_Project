"""
Measurement record encodings for the phase classifiers.

List of encoders used across project:
  1) encode_flat = MLP input (1D feature vector)
  2) encode_2d = CNN input (2-channel spatiotemporal image)
  3) encode_temporal = TCN / GRU input (time-step sequence)

generate_data and generate_data_all build labelled datasets from
Stim trajectories for the phase classifiers.
"""
# Imports
import numpy as np
from typing import Dict
from src.simulators.stim_clifford import run_trajectory_stim

# =====================================
# ENCODING FUNCTIONS (MLP,CNN,TCN/GRU)
# =====================================

def encode_flat(rec: np.ndarray) -> np.ndarray:
    """
    Encode measurement record as a flat feature vector for MLP input.

    Concatenates four feature groups:
        wm = was-measured mask (depth × L,) —> 1 if measured, 0 if not
        oc = outcomes (depth × L,) —> 0/1 if measured, 0 if not (uses wm to disambiguate)
        md = measurement density (depth,) —> fraction of qubits measured per layer
        of = mean outcome per layer (depth,) —> mean outcome where measured

    Output shape: (2·depth·L + 2·depth,)

    Parameters:
    - rec = raw measurement record (depth, L)
            [-1 if unmeasured, 0/1 if measurement outcome]

    Returns:
    - x = flattened feature vector ready for MLP input
          (2·depth·L + 2·depth,)
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
    Encode measurement record as a two-channel spatiotemporal image
    for CNN input.

    Treats the measurement record as an image with:
    - time (circuit layers) along one spatial axis (depth)
    - qubit index along the other spatial axis (L)
    - two channels encoding complementary information:
        
        Channel 0 (was-measured mask) = 1 if measured, 0 otherwise
        Channel 1 (outcomes) = 0/1 if measured, 0 otherwise
                *use Channel 0 to disambiguate*

    The CNN learns spatial patterns across both time and qubit index
    simultaneously — e.g. clustering of measurements near the boundary,
    or checkerboard patterns near criticality at p_m ≈ p_c.

    Parameters:
    - rec = raw measurement record (depth, L)
            [-1 if unmeasured, 0/1 if measurement outcome]
    Returns:
    - x = two-channel image ready for CNN input (2, depth, L)
    """
    wm = (rec >= 0).astype(np.float32)
    oc = np.clip(rec, 0, 1).astype(np.float32)
    return np.stack([wm, oc], axis=0)


def encode_temporal(rec: np.ndarray) -> np.ndarray:
    """
    Encode measurement record as a causal sequence for TCN / GRU input.

    At each layer t the input is a 2L-dimensional vector:

    x_t = [ (m_t⁰, m_t¹, ..., m_t^(L-1)) , 
            (o_t⁰, o_t¹, ..., o_t^(L-1)) ] ∈ {0,1}^(2L)
          i.e. ([was-measured] ​​⊕ [outcomes])
          
    Processed sequentially t = 0, 1, ..., depth-1, respecting the
    causal ordering of the circuit, at time t the model sees only
    measurements from layers t' ≤ t.

    This is the most physically faithful encoding since in a real
    experiment measurement outcomes arrive one layer at a time.

    Parameters:
    - rec = raw measurement record (depth, L)
            [-1 if unmeasured, 0/1 if measurement outcome]

    Returns:
    x = causal sequence ready for TCN / GRU input (depth, 2·L)
    """
    wm = (rec >= 0).astype(np.float32)
    oc = np.clip(rec, 0, 1).astype(np.float32)
    return np.concatenate([wm, oc], axis=1)   # (depth, 2L)

# ====================================================================
# Dataset Generation — labelled trajectories for phase classification
# ====================================================================

def generate_data(L: int, depth: int, p_low: float, p_high: float,
                  n_samples: int, meas_basis: str = 'Z') -> Dict:
    """
    Generate a balanced labelled dataset of measurement records for
    binary phase classification.

    Two classes, n_samples/2 trajectories each:
        - Label 0 (volume-law phase) = trajectories at p_m = p_low
        - Label 1 (area-law phase) = trajectories at p_m = p_high

    Each trajectory encoded in two formats:
        - flat = encode_flat(rec) (MLP) 
        - img = encode_2d(rec) (CNN) 

    Dataset shuffled and split 80/20 train/validation

    Parameters:
    - L = number of qubits
    - depth = number of circuit layers (typically 4L)
    - p_low = measurement rate for volume-law class (p_low < p_c)
    - p_high = measurement rate for area-law class   (p_high > p_c)
    - n_samples = total number of trajectories (n_samples/2 per class)
    - meas_basis = measurement basis ∈ {X,Y,Z}

    Returns:
     dict:
      - flat_train, flat_val = MLP input arrays
      - img_train,  img_val  = CNN input arrays  
      - y_train, y_val = binary labels ∈ {0, 1}
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
    Generate a balanced labelled dataset of measurement records in all
    three encodings for the CNN / MLP / TCN / GRU architecture comparison.

    Two classes, n_samples/2 trajectories each:
        - Label 0 (volume-law phase) = trajectories at p_m = p_low
        - Label 1 (area-law phase) = trajectories at p_m = p_high

    Each trajectory encoded in three formats simultaneously:
        - flat = encode_flat(rec) (MLP)
        - img = encode_2d(rec) (CNN) 
        - seq = encode_temporal(rec) (TCN/GRU)

    All four architectures train on identical trajectories — only the
    encoding differs — ensuring a fair comparison of inductive biases.

    Dataset shuffled and split 80/20 train/validation

    Parameters:
    - L = number of qubits
    - depth = number of circuit layers (typically 4L)
    - p_low = measurement rate for volume-law class (p_low < p_c)
    - p_high = measurement rate for area-law class (p_high > p_c)
    - n_samples = total number of trajectories (n_samples/2 per class)
    - meas_basis = measurement basis ∈ {X,Y,Z}

    Returns:
     dict:
     - flat_train, flat_val = MLP input arrays
     - img_train, img_val = CNN input arrays
     - seq_train, seq_val = TCN/GRU input arrays
     - y_train, y_val = binary labels ∈ {0, 1}
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
