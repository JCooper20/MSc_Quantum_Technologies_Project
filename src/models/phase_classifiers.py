"""
models/cnn_jax.py
-----------------
Spatial CNN phase classifier implemented in JAX (Stages 3–5).

Architecture:
  Conv(32) → Conv(64) → ResBlock(64) → ResBlock(64)
  → GlobalAvgPool → FC(32) → FC(2)

Input: (batch, 2, depth, L)  — two-channel spatiotemporal image.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random as jrandom, grad, jit
import numpy as np

from src.training.adam import AdamOptimizer

SEED = 42


# ============================================================================
# CONV HELPER
# ============================================================================

def conv2d(x, w, b):
    """2D convolution with (1,1) stride and same padding."""
    out = lax.conv_with_general_padding(
        x, w, window_strides=(1, 1),
        padding=((1, 1), (1, 1)),
        lhs_dilation=(1, 1), rhs_dilation=(1, 1))
    return out + b[None, :, None, None]


# ============================================================================
# PARAMETER INIT
# ============================================================================

def init_cnn_params(key, in_channels: int = 2):
    keys = jrandom.split(key, 6)
    s = 0.05
    return {
        'conv1_w': jrandom.normal(keys[0], (32, in_channels, 3, 3)) * s,
        'conv1_b': jnp.zeros(32),
        'conv2_w': jrandom.normal(keys[1], (64, 32, 3, 3)) * s,
        'conv2_b': jnp.zeros(64),
        'res1_w':  jrandom.normal(keys[2], (64, 64, 3, 3)) * s,
        'res1_b':  jnp.zeros(64),
        'res2_w':  jrandom.normal(keys[3], (64, 64, 3, 3)) * s,
        'res2_b':  jnp.zeros(64),
        'fc1_w':   jrandom.normal(keys[4], (64, 32)) * s,
        'fc1_b':   jnp.zeros(32),
        'fc2_w':   jrandom.normal(keys[5], (32, 2)) * s,
        'fc2_b':   jnp.zeros(2),
    }


# ============================================================================
# FORWARD PASS
# ============================================================================

def cnn_forward(params, x):
    """
    x: (batch, 2, depth, L)
    Returns logits: (batch, 2)
    """
    h    = jax.nn.relu(conv2d(x, params['conv1_w'], params['conv1_b']))
    h    = jax.nn.relu(conv2d(h, params['conv2_w'], params['conv2_b']))
    skip = h
    h    = jax.nn.relu(conv2d(h, params['res1_w'],  params['res1_b']))
    h    = jax.nn.relu(conv2d(h, params['res2_w'],  params['res2_b']) + skip)
    h    = jnp.mean(h, axis=(2, 3))      # global average pooling
    h    = jax.nn.relu(h @ params['fc1_w'] + params['fc1_b'])
    return h @ params['fc2_w'] + params['fc2_b']


def cross_entropy_loss(params, x, y):
    logits    = cnn_forward(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot   = jax.nn.one_hot(y, 2)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_cnn(img_train, y_train, img_val, y_val,
              epochs: int = 40, batch_size: int = 32, lr: float = 1e-3):
    """
    Train the CNN phase classifier.

    Returns (params, train_accuracy, val_accuracy, history_dict)
    """
    key    = jrandom.PRNGKey(SEED)
    params = init_cnn_params(key, in_channels=img_train.shape[1])
    opt    = AdamOptimizer(params, lr=lr)

    X_tr = jnp.array(img_train)
    y_tr = jnp.array(y_train, dtype=jnp.int32)
    X_va = jnp.array(img_val)
    y_va = jnp.array(y_val, dtype=jnp.int32)

    n_train  = len(X_tr)
    loss_fn  = jit(cross_entropy_loss)
    grad_fn  = jit(grad(cross_entropy_loss))
    history  = {'loss': [], 'val_acc': []}

    for epoch in range(epochs):
        perm        = np.random.permutation(n_train)
        epoch_loss  = 0.0
        n_batches   = 0
        for start in range(0, n_train, batch_size):
            idx    = perm[start:start + batch_size]
            g      = grad_fn(params, X_tr[idx], y_tr[idx])
            params = opt.step(params, g)
            epoch_loss += float(loss_fn(params, X_tr[idx], y_tr[idx]))
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        history['loss'].append(avg_loss)

        val_logits = cnn_forward(params, X_va)
        val_acc    = float(jnp.mean(jnp.argmax(val_logits, axis=-1) == y_va))
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0:
            tr_logits = cnn_forward(params, X_tr)
            tr_acc    = float(jnp.mean(jnp.argmax(tr_logits, axis=-1) == y_tr))
            print(f"      Epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
                  f"train={tr_acc:.3f} | val={val_acc:.3f}")

    tr_acc = float(jnp.mean(jnp.argmax(cnn_forward(params, X_tr), axis=-1) == y_tr))
    va_acc = float(jnp.mean(jnp.argmax(cnn_forward(params, X_va), axis=-1) == y_va))
    return params, tr_acc, va_acc, history


def cnn_predict_proba(params, img: np.ndarray) -> float:
    """Return P(area-law) for a single trajectory image."""
    logits = cnn_forward(params, jnp.array(img[None, ...]))
    return float(jax.nn.softmax(logits, axis=-1)[0, 1])
"""
models/tcn_jax.py
-----------------
Temporal Convolutional Network (TCN) phase classifier in JAX (Stage 5).

Architecture:
  Input projection → 3 residual blocks (dilations 1, 2, 4)
  → GlobalAvgPool over time → FC(32) → FC(2)

Each residual block uses causal 1D convolution, so the representation
at time t depends only on t′ ≤ t. This respects the causal ordering
of the measurement record and is the key distinction from the spatial CNN.

Input:  (batch, depth, 2·L)  — temporal sequence from encode_temporal
Output: (batch, 2)           — phase logits
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random as jrandom, grad, jit
import numpy as np

from src.training.adam import AdamOptimizer

SEED = 42


# ============================================================================
# CAUSAL 1D CONVOLUTION
# ============================================================================

def causal_conv1d(x, w, b, dilation: int = 1):
    """
    Causal 1D convolution along the time axis.

    x : (batch, in_channels, time)
    w : (out_channels, in_channels, kernel_size)
    b : (out_channels,)

    Left-padding ensures the output at position t depends only on
    inputs at positions t′ ≤ t (causality).
    """
    kernel_size = w.shape[2]
    pad_len     = (kernel_size - 1) * dilation
    x_padded    = jnp.pad(x, ((0, 0), (0, 0), (pad_len, 0)))
    out = lax.conv_general_dilated(
        x_padded, w,
        window_strides=(1,),
        padding='VALID',
        lhs_dilation=(1,),
        rhs_dilation=(dilation,),
        dimension_numbers=('NCH', 'OIH', 'NCH'))
    return out + b[None, :, None]


# ============================================================================
# PARAMETER INIT
# ============================================================================

def init_tcn_params(key, in_features: int, hidden: int = 64,
                    kernel_size: int = 3):
    """
    Initialise TCN parameters.

    Architecture: input projection → 3 residual blocks (dilations 1,2,4).
    """
    keys = jrandom.split(key, 20)
    s    = 0.05
    k    = kernel_size
    p    = {}

    # Input projection: in_features → hidden (kernel=1, no causal padding needed)
    p['proj_w'] = jrandom.normal(keys[0], (hidden, in_features, 1)) * s
    p['proj_b'] = jnp.zeros(hidden)

    for i, dil in enumerate([1, 2, 4]):
        ki = keys[1 + i * 4: 5 + i * 4]
        p[f'b{i}_conv1_w'] = jrandom.normal(ki[0], (hidden, hidden, k)) * s
        p[f'b{i}_conv1_b'] = jnp.zeros(hidden)
        p[f'b{i}_conv2_w'] = jrandom.normal(ki[1], (hidden, hidden, k)) * s
        p[f'b{i}_conv2_b'] = jnp.zeros(hidden)

    p['fc1_w'] = jrandom.normal(keys[15], (hidden, 32)) * s
    p['fc1_b'] = jnp.zeros(32)
    p['fc2_w'] = jrandom.normal(keys[16], (32, 2)) * s
    p['fc2_b'] = jnp.zeros(2)
    return p


# ============================================================================
# FORWARD PASS
# ============================================================================

def tcn_forward(params, x):
    """
    x: (batch, depth, 2·L)  — temporal sequence
    Returns logits: (batch, 2)
    """
    # Transpose to (batch, channels, time) for causal_conv1d
    h = jnp.transpose(x, (0, 2, 1))

    # Input projection
    h = jax.nn.relu(causal_conv1d(h, params['proj_w'], params['proj_b'],
                                   dilation=1))

    # Residual blocks at increasing dilations
    for i, dil in enumerate([1, 2, 4]):
        skip = h
        h    = jax.nn.relu(causal_conv1d(h, params[f'b{i}_conv1_w'],
                                          params[f'b{i}_conv1_b'], dilation=dil))
        h    = causal_conv1d(h, params[f'b{i}_conv2_w'],
                              params[f'b{i}_conv2_b'], dilation=dil)
        h    = jax.nn.relu(h + skip)

    # Global average pooling over time
    h = jnp.mean(h, axis=2)   # (batch, hidden)

    h = jax.nn.relu(h @ params['fc1_w'] + params['fc1_b'])
    return h @ params['fc2_w'] + params['fc2_b']


def tcn_loss(params, x, y):
    logits    = tcn_forward(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(y, 2) * log_probs, axis=-1))


# ============================================================================
# TRAINING
# ============================================================================

def train_tcn(seq_train, y_train, seq_val, y_val,
              hidden: int = 64, kernel_size: int = 3,
              epochs: int = 60, batch_size: int = 32, lr: float = 1e-3):
    """Train the TCN and return (params, train_acc, val_acc, history)."""
    in_features = seq_train.shape[2]   # 2*L
    key    = jrandom.PRNGKey(SEED)
    params = init_tcn_params(key, in_features, hidden, kernel_size)
    opt    = AdamOptimizer(params, lr=lr)

    X_tr   = jnp.array(seq_train)
    y_tr   = jnp.array(y_train, dtype=jnp.int32)
    X_va   = jnp.array(seq_val)
    y_va   = jnp.array(y_val, dtype=jnp.int32)

    grad_fn = jit(grad(tcn_loss))
    loss_fn = jit(tcn_loss)
    history = {'loss': [], 'val_acc': []}

    for epoch in range(epochs):
        perm       = np.random.permutation(len(X_tr))
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, len(X_tr), batch_size):
            idx    = perm[start:start + batch_size]
            g      = grad_fn(params, X_tr[idx], y_tr[idx])
            params = opt.step(params, g)
            epoch_loss += float(loss_fn(params, X_tr[idx], y_tr[idx]))
            n_batches  += 1

        history['loss'].append(epoch_loss / n_batches)
        val_acc = float(jnp.mean(jnp.argmax(tcn_forward(params, X_va), -1) == y_va))
        history['val_acc'].append(val_acc)
        if (epoch + 1) % 10 == 0:
            tr_acc = float(jnp.mean(jnp.argmax(tcn_forward(params, X_tr), -1) == y_tr))
            print(f"      Epoch {epoch+1:3d} | loss={history['loss'][-1]:.4f} | "
                  f"train={tr_acc:.3f} | val={val_acc:.3f}")

    tr_acc = float(jnp.mean(jnp.argmax(tcn_forward(params, X_tr), -1) == y_tr))
    va_acc = float(jnp.mean(jnp.argmax(tcn_forward(params, X_va), -1) == y_va))
    return params, tr_acc, va_acc, history
"""
models/gru_jax.py
-----------------
Gated Recurrent Unit (GRU) phase classifier in JAX (Stage 5).

Processes the measurement record layer-by-layer. At each time step t
the input is a 2L-vector [was_measured, outcome]. The GRU update:

    z_t = σ(W_z [h_{t-1}; x_t] + b_z)          (update gate)
    r_t = σ(W_r [h_{t-1}; x_t] + b_r)          (reset gate)
    h̃_t = tanh(W_h [r_t ⊙ h_{t-1}; x_t] + b_h)
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

The final hidden state h_T is fed to a two-class FC classifier.

Input:  (batch, depth, 2·L)
Output: (batch, 2)
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random as jrandom, grad, jit
import numpy as np

from src.training.adam import AdamOptimizer

SEED = 42


# ============================================================================
# PARAMETER INIT
# ============================================================================

def init_gru_params(key, in_features: int, hidden: int = 64):
    keys     = jrandom.split(key, 10)
    s        = 0.05
    total_in = hidden + in_features

    return {
        'W_z':   jrandom.normal(keys[0], (total_in, hidden)) * s,
        'b_z':   jnp.zeros(hidden),
        'W_r':   jrandom.normal(keys[1], (total_in, hidden)) * s,
        'b_r':   jnp.zeros(hidden),
        'W_h':   jrandom.normal(keys[2], (total_in, hidden)) * s,
        'b_h':   jnp.zeros(hidden),
        'fc1_w': jrandom.normal(keys[3], (hidden, 32)) * s,
        'fc1_b': jnp.zeros(32),
        'fc2_w': jrandom.normal(keys[4], (32, 2)) * s,
        'fc2_b': jnp.zeros(2),
    }


# ============================================================================
# GRU CELL
# ============================================================================

def gru_cell(params, h, x):
    """
    Single GRU step.

    h : (batch, hidden)
    x : (batch, in_features)
    Returns h_new : (batch, hidden)
    """
    hx      = jnp.concatenate([h, x], axis=-1)
    z       = jax.nn.sigmoid(hx @ params['W_z'] + params['b_z'])
    r       = jax.nn.sigmoid(hx @ params['W_r'] + params['b_r'])
    rhx     = jnp.concatenate([r * h, x], axis=-1)
    h_tilde = jnp.tanh(rhx @ params['W_h'] + params['b_h'])
    return (1 - z) * h + z * h_tilde


# ============================================================================
# FORWARD PASS
# ============================================================================

def gru_forward(params, x):
    """
    x: (batch, depth, 2·L)
    Returns logits: (batch, 2)
    """
    batch_size = x.shape[0]
    hidden_dim = params['W_z'].shape[1]
    h0         = jnp.zeros((batch_size, hidden_dim))

    # Scan over time: (time, batch, features)
    x_t = jnp.transpose(x, (1, 0, 2))

    def scan_fn(h, x_step):
        return gru_cell(params, h, x_step), None

    h_final, _ = lax.scan(scan_fn, h0, x_t)

    h = jax.nn.relu(h_final @ params['fc1_w'] + params['fc1_b'])
    return h @ params['fc2_w'] + params['fc2_b']


def gru_loss(params, x, y):
    logits    = gru_forward(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(y, 2) * log_probs, axis=-1))


# ============================================================================
# TRAINING
# ============================================================================

def train_gru(seq_train, y_train, seq_val, y_val,
              hidden: int = 64,
              epochs: int = 60, batch_size: int = 32, lr: float = 1e-3):
    """Train the GRU and return (params, train_acc, val_acc, history)."""
    in_features = seq_train.shape[2]   # 2*L
    key    = jrandom.PRNGKey(SEED)
    params = init_gru_params(key, in_features, hidden)
    opt    = AdamOptimizer(params, lr=lr)

    X_tr   = jnp.array(seq_train)
    y_tr   = jnp.array(y_train, dtype=jnp.int32)
    X_va   = jnp.array(seq_val)
    y_va   = jnp.array(y_val, dtype=jnp.int32)

    grad_fn = jit(grad(gru_loss))
    loss_fn = jit(gru_loss)
    history = {'loss': [], 'val_acc': []}

    for epoch in range(epochs):
        perm       = np.random.permutation(len(X_tr))
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, len(X_tr), batch_size):
            idx    = perm[start:start + batch_size]
            g      = grad_fn(params, X_tr[idx], y_tr[idx])
            params = opt.step(params, g)
            epoch_loss += float(loss_fn(params, X_tr[idx], y_tr[idx]))
            n_batches  += 1

        history['loss'].append(epoch_loss / n_batches)
        val_acc = float(jnp.mean(jnp.argmax(gru_forward(params, X_va), -1) == y_va))
        history['val_acc'].append(val_acc)
        if (epoch + 1) % 10 == 0:
            tr_acc = float(jnp.mean(jnp.argmax(gru_forward(params, X_tr), -1) == y_tr))
            print(f"      Epoch {epoch+1:3d} | loss={history['loss'][-1]:.4f} | "
                  f"train={tr_acc:.3f} | val={val_acc:.3f}")

    tr_acc = float(jnp.mean(jnp.argmax(gru_forward(params, X_tr), -1) == y_tr))
    va_acc = float(jnp.mean(jnp.argmax(gru_forward(params, X_va), -1) == y_va))
    return params, tr_acc, va_acc, history
