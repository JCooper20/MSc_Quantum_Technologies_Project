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
