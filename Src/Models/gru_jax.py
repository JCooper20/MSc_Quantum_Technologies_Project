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
