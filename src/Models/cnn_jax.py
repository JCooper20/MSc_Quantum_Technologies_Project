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
