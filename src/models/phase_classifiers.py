"""
======================================================================================
(1)
Spatial CNN phase classifier for measurement-induced phase detection 

Architecture:
- Conv(32) → Conv(64) → ResBlock(64) → ResBlock(64) → GlobalAvgPool → FC(32) → FC(2)

Input:
- (batch, 2, depth, L) where 2 ∈ Ch0: was-measured mask and Ch1: measurement outcomes
Output: 
- (batch, 2) logits for [volume-law, area-law]

Processes the full measurement record simultaneously as a spatial image
(no notion of temporal ordering)
======================================================================================
"""
# Imports
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random as jrandom, grad, jit
import numpy as np
from src.training.adam import AdamOptimizer

SEED = 42

# =====================
# 2D convolution helper 
# =====================

def conv2d(x, w, b):
    """
    2D convolution with unit stride and same padding.

    Applies learned filters w across the spatiotemporal input x,
    preserving spatial dimensions (depth, L) via same padding:

    out[b,c,i,j] = Σ_{k,di,dj} w[c,k,di,dj] · x[b,k,i+di,j+dj] + b[c]

    Parameters:
    x = input feature map (batch, in_channels, depth, L)
    w = convolutional filters (out_channels, in_channels, kH, kW)
    b = bias per output channel (out_channels,)

    Returns:
    out = same spatial dimensions as input (batch, out_channels, depth, L) 
    """
    out = lax.conv_with_general_padding(
        x, w, window_strides=(1, 1),
        padding=((1, 1), (1, 1)),
        lhs_dilation=(1, 1), rhs_dilation=(1, 1))
    return out + b[None, :, None, None]
  
# =============================================================
# Parameter Initialisation — random normal weights, zero biases
# =============================================================

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
    Forward pass of the spatial CNN phase classifier.
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
    """
    Cross-entropy loss function for binary phase classification:

        L = -(1/N)Σ_iΣ_c[y_c^i ⋅ log(p_c^i)]

    where:
        p_c = softmax(logits)_c (predicted probability of class c)
        y_c = one_hot(label)_c (true class indicator ∈ {0,1})
        N = batch size

    Parameters:
    params = dict of JAX arrays (current network parameters)
    x = input spatiotemporal images (batch, 2, depth, L)
    y = (batch,) int (true labels ∈ {0=volume-law, 1=area-law})

    Returns;
    loss = mean cross-entropy over batch
    """
    logits    = cnn_forward(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot   = jax.nn.one_hot(y, 2)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


# ==============
# TRAINING LOOP
# ==============

def train_cnn(img_train, y_train, img_val, y_val,
              epochs: int = 40, batch_size: int = 32, lr: float = 1e-3):
    """
    Train the CNN phase classifier via mini-batch gradient descent.

    Each epoch:
        1. Shuffle training set
        2. For each mini-batch:
              - g = ∇_θ L(θ, x_batch, y_batch) [JAX grad]
              - θ = Adam(θ, g) [parameter update]
        3. Evaluate cross-entropy loss on full training set
        4. Evaluate accuracy on validation set:
              - acc = (1/N) Σᵢ 𝟙[argmax(f(xᵢ)) = yᵢ]

    Gradients computed via JAX automatic differentiation (jit-compiled
    for performance). Parameters stored as JAX pytree dict.

    Parameters:
      - img_train = (N_train, 2, depth, L) (training images)
      - y_train = (N_train,) (training labels ∈ {0, 1})
      - img_val = (N_val, 2, depth, L) (validation images)
      - y_val = (N_val,) (validation labels ∈ {0, 1})
      - epochs = number of full passes over training set
      - batch_size = number of trajectories per gradient update
      - lr = Adam learning rate

    Returns:
      - params = trained parameter dict
      - tr_acc = final training accuracy
      - va_acc = final validation accuracy
      - history = dict with keys 'loss' and 'val_acc' (one value per epoch)
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
    """
    Return the predicted probability of the area-law phase for a
    single trajectory measurement record:

    P(area-law) = softmax(f(x))_1 = e^{z_1} / (e^{z_0} + e^{z_1})

    Used during the learnability sweep, evaluates the trained
    classifier at a single p_m value to trace the phase boundary.

    Parameters:
    - params = trained CNN parameter dict
    - img = single trajectory two-channel image (2, depth, L)

    Returns:
    - P(area-law) ∈ [0, 1]:
        - 0.0 = network certain this is volume-law
        - 0.5 = network uncertain — near the phase boundary p_c
        - 1.0 = network certain this is area-law
    """
    logits = cnn_forward(params, jnp.array(img[None, ...]))
    return float(jax.nn.softmax(logits, axis=-1)[0, 1])


"""
======================================================================================
(2)
Temporal Convolutional Network (TCN) phase classifier in JAX .

Architecture:
- Input projection → ResBlock(dilation=1) → ResBlock(dilation=2)
                   → ResBlock(dilation=4) → GlobalAvgPool → FC(32) → FC(2)

Each residual block uses causal 1D convolution — the representation
at time t depends only on measurements at t' ≤ t:

    out[t] = f(x[t], x[t-1], ..., x[t-k])   k = (kernel_size-1)·dilation

Increasing dilations (1→2→4) expand the receptive field exponentially
without adding parameters — at dilation 4 each output sees 4·(k-1)
past timesteps, capturing long-range temporal correlations that develop
near the critical point p_c.

Key distinction from CNN: causal ordering is strictly enforced —
the network processes the measurement record as it would arrive
in a real experiment, one layer at a time!

Input:
- causal sequence from 'encode_temporal()' (batch, depth, 2·L)
- depth (circuit layers (time axis))
- [was-measured mask | outcomes] at each layern (2·L)

Output : 
- logits [score(volume-law), score(area-law)] (batch, 2)
======================================================================================
"""
# Imports
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random as jrandom, grad, jit
import numpy as np
from src.training.adam import AdamOptimizer

SEED = 42

# ======================
# Causal 1D Convolution
# ======================

def causal_conv1d(x, w, b, dilation: int = 1):
    """
    Causal 1D convolution along the time axis with optional dilation.

    Causality is enforced by left-padding only — padding (pad_len, 0)
    adds zeros before the sequence so output at time t sees only:

        out[t] = Σ_{k=0}^{K-1} w[k] · x[t - k·dilation]   t' ≤ t

    where pad_len = (kernel_size - 1) · dilation ensures no future
    information leaks into the past.

    Dilation skips d-1 inputs between filter, expanding the
    receptive field without increasing parameters:

        dilation=1 : receptive field = kernel_size
        dilation=2 : receptive field = 2·(kernel_size-1) + 1
        dilation=4 : receptive field = 4·(kernel_size-1) + 1

    Parameters:
    - x = input sequence (batch, in_channels, time)
    - w = causal filters (out_channels, in_channels, kernel_size)
    - b = bias (out_channels,)
    - dilation = gap between filter taps (1=standard, 2/4=dilated)

    Returns:
    - out = causal output sequence (batch, out_channels, time)
            out[t] depends only on x[t'], t' ≤ t
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


# =========================
# Parameter Initialisation
# =========================

def init_tcn_params(key, in_features: int, hidden: int = 64,
                    kernel_size: int = 3):
    """
    Initialise all TCN parameters as a JAX pytree dict.

    Architecture:
        Input projection = (in_features, 1) → (hidden, 1)   kernel=1
        ResBlock 0 = (hidden, hidden, k)  dilation=1
        ResBlock 1 = (hidden, hidden, k)  dilation=2
        ResBlock 2 = (hidden, hidden, k)  dilation=4
        FC1 = (hidden, 32)
        FC2 = (32, 2)

    Each ResBlock has two causal conv layers (conv1, conv2).
    Weights ~ N(0, 0.05²), biases = 0.

    Parameters:
    - key = JAX PRNG key — for reproducible initialisation
    - in_features = input dimension => 2·L (mask + outcomes per layer)
    - hidden = number of channels throughout the TCN
    - kernel_size = causal filter width k (default 3)

    Returns:
    - p (dict of JAX arrays): 
        - proj_w, proj_b
        - b{0,1,2}_conv{1,2}_w, b{0,1,2}_conv{1,2}_b
        - fc1_w, fc1_b, fc2_w, fc2_b
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


# =============
# Forward Pass 
# =============

def tcn_forward(params, x):
    """
    Forward pass of the causal TCN phase classifier.

    Data flow:
        x  (batch, depth, 2·L)
        → transpose  (batch, 2·L, depth)
        → ReLU(CausalConv, dilation=1)  (batch, hidden, depth)
        → ReLU(CausalConv + skip, dilation=1)  (batch, hidden, depth)
        → ReLU(CausalConv + skip, dilation=2)  (batch, hidden, depth)
        → ReLU(CausalConv + skip, dilation=4)  (batch, hidden, depth)
        → GlobalAvgPool over time  (batch, hidden)
        → ReLU(FC(32))  (batch, 32)
        → FC(2)  (batch, 2)

    Each residual block:
        - h → ReLU(CausalConv₂(ReLU(CausalConv₁(h))) + h)

    Receptive field grows exponentially with dilation:
        - dilation=1 : sees k past steps
        - dilation=2 : sees 2(k-1)+1 past steps
        - dilation=4 : sees 4(k-1)+1 past steps

    Parameters:
    - params = dict of JAX arrays (from 'init_tcn_params()')
    - x = causal measurement sequence (batch, depth, 2·L) 

    Returns:
    - logits = [score(volume-law), score(area-law)] (batch, 2) 
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
    """
    Cross-entropy loss for TCN phase classification.

    - L = -1/N Σᵢ Σ_c y_c^(i) · log p_c^(i)

    where p_c = softmax(tcn_forward(x))_c

    Parameters:
    - params = dict of JAX arrays
    - x = causal measurement sequences (batch, depth, 2·L)
    - y = true labels ∈ {0=volume-law, 1=area-law} (batch,)

    Returns:
    - loss = mean cross-entropy over batch
    """
    logits    = tcn_forward(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(y, 2) * log_probs, axis=-1))

# =============
# Training TCN
# =============

def train_tcn(seq_train, y_train, seq_val, y_val,
              hidden: int = 64, kernel_size: int = 3,
              epochs: int = 60, batch_size: int = 32, lr: float = 1e-3):
    """
    Train the TCN phase classifier via mini-batch gradient descent.

    Each epoch:
        1. Shuffle training set
        2. For each mini-batch:
           - g = ∇_θ L(θ, x_batch, y_batch) (JAX autodiff)
           - θ = Adam(θ, g) (parameter update)
        3. Record mean cross-entropy loss over epoch
        4. Evaluate accuracy on validation set:
           - acc = (1/N) Σᵢ 𝟙[argmax(f(xᵢ)) = yᵢ]

    Parameters:
    - seq_train = training causal sequences (N_train, depth, 2·L)
    - y_train = training labels ∈ {0, 1} (N_train,)
    - seq_val = validation causal sequences (N_val, depth, 2·L)
    - y_val = validation labels ∈ {0, 1} (N_val,)
    - hidden = number of TCN channels throughout
    - kernel_size = causal filter width k
    - epochs = number of full passes over training set
    - batch_size = number of trajectories per gradient update
    - lr = Adam learning rate

    Returns:
    - params = trained parameter dict
    - tr_acc = final training accuracy
    - va_acc = final validation accuracy
    - history = 'loss' and 'val_acc', one value per epoch
    """
    in_features = seq_train.shape[2]   # 2xL
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
======================================================================================
(3)
Gated Recurrent Unit (GRU) phase classifier in JAX.

Processes the measurement record sequentially, one layer at a time.
At each step t the GRU updates its hidden state h_t ∈ ℝ^hidden via:

    z_t = σ(W_z [h_{t-1} ; x_t] + b_z) — update gate
    r_t = σ(W_r [h_{t-1} ; x_t] + b_r) — reset gate
    h̃_t = tanh(W_h [r_t ⊙ h_{t-1} ; x_t] + b_h) — candidate state
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  — new hidden state

where:
    [· ; ·] = concatenation
    ⊙ = elementwise multiplication (Hadamard product)
    σ = sigmoid activation ∈ (0,1)

Gates control information flow:
    z_t — how much of h_{t-1} to carry forward vs replace with h̃_t
    r_t — how much of h_{t-1} to expose when computing h̃_t

The final hidden state h_T summarises the full measurement history
and is passed to a two-class FC classifier.

Key distinction from TCN: the GRU maintains an explicit hidden state
that accumulates information across all past timesteps — no fixed
receptive field limit as in the dilated TCN.

Input: 
- causal sequence from encode_temporal (batch, depth, 2·L)
  x_t = [m_t⁰,...,m_t^(L-1), o_t⁰,...,o_t^(L-1)] ∈ {0,1}^{2L}

Output:        
— logits [score(volume-law), score(area-law)] (batch, 2) 
======================================================================================
"""
# Imports
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random as jrandom, grad, jit
import numpy as np
from src.training.adam import AdamOptimizer

SEED = 42

# =========================
# Parameter Initialisation
# =========================

# Initialise GRU parameters: W_z, W_r, W_h ~ N(0, 0.05²), biases = 0
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


# ================================================================
# GRU Cell — single step hidden state update via gating mechanism
# ================================================================

def gru_cell(params, h, x):
    """
    Single GRU step: update hidden state h_{t-1} → h_t given input x_t.

        hx = [h_{t-1} ; x_t] — concatenate
        z_t = σ(hx · W_z + b_z) — update gate
        r_t = σ(hx · W_r + b_r) — reset gate
        h̃_t = tanh([r_t ⊙ h_{t-1} ; x_t] · W_h + b_h) — candidate
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  — new state

    Parameters:
    - params = dict — W_z, W_r, W_h, b_z, b_r, b_h
    - h = previous hidden state h_{t-1} (batch, hidden)
    - x = current input x_t (batch, in_features)

    Returns:
    - h_t = updated hidden state (batch, hidden)
    """
    hx      = jnp.concatenate([h, x], axis=-1)
    z       = jax.nn.sigmoid(hx @ params['W_z'] + params['b_z'])
    r       = jax.nn.sigmoid(hx @ params['W_r'] + params['b_r'])
    rhx     = jnp.concatenate([r * h, x], axis=-1)
    h_tilde = jnp.tanh(rhx @ params['W_h'] + params['b_h'])
    return (1 - z) * h + z * h_tilde


# =======================================================
# Forward Pass — sequential scan over measurement layers
# =======================================================

def gru_forward(params, x):
    """
    Forward pass of the GRU phase classifier.
    Sequentially updates hidden state h_t across all depth layers
    via lax.scan — efficient JAX primitive for recurrent computation:
      - h_0 = 0
      - h_t = GRUCell(h_{t-1}, x_t) {t = 1, ..., depth}
      - logits = FC(ReLU(FC(h_depth)))

    Data flow:
        x  (batch, depth, 2·L)
        → transpose  (depth, batch, 2·L)
        → lax.scan(GRUCell)  h_T : (batch, hidden)
        → ReLU(FC(32))  (batch, 32)
        → FC(2)  (batch, 2)

    h_T encodes the full measurement history — unlike the TCN which
    has a fixed receptive field, the GRU in principle retains
    information from all past timesteps via the gating mechanism.

    Parameters:
    - params = dict of JAX arrays (from 'init_gru_params()')
    - x = causal measurement sequence (batch, depth, 2·L)

    Returns:
    - logits = [score(volume-law), score(area-law)] (batch, 2)
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
    """
    Cross-entropy loss for GRU phase classification.

    - L = -1/N Σᵢ Σ_c y_c^(i) · log p_c^(i)

    where p_c = softmax(gru_forward(x))_c

    Parameters:
    - params = dict of JAX arrays
    - x = causal measurement sequences (batch, depth, 2·L)
    - y = true labels ∈ {0=volume-law, 1=area-law} (batch,)

    Returns:
    - loss : mean cross-entropy over batch (scalar)
    """
    logits    = gru_forward(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(y, 2) * log_probs, axis=-1))


# =============
# Training GRU
# =============

def train_gru(seq_train, y_train, seq_val, y_val,
              hidden: int = 64,
              epochs: int = 60, batch_size: int = 32, lr: float = 1e-3):
    """
    Train the GRU phase classifier via mini-batch gradient descent.

    Each epoch:
        1. Shuffle training set
        2. For each mini-batch:
           - g = ∇_θ L(θ, x_batch, y_batch) — JAX autodiff
           - θ = Adam(θ, g) — parameter update
        3. Record mean cross-entropy loss over epoch
        4. Evaluate accuracy on validation set:
           - acc = (1/N) Σᵢ 𝟙[argmax(f(xᵢ)) = yᵢ]

    Parameters:
    - seq_train = training causal sequences (N_train, depth, 2·L)
    - y_train = training labels ∈ {0, 1} (N_train,)
    - seq_val = validation causal sequences (N_val, depth, 2·L) 
    - y_val = validation labels ∈ {0, 1} (N_val,)
    - hidden = GRU hidden state dimension
    - epochs = number of full passes over training set
    - batch_size = number of trajectories per gradient update
    - lr = Adam learning rate

    Returns:
    - params = trained parameter dict
    - tr_acc = final training accuracy
    - va_acc = final validation accuracy
    - history = dict — 'loss' and 'val_acc', one value per epoch
    """
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
