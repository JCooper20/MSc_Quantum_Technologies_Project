"""
training/adam.py
----------------
Canonical JAX Adam optimiser shared by all JAX models
(CNN, TCN, GRU, policy network).

Previously copy-pasted ~6 times in the original monolith.
"""

import jax
import jax.numpy as jnp


class AdamOptimizer:
    """
    Vanilla Adam optimiser for arbitrary JAX pytree parameter trees.

    Parameters
    ----------
    params : pytree
        Initial parameters (used only to initialise moment accumulators).
    lr : float
        Learning rate.
    beta1, beta2 : float
        Exponential decay rates for first and second moment estimates.
    eps : float
        Numerical stability constant.
    """

    def __init__(self, params, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m = jax.tree.map(jnp.zeros_like, params)
        self.v = jax.tree.map(jnp.zeros_like, params)
        self.t = 0

    def step(self, params, grads):
        """
        Perform one Adam update step.

        Parameters
        ----------
        params : pytree   — current parameters
        grads  : pytree   — gradients (same tree structure as params)

        Returns
        -------
        updated_params : pytree
        """
        self.t += 1
        b1, b2 = self.beta1, self.beta2

        self.m = jax.tree.map(
            lambda m, g: b1 * m + (1 - b1) * g, self.m, grads)
        self.v = jax.tree.map(
            lambda v, g: b2 * v + (1 - b2) * g ** 2, self.v, grads)

        m_hat = jax.tree.map(lambda m: m / (1 - b1 ** self.t), self.m)
        v_hat = jax.tree.map(lambda v: v / (1 - b2 ** self.t), self.v)

        return jax.tree.map(
            lambda p, m, v: p - self.lr * m / (jnp.sqrt(v) + self.eps),
            params, m_hat, v_hat)
