"""
Plateau estimation for the reduced model (thesis Section 3.6).

A composite trajectory has two parts: a descent, in which the walk drops
one entropy level at a time, and a plateau, in which it sits at a low
level until it escapes to zero at a single exponential rate.

The plateau level is MEASURED, not assumed:

    k_plateau = mean over CENSORED trajectories of that trajectory's
                mean rung over [0.75 t_max, t_max)

Fewer than 3 censored trajectories cannot support this, and the cell is
flagged rather than fitted. Escape times are then measured from plateau
entry (first rung <= ceil(k_plateau), above zero) to purification, and
the rate follows from the Kaplan-Meier survival S(t) of those times:

    log S(t) = -lambda (t - t0)      fitted by least squares, t >= t0

with t0 the 25th percentile of observed escapes, which excludes the
initial transient. Only points with S > 0 enter the fit.

This replaces the fixed-level variant (K_PLATEAU = 3) kept in
shadow_sweep/composite.py for reproducibility of the Chapter 7 run.
"""

# Imports
import math

import numpy as np

from scripts.km_kernels import build_km_kernels
from shadow_sweep.composite import (FALLBACK_KM, km_with_band, path_curve,
                                    sample_escape, sample_resolved)

MIN_CENSORED = 3      # below this the level is unmeasurable
MIN_ESCAPES  = 5      # below this lambda is unreliable
MIN_SURV_PTS = 5      # below this lambda is unreliable
SENTINEL     = 1e-9   # placeholder rate when no fit is possible


# =====================================
#  Measure the plateau level
# =====================================

def measure_level(z, t_max):
    """
    k_plateau from the censored trajectories only. A trajectory that
    purified has left the plateau and says nothing about where it sits.

    Returns (k_plateau, n_censored); k_plateau is None when the level
    cannot be measured, and the caller must fall back to (3, 1.5).
    """
    off, R, E, ab = z["off"], z["rungs"], z["entries"], z["absorbed"]
    grid = np.arange(int(0.75 * t_max), t_max)
    vals = [float(path_curve(R[off[j]:off[j + 1]],
                             E[off[j]:off[j + 1]], grid).mean())
            for j in range(len(off) - 1) if not ab[j]]
    if len(vals) < MIN_CENSORED:
        return None, len(vals)
    return float(np.mean(vals)), len(vals)


# =====================================
#  Fit the escape rate
# =====================================

def fit_escape(z, t_max, k_cut):
    """
    lambda from the Kaplan-Meier survival of plateau-entry-to-
    purification times. Trajectories still alive at t_max are censored
    and enter through the risk sets only.

    Returns dict(lam, t0, km, n_escapes, n_pts, fitted, unreliable).
    A cell with lam = SENTINEL has no fit at all: that value is a
    placeholder the roll consumes, never a measured rate.
    """
    off, R, E = z["off"], z["rungs"], z["entries"]
    ab, tau = z["absorbed"], z["tau"]

    obs = []
    for j in range(len(off) - 1):
        rungs, entries = R[off[j]:off[j + 1]], E[off[j]:off[j + 1]]
        low = np.nonzero((rungs <= k_cut) & (rungs > 0))[0]
        if not len(low):
            continue
        t_entry = int(entries[low[0]])
        end = int(tau[j]) if ab[j] else t_max
        obs.append((end - t_entry, 1 if ab[j] else 0))

    km = km_with_band(obs)
    ev = [t for t, e in obs if e]

    def out(lam, t0, km_, n_pts, fitted):
        return dict(lam=lam, t0=t0, km=km_, n_escapes=len(ev),
                    n_pts=n_pts, fitted=fitted,
                    unreliable=len(ev) < MIN_ESCAPES
                    or n_pts < MIN_SURV_PTS)

    if not ev or not len(km):
        return out(SENTINEL, 0.0, FALLBACK_KM, 0, False)

    t0 = float(np.percentile(ev, 25))
    m = (km[:, 0] >= t0) & (km[:, 1] > 0)
    n_pts = int(m.sum())
    if n_pts < 2:
        return out(SENTINEL, t0, km, n_pts, False)

    x, y = km[m, 0], np.log(km[m, 1])
    c, *_ = np.linalg.lstsq(np.stack([np.ones_like(x), x], 1), y,
                            rcond=None)
    lam = -float(c[1])
    if not np.isfinite(lam) or lam <= 0:
        return out(SENTINEL, t0, km, n_pts, False)
    return out(lam, t0, km, n_pts, True)


# =====================================
#  Generate composite trajectories
# =====================================

def roll(N, kern, k_cut, level, esc, t_max, rng, n_walks):
    """
    Walks from S_R = N to zero with no quantum state anywhere: dwell
    times from the per-level kernels down to k_cut, then one escape
    time from the fitted law. Absorption is emergent — nothing sets a
    purified fraction.
    """
    paths = []
    for _ in range(n_walks):
        k, t, guard = N, 0, 0
        rungs, entries = [float(N)], [0]
        while k > k_cut and t <= t_max and guard < 20000:
            guard += 1
            if k not in kern:
                k -= 1
                continue
            d, nxt = sample_resolved(kern, k, rng)
            t, k = t + d, nxt
            if t <= t_max and k > k_cut:
                rungs.append(float(k))
                entries.append(t)
        if k == 0 and t <= t_max:
            rungs.append(0.0)
            entries.append(t)
        elif t <= t_max:
            e = sample_escape(esc["lam"], esc["t0"], esc["km"], rng)
            rungs.append(level)
            entries.append(t)
            if t + e <= t_max:
                rungs.append(0.0)
                entries.append(int(t + e))
        paths.append((np.array(rungs), np.array(entries)))
    return paths


def mean_curve(paths, grid, N):
    """Ensemble-mean S_R/N of a set of paths at the given times."""
    acc = np.zeros(len(grid))
    for rungs, entries in paths:
        i = np.searchsorted(entries, grid, side="right") - 1
        acc += rungs[np.clip(i, 0, len(rungs) - 1)]
    return acc / len(paths) / N


# =====================================
#  One-call pipeline
# =====================================

def fit_and_roll(z, rng, n_walks=20000):
    """
    Fit one cell's anchors and roll an ensemble. Returns
    (curve at the stored sample times, info) where info carries every
    fitted quantity and both reliability flags. Anyone reporting a
    number from info must report its flag with it.
    """
    N, t_max = int(z["N"]), int(z["t_max"])
    kp, n_cens = measure_level(z, t_max)
    k_cut, level = (3, 1.5) if kp is None else (int(math.ceil(kp)), kp)
    kern = build_km_kernels(z, np.arange(len(z["off"]) - 1))
    esc = fit_escape(z, t_max, k_cut)
    paths = roll(N, kern, k_cut, level, esc, t_max, rng, n_walks)
    return mean_curve(paths, z["times"].astype(float), N), dict(
        k_plateau=kp, k_cut=k_cut, level=level, n_censored=n_cens,
        plateau_fallback=kp is None, lam=esc["lam"],
        fitted=esc["fitted"], n_escapes=esc["n_escapes"],
        n_pts=esc["n_pts"], lambda_unreliable=esc["unreliable"])
