"""
Plateau estimation (thesis Section 3.6).

A composite trajectory descends the entropy ladder with dwell times from
kernels.py, then sits at a low plateau until it escapes to zero.

The level is measured, not assumed:

    k_plateau = mean over CENSORED trajectories of that trajectory's
                mean rung over [0.75 t_max, t_max)

Only censored trajectories contribute — one that purified has left the
plateau. Fewer than three censored, and the cell is flagged.

Entry is the first rung <= ceil(k_plateau), above zero. The rate comes
from least squares on the log Kaplan-Meier survival of the
entry-to-purification times,

    log S(t) = -lambda (t - t0),    t >= t0

with t0 the 25th percentile of escapes (excludes the transient) and only
S > 0 points fitted. Single-rate exponential escape is the signature of
a quasi-stationary distribution.

Absorption is emergent; no purified fraction is set anywhere.

composite.py keeps the earlier fixed-level variant (K_PLATEAU = 3) that
the Chapter 7 sweep ran under. Use this module for new work.
"""

# Imports
import math

import numpy as np

from src.reduced_model.kernels import build_km_kernels
from src.reduced_model.composite import (FALLBACK_KM, km_with_band,
                                         path_curve, sample_escape,
                                         sample_resolved)

MIN_CENSORED = 3       # below this the level is unmeasurable
MIN_ESCAPES = 5        # below this lambda is unreliable
MIN_SURV_PTS = 5       # below this lambda is unreliable
SENTINEL = 1e-9        # placeholder rate, never a measurement


# =====================================
#  Measure the plateau level
# =====================================

def measure_level(z, t_max):
    """
    Returns (k_plateau, n_censored). k_plateau is None when fewer than
    MIN_CENSORED trajectories are censored; the caller must then fall
    back and flag.
    """
    off, R, E, ab = z["off"], z["rungs"], z["entries"], z["absorbed"]
    grid = np.arange(int(0.75 * t_max), t_max)
    vals = [float(path_curve(R[off[j]:off[j + 1]],
                             E[off[j]:off[j + 1]], grid).mean())
            for j in range(len(off) - 1) if not ab[j]]
    if len(vals) < MIN_CENSORED:
        return None, len(vals)
    return float(np.mean(vals)), len(vals)


def plateau_cut(k_plateau):
    """(k_cut, level, fell_back); (3, 1.5, True) is the fallback."""
    if k_plateau is None:
        return 3, 1.5, True
    return int(math.ceil(k_plateau)), float(k_plateau), False


# =====================================
#  Fit the escape rate
# =====================================

def fit_escape(z, t_max, k_cut):
    """
    Returns dict(lam, t0, km, n_escapes, n_pts, fitted, unreliable).

    Each trajectory reaching the plateau contributes one interval: entry
    to purification (event) or to the horizon (censored). lam = SENTINEL
    means unfittable — a placeholder the roll consumes, never reported
    as a rate.
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
#  Roll the generator
# =====================================

def roll(N, kern, k_cut, level, esc, t_max, rng, n_walks):
    """
    n_walks from S_R = N to zero, no quantum state anywhere. Descend on
    kernel draws to k_cut, then one escape draw. A level with no kernel
    steps down one rung; the guard is a runaway backstop.
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
    """Ensemble-mean S_R/N at the given sweeps."""
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
    Fit one cell's anchors and roll from them.

    Returns (curve at the stored sample times, info). Both flags in info
    depend on the fit window alone and are set before any walk is
    generated, so neither can be tuned to the answer — report them with
    any number taken from info.
    """
    N, t_max = int(z["N"]), int(z["t_max"])
    kp, n_cens = measure_level(z, t_max)
    k_cut, level, fell_back = plateau_cut(kp)
    kern = build_km_kernels(z, np.arange(len(z["off"]) - 1))
    esc = fit_escape(z, t_max, k_cut)
    paths = roll(N, kern, k_cut, level, esc, t_max, rng, n_walks)
    return mean_curve(paths, z["times"].astype(float), N), dict(
        k_plateau=kp, k_cut=k_cut, level=level, n_censored=n_cens,
        plateau_fallback=fell_back, lam=esc["lam"],
        fitted=esc["fitted"], n_escapes=esc["n_escapes"],
        n_pts=esc["n_pts"], lambda_unreliable=esc["unreliable"])
