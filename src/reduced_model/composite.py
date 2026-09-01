"""
Seed-exact replay and the composite generator.

Densify: re-run a stored seed with fine logging so a trajectory archived
at 22 sample times becomes a full S_R staircase. Generate: fit a
composite walk to those staircases and roll ensembles with no quantum
state involved.

PLATEAU ESTIMATOR: this file uses the original fixed-level definition,
K_PLATEAU = 3 at level 1.5. That is what the Chapter 7 sweep ran under
and it is kept unchanged so those results stay reproducible. Section 3.6
describes the measured version, which is in plateau.py.

STIM MUST BE 1.16.0. Replay indexes stim's enumeration of the two-qubit
Cliffords by seeded numpy draws, so a different version gives
different-but-valid circuits and silently fails every comparison.
anchor_ok() is the guard: exact integer equality at all 22 stored times,
no tolerance.

Estimators are verbatim copies of campaign-certified machinery; do not
"improve" them. dense_replay copies run_cell_myriad.run_one's draw loop
rather than calling it, because run_one evaluates S_R only at the 22
stored times.
"""

# Imports
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.reduced_model.kernels import build_km_kernels, wilson   # noqa
from scripts.run_cell_myriad import (_two_qubit_cliffords,       # noqa
                                     _reference_entropy)

K_PLATEAU = 3
FALLBACK_KM = np.array([[1.0, 1.0, 1.0, 1.0]])


# =====================================
#  Seed-exact replay
# =====================================

def dense_replay(args):
    """
    args = (N, r, seed, t_max). Returns (rungs, entries, absorbed, tau,
    seed).

    System starts Bell-paired to the reference, so S_R = N. Each sweep is
    N operations: probability r a Z-measurement on a random system qubit,
    else a random 2-qubit Clifford on a random distinct pair. S_R is
    evaluated only on sweeps containing a measurement — a unitary cannot
    change it.
    """
    N, r, seed, t_max = args
    import stim
    rng = np.random.default_rng(seed)
    cliffords = _two_qubit_cliffords()
    sim = stim.TableauSimulator(seed=seed)
    sim.set_num_qubits(2 * N)
    for i in range(N):
        sim.h(N + i)
        sim.cnot(N + i, i)
    rungs, entries = [N], [0]
    cur = N
    absorbed, tau = False, -1
    for step in range(1, t_max + 1):
        n_meas = 0
        for _ in range(N):
            if rng.random() < r:
                q = int(rng.integers(0, N))
                sim.measure(q)
                n_meas += 1
            else:
                a = int(rng.integers(0, N))
                b = int(rng.integers(0, N - 1))
                if b >= a:
                    b += 1
                c = int(rng.integers(0, len(cliffords)))
                sim.do_tableau(cliffords[c], [a, b])
        if n_meas and cur > 0:
            s = int(_reference_entropy(sim, N))
            if s < cur:
                rungs.append(s)
                entries.append(step)
                cur = s
        if cur == 0:
            absorbed, tau = True, step
            break
    return (np.array(rungs, np.int16), np.array(entries, np.int32),
            absorbed, tau, seed)


# =====================================
#  Reading a staircase
# =====================================

def stair_at(rungs, entries, t):
    """S_R at sweep t: the level entered most recently at or before t."""
    from bisect import bisect_right
    return int(rungs[bisect_right(entries, t) - 1])


def anchor_ok(rungs, entries, times, stored_row):
    """
    The anchor gate. S_R is an integer, so this is equality, not a
    tolerance. One mismatch means the replay is not the archived
    trajectory and everything downstream is void.
    """
    vals = np.array([stair_at(rungs, entries, int(t)) for t in times])
    return np.array_equal(vals, stored_row.astype(int)), vals


def path_curve(rungs, entries, grid):
    """Vectorised stair_at over a grid of sweeps."""
    idx = np.searchsorted(entries, grid, side="right") - 1
    return rungs[np.clip(idx, 0, len(rungs) - 1)]


def traj_curves(z, idx, grid):
    """S_R(t) on a common grid for the selected trajectories."""
    off, R, E = z["off"], z["rungs"], z["entries"]
    out = np.empty((len(idx), len(grid)))
    for i, j in enumerate(idx):
        out[i] = path_curve(R[off[j]:off[j + 1]],
                            E[off[j]:off[j + 1]], grid)
    return out


# =====================================
#  Kaplan-Meier with a Greenwood band
# =====================================

def km_with_band(obs):
    """
    Columns (t, S, lo, hi), 95% Greenwood interval:

        Var(S) ~= S^2 sum_{t_i <= t} d_i / ( n_i (n_i - d_i) )

    The sum skips times where everyone at risk fails, where the term
    diverges and S is zero anyway.
    """
    ts = sorted({t for t, e in obs if e})
    S, var_acc = 1.0, 0.0
    out = []
    for t in ts:
        n_risk = sum(1 for d, e in obs if d >= t)
        d_t = sum(1 for d, e in obs if e and d == t)
        if n_risk == 0:
            break
        S *= 1 - d_t / n_risk
        if n_risk > d_t:
            var_acc += d_t / (n_risk * (n_risk - d_t))
        se = S * np.sqrt(var_acc)
        out.append((t, S, max(S - 1.96 * se, 1e-6),
                    min(S + 1.96 * se, 1.0)))
    return np.array(out)


def fit_plateau_params(z, fit_idx, t_max):
    """
    Fixed-level plateau fit (superseded; see plateau.py). Returns
    (lam, t_start, km, level, med_tp, n_escapes, n_surv_pts), with
    lam = 1e-9 the sentinel for unfittable.
    """
    off, R, E = z["off"], z["rungs"], z["entries"]
    ab, tau = z["absorbed"], z["tau"]
    obs, tps, lvls = [], [], []
    for j in fit_idx:
        Rr, Ee = R[off[j]:off[j + 1]], E[off[j]:off[j + 1]]
        low = np.nonzero((Rr <= K_PLATEAU) & (Rr > 0))[0]
        if not len(low):
            continue
        t_p = int(Ee[low[0]])
        end = int(tau[j]) if ab[j] else t_max
        obs.append((end - t_p, 1 if ab[j] else 0))
        tps.append(t_p)
        if end > t_p:
            g = np.arange(t_p, end)
            lvls.append(float(path_curve(Rr, Ee, g).mean()))
    km = km_with_band(obs)
    ev = [t for t, e in obs if e]
    if not ev or not len(km):
        return 1e-9, 0.0, FALLBACK_KM, 1.5, 0.0, 0, 0
    t_start = float(np.percentile(ev, 25))
    m = (km[:, 0] >= t_start) & (km[:, 1] > 0)
    n_pts = int(m.sum())
    if n_pts < 2:
        return 1e-9, 0.0, FALLBACK_KM, \
            float(np.mean(lvls)) if lvls else 1.5, \
            float(np.median(tps)), len(ev), n_pts
    x, y = km[m, 0], np.log(km[m, 1])
    A = np.stack([np.ones_like(x), x], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    lam = -float(coef[1])
    if not np.isfinite(lam) or lam <= 0:
        lam = 1e-9
    return lam, t_start, km, float(np.mean(lvls)), \
        float(np.median(tps)), len(ev), n_pts


# =====================================
#  The composite generator
# =====================================

def sample_resolved(kern, k, rng):
    """
    (duration, next_level) at level k. Inverse transform on the KM CDF,
    rescaled by F[-1] so the unresolved tail mass is redistributed over
    observed durations rather than sampled from. The target is drawn
    from visits that lasted exactly that long.
    """
    kk = kern[k]
    if not len(kk["durs"]):
        return 1, k - 1
    u = rng.random() * kk["F"][-1]
    i = min(int(np.searchsorted(kk["F"], u, side="right")),
            len(kk["durs"]) - 1)
    d = int(kk["durs"][i])
    pool = kk["nxt"][(k, d)]
    return d, int(pool[rng.integers(0, len(pool))])


def sample_escape(lam, t_start, km, rng):
    """
    Piecewise: below t_start sample the empirical KM curve, since the
    transient is not exponential; above it use t_start + Exp(lambda).
    F_start is the mass the empirical part carries.
    """
    F_start = 1.0 - float(np.interp(t_start, km[:, 0], km[:, 1]))
    if rng.random() < F_start:
        u = rng.random() * F_start
        F = 1.0 - km[:, 1]
        i = min(int(np.searchsorted(F, u)), len(km) - 1)
        return float(km[i, 0])
    return t_start + rng.exponential(1.0 / lam)


def roll_composite(N, kern, lam, t_start, km, level, t_max, rng,
                   n_walks):
    """
    Fixed-level roll. Returns (taus, paths), tau = -1 for a walk that did
    not purify within the horizon. Absorption is emergent.
    """
    taus = np.full(n_walks, -1, np.int64)
    paths = []
    for w in range(n_walks):
        k, t = N, 0
        rungs, entries = [N], [0]
        guard = 0
        while k > K_PLATEAU and t <= t_max:
            guard += 1
            if guard > 20000:
                break
            if k not in kern:
                k -= 1
                continue
            d, nx = sample_resolved(kern, k, rng)
            t += d
            k = nx
            if t <= t_max and k > K_PLATEAU:
                rungs.append(k)
                entries.append(t)
        if k == 0 and t <= t_max:
            rungs.append(0)
            entries.append(t)
            taus[w] = t
        elif t <= t_max:
            e = sample_escape(lam, t_start, km, rng)
            rungs.append(int(round(level)))
            entries.append(t)
            if t + e <= t_max:
                taus[w] = int(t + e)
                rungs.append(0)
                entries.append(int(t + e))
        paths.append((np.array(rungs, float), np.array(entries)))
    return taus, paths


def mean_curve(paths, grid, N, level):
    """
    Ensemble-mean S_R/N. The plateau rung is stored rounded inside the
    walk but the fitted level is fractional, so it is restored before
    averaging — rounding it in would bias the plateau height.
    """
    acc = np.zeros(len(grid))
    for rungs, entries in paths:
        rr = rungs.copy()
        rr[np.isclose(rr, round(level))] = level
        idx = np.searchsorted(entries, grid, side="right") - 1
        acc += rr[np.clip(idx, 0, len(rr) - 1)]
    return acc / len(paths) / N


# =====================================
#  One-call pipeline (fixed-level)
# =====================================

def fit_and_generate(stairs_npz, n_walks, seed=20260819):
    """
    Fit one cell and roll. Every trajectory in the file is used for
    fitting; the comparison is against the separate exact archive, not a
    held-out split of these.
    """
    z = np.load(stairs_npz)
    N = int(z["N"])
    t_max, times = int(z["t_max"]), z["times"]
    n = len(z["off"]) - 1
    idx = np.arange(n)
    kern = build_km_kernels(z, idx)
    lam, t_start, km, level, med_tp, n_esc, n_pts = \
        fit_plateau_params(z, idx, t_max)
    unreliable = (n_esc < 5) or (n_pts < 5)
    rng = np.random.default_rng(seed)
    taus, paths = roll_composite(N, kern, lam, t_start, km, level,
                                 t_max, rng, n_walks)
    curve = mean_curve(paths, times, N, level)
    return dict(curve=curve, times=times,
                absorbed_frac=float((taus >= 0).mean()),
                lam=lam, t_start=t_start, level=level,
                n_escapes=n_esc, n_surv_pts=n_pts,
                lambda_unreliable=bool(unreliable),
                anchor_absorbed=float(z["absorbed"].mean()))
