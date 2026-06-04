"""
Re-analysis of the saved Phase A sweep data (no new simulations).

Extracts the measurement-induced critical point p_c from the CROSSING of
the tripartite-mutual-information curves I3(p_m) across system sizes N,
following Zabalo et al. (PRB 102, 064305, 2020). Raw I3 is extensive in
the volume-law phase (I3 ~ -N), so the size-resolved curves are nested
and intersect only where their ordering reverses near criticality — that
reversal point is p_c.

For each geometry it also performs a finite-size-scaling (FSS) collapse of
I3 onto the scaling form
    I3(p_m, L) = G[ (p_m - p_c) * L^{1/nu} ]
with the linear size L defined per geometry:
    1D brickwork    L = N
    2D lattice      L = sqrt(N)        (so L^{1/nu} = N^{1/(2 nu)})
    fully connected L = sqrt(N)
"""

import json
import itertools
import numpy as np
import matplotlib.pyplot as plt

DATA = {
    "1d": ("results/data/sweep_1d_brickwork.json",   "1D brickwork"),
    "2d": ("results/data/sweep_2d_lattice.json",      "2D lattice"),
    "fc": ("results/data/sweep_fully_connected.json", "Fully connected"),
}
COLORS = {16: "C0", 36: "C1", 64: "C2", 121: "C3"}


def linear_size(geom: str, N: int) -> float:
    """Linear size L used in the FSS ansatz for each geometry."""
    if geom == "2d":
        return float(np.sqrt(N))
    if geom == "fc":
        return float(np.sqrt(N))
    return float(N)


def load(geom: str):
    """Return (Ns, pms, I3_mean, I3_sem, S_mean) for a geometry."""
    path = DATA[geom][0]
    d = json.load(open(path))
    Ns = sorted((int(k) for k in d["I3"].keys()))
    pms = np.array(sorted((float(p) for p in d["I3"][str(Ns[0])].keys())))
    I3_mean, I3_sem, S_mean = {}, {}, {}
    for N in Ns:
        i3 = np.array([np.mean(d["I3"][str(N)][f"{p:g}"]) for p in pms])
        se = np.array([np.std(d["I3"][str(N)][f"{p:g}"]) / np.sqrt(len(d["I3"][str(N)][f"{p:g}"]))
                       for p in pms])
        sm = np.array([np.mean(d["S"][str(N)][f"{p:g}"]) for p in pms])
        I3_mean[N], I3_sem[N], S_mean[N] = i3, se, sm
    return Ns, pms, I3_mean, I3_sem, S_mean


# ----------------------------------------------------------------------
# 1. Crossing extraction
# ----------------------------------------------------------------------
def find_crossings(Ns, pms, I3_mean, window=(0.10, 0.50)):
    """
    Find pairwise intersections of I3(p_m) curves within `window`.
    For every pair (Ni, Nj) the difference d(p) = I3_Ni(p) - I3_Nj(p)
    is interpolated on a fine grid; sign changes give crossing p_m's.
    """
    grid = np.linspace(window[0], window[1], 2001)
    crossings = []
    for Ni, Nj in itertools.combinations(Ns, 2):
        di = np.interp(grid, pms, I3_mean[Ni])
        dj = np.interp(grid, pms, I3_mean[Nj])
        diff = di - dj
        sign = np.sign(diff)
        idx = np.where(np.diff(sign) != 0)[0]
        for k in idx:
            x0, x1 = grid[k], grid[k + 1]
            y0, y1 = diff[k], diff[k + 1]
            if y1 != y0:
                crossings.append(x0 - y0 * (x1 - x0) / (y1 - y0))
    return np.array(crossings)


def _bootstrap_crossing_std(geom, Ns, pms, window, B=400, seed=0):
    """
    Bootstrap estimate of the crossing-location uncertainty.

    Used as a fallback when the deterministic pairwise method yields fewer
    than two intersections (the fully-connected I3 curves are nested and
    merge toward zero rather than reversing, so only the two smallest sizes
    intersect). The 200 trajectories per (N, p_m) are resampled with
    replacement, the median pairwise crossing recomputed each time, and the
    standard deviation of those medians returned.
    """
    d = json.load(open(DATA[geom][0]))
    raw = {N: {f"{p:g}": np.asarray(d["I3"][str(N)][f"{p:g}"]) for p in pms}
           for N in Ns}
    rng = np.random.default_rng(seed)
    meds = []
    for _ in range(B):
        means = {}
        for N in Ns:
            row = []
            for p in pms:
                arr = raw[N][f"{p:g}"]
                row.append(arr[rng.integers(0, len(arr), len(arr))].mean())
            means[N] = np.asarray(row)
        cr = find_crossings(Ns, pms, means, window)
        if len(cr):
            meds.append(np.median(cr))
    return float(np.std(meds)) if meds else float("nan")


def crossing_pc(geom, Ns, pms, I3_mean, window):
    """
    Crossing p_c and its uncertainty for one geometry.

    The point estimate is the median of the deterministic pairwise
    intersections of the size-resolved I3 curves. When two or more such
    intersections exist (1D, 2D) their standard deviation is the
    uncertainty. When fewer than two exist (fully connected, whose curves
    merge rather than reverse) the deterministic spread is degenerate, so
    the uncertainty is taken from a bootstrap over trajectories instead.
    """
    cr = find_crossings(Ns, pms, I3_mean, window)
    if len(cr) >= 2:
        return float(np.median(cr)), float(np.std(cr))
    pc = float(np.median(cr)) if len(cr) else float("nan")
    return pc, _bootstrap_crossing_std(geom, Ns, pms, window)


# ----------------------------------------------------------------------
# 2. FSS collapse
# ----------------------------------------------------------------------
def collapse_residual(p_c, nu, geom, Ns, pms, ymean, ysem, pwin):
    """
    Cross-size leave-one-out collapse quality (Houdayer-Hartmann style).

    Each point (x_i, y_i) of size N_i is compared against the master curve
    built from the OTHER system sizes only, by linear interpolation at x_i.
    The residual is normalised by the combined statistical error. This
    penalises the trivial nu -> infinity solution: there L^{1/nu} -> 1, all
    sizes land at the same x but keep their (extensive) y-scatter, giving a
    large cross-size residual.
    """
    data = {}
    for N in Ns:
        L = linear_size(geom, N)
        sel = (pms >= pwin[0]) & (pms <= pwin[1])
        x = (pms[sel] - p_c) * L ** (1.0 / nu)
        order = np.argsort(x)
        data[N] = (x[order], ymean[N][sel][order], ysem[N][sel][order])

    num, cnt = 0.0, 0
    for N in Ns:
        xi, yi, si = data[N]
        xo = np.concatenate([data[M][0] for M in Ns if M != N])
        yo = np.concatenate([data[M][1] for M in Ns if M != N])
        oo = np.argsort(xo)
        xo, yo = xo[oo], yo[oo]
        for xv, yv, sv in zip(xi, yi, si):
            if xv < xo[0] or xv > xo[-1]:
                continue  # no two-sided support -> skip (no extrapolation)
            yhat = np.interp(xv, xo, yo)
            num += (yv - yhat) ** 2 / (sv ** 2 + 1e-6)
            cnt += 1
    if cnt < 8:
        return np.inf
    return num / cnt


def fit_collapse(geom, Ns, pms, ymean, ysem, pwin, pc_range, nu_range):
    """Grid-search then local refine (p_c, nu) minimising collapse residual."""
    best = (np.inf, None, None)
    for p_c in pc_range:
        for nu in nu_range:
            r = collapse_residual(p_c, nu, geom, Ns, pms, ymean, ysem, pwin)
            if r < best[0]:
                best = (r, p_c, nu)
    _, pc0, nu0 = best
    pc_fine = np.linspace(pc0 - 0.02, pc0 + 0.02, 21)
    nu_fine = np.linspace(max(0.3, nu0 - 0.3), nu0 + 0.3, 31)
    for p_c in pc_fine:
        for nu in nu_fine:
            r = collapse_residual(p_c, nu, geom, Ns, pms, ymean, ysem, pwin)
            if r < best[0]:
                best = (r, p_c, nu)
    return best  # (residual, p_c, nu)


def collapse_quality(geom, Ns, pms, ymean, ysem, pwin, p_c, nu):
    """
    Normalised goodness-of-collapse statistic Q.

    Q = sqrt( <(I3 - I3_master)^2 / sem^2> ) evaluated with the same
    cross-size leave-one-out residual that drives the collapse fit.
    Q ~ 1 means the size-resolved data fall onto a single curve to within
    their statistical error bars; Q >> 1 means a poor collapse.
    """
    return float(np.sqrt(
        collapse_residual(p_c, nu, geom, Ns, pms, ymean, ysem, pwin)))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    figdir = "results/figures"
    pwin = {"1d": (0.16, 0.45), "2d": (0.16, 0.45), "fc": (0.20, 0.55)}
    cross_win = {"1d": (0.20, 0.45), "2d": (0.20, 0.45), "fc": (0.30, 0.55)}

    results = {}
    for geom in ("1d", "2d", "fc"):
        Ns, pms, I3m, I3s, Sm = load(geom)
        pc_cross, pc_spread = crossing_pc(geom, Ns, pms, I3m, cross_win[geom])
        res, pc_fit, nu_fit = fit_collapse(
            geom, Ns, pms, I3m, I3s, pwin[geom],
            pc_range=np.linspace(0.18, 0.50, 33),
            nu_range=np.linspace(0.5, 2.5, 41),
        )
        Q = collapse_quality(geom, Ns, pms, I3m, I3s, pwin[geom], pc_fit, nu_fit)
        results[geom] = dict(Ns=Ns, pms=pms, I3m=I3m, I3s=I3s,
                             pc_cross=pc_cross, pc_spread=pc_spread,
                             pc_fit=pc_fit, nu_fit=nu_fit, resid=res, Q=Q)
