"""
Extracts the measurement-induced critical point p_c from the CROSSING of
the tripartite-mutual-information curves I3(p_m) across system sizes N,
following Zabalo et al. (PRB 102, 064305, 2020). 

For each geometry it also performs a finite-size-scaling (FSS) collapse of
I3 onto the scaling form
    I3(p_m, L) = G[ (p_m - p_c) * L^{1/nu} ]
with the linear size L defined per geometry:
    1D brickwork    L = N
    2D lattice      L = sqrt(N)    
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
    """
    Linear size L used in the FSS ansatz for each geometry
    """
    if geom == "2d":
        return float(np.sqrt(N))
    if geom == "fc":
        return float(np.sqrt(N))
    return float(N)


def load(geom: str):
    """
    Return (Ns, pms, I3_mean, I3_sem, S_mean) for a geometry
    """
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
    than two intersections. The 200 trajectories per (N, p_m) are resampled with
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
    built from the other system sizes only, by linear interpolation at x_i.
   
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
                continue  # no two-sided support => skip 
            yhat = np.interp(xv, xo, yo)
            num += (yv - yhat) ** 2 / (sv ** 2 + 1e-6)
            cnt += 1
    if cnt < 8:
        return np.inf
    return num / cnt


def fit_collapse(geom, Ns, pms, ymean, ysem, pwin, pc_range, nu_range):
    """
    Grid-search then local refine (p_c, nu) minimising collapse residual.
    """
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

        # crossing-zoom figure (one per geometry) 
        fig, ax = plt.subplots(figsize=(7, 5))
        zoom = (pms >= 0.10) & (pms <= 0.45)
        for N in Ns:
            ax.errorbar(pms[zoom], I3m[N][zoom], yerr=I3s[N][zoom],
                        color=COLORS[N], marker="o", ms=4, lw=1.4,
                        capsize=2, label=f"$N={N}$")
        if not np.isnan(pc_cross):
            ax.axvspan(pc_cross - pc_spread, pc_cross + pc_spread,
                       color="grey", alpha=0.2)
            ax.axvline(pc_cross, color="k", ls="--", lw=1.2,
                       label=fr"$p_c={pc_cross:.3f}\pm{pc_spread:.3f}$")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(r"$p_m$")
        ax.set_ylabel(r"$I_3$")
        ax.set_title(fr"{DATA[geom][1]} — $I_3$ crossing")
        ax.legend()
        fig.tight_layout()
        out = f"{figdir}/I3_intersection_{geom}"
        fig.savefig(out + ".png", dpi=200)
        fig.savefig(out + ".pdf")
        plt.close(fig)

    #  combined collapse figure (3 panels) 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, geom in zip(axes, ("1d", "2d", "fc")):
        r = results[geom]
        Ns, pms, I3m, I3s = r["Ns"], r["pms"], r["I3m"], r["I3s"]
        pc, nu, Q = r["pc_fit"], r["nu_fit"], r["Q"]
        win = pwin[geom]
        for N in Ns:
            L = linear_size(geom, N)
            sel = (pms >= win[0]) & (pms <= win[1])
            x = (pms[sel] - pc) * L ** (1.0 / nu)
            ax.errorbar(x, I3m[N][sel], yerr=I3s[N][sel],
                        color=COLORS[N], marker="o", ms=4, lw=1.2,
                        capsize=2, label=f"$N={N}$")
        ax.set_xlabel(r"$(p_m - p_c)\,L^{1/\nu}$")
        ax.set_ylabel(r"$I_3$")
        ax.set_title(fr"{DATA[geom][1]}"
                     "\n"
                     fr"$p_c={pc:.3f},\ \nu={nu:.2f},\ Q={Q:.2f}$")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = f"{figdir}/I3_collapse"
    fig.savefig(out + ".png", dpi=200)
    fig.savefig(out + ".pdf")
    plt.close(fig)

    # report 
    print(f"{'geometry':<18}{'p_c (I3 crossing)':<24}{'nu (collapse)':<16}{'Q':<8}{'p_c (collapse)'}")
    print("-" * 72)
    for geom in ("1d", "2d", "fc"):
        r = results[geom]
        pc_c = f"{r['pc_cross']:.3f} ± {r['pc_spread']:.3f}"
        print(f"{DATA[geom][1]:<18}{pc_c:<24}{r['nu_fit']:<16.2f}{r['Q']:<8.2f}{r['pc_fit']:.3f}")
    print("-" * 72)
   


if __name__ == "__main__":
    main()
