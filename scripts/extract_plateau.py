"""
Extract the thermodynamic-limit purification plateau value s_inf(r) by
finite-size extrapolation of the late-time S_R(t)/N.

Source: results/checkpoints/checkpoint.json (nested {str(N): {f"{r:.4f}":
cell}}; each cell has times, mean=raw S_R, sem, n_traj).  We divide by N.

Pipeline:
  1. plateau_value(N, r) = mean of S_R/N over the last n_tail sample times
  2. for each r, fit s_plateau vs 1/N; intercept = s_inf(r).  N<=8 is
     EXCLUDED from the fit (outside the scaling regime) but kept in the
     plot as open grey circles.
  3. two-panel figure: extrapolation (left), s_inf(r) vs r (right)
  4. dump every number to results.json
"""

import os
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT  = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "checkpoint.json"
OUTDIR     = REPO_ROOT / "results" / "data" / "plateau_analysis"
N_TAIL     = 5                          # plateau = mean of last N_TAIL samples
DROP_N     = 8                          # exclude N <= this from the fit
REP_RS     = [0.10, 0.20, 0.30, 0.40]   # left-panel representatives


# =====================================================================
# Step 1: load + per-cell plateau
# =====================================================================

def load_checkpoint(path: Path) -> dict:
    """Return {(N, r) -> {'t', 'y', 'ysem', 'n_traj'}} with y = S_R/N."""
    raw = json.load(open(path))
    out: dict = {}
    for N_str, r_dict in raw.items():
        N = int(N_str)
        for r_str, cell in r_dict.items():
            r = float(r_str)
            S    = np.asarray(cell["mean"], dtype=float)
            Ssem = np.asarray(cell["sem"],  dtype=float)
            out[(N, r)] = {"t":      np.asarray(cell["times"], dtype=float),
                           "y":      S / N,
                           "ysem":   Ssem / N,
                           "n_traj": int(cell["n_traj"])}
    return out


def plateau_value(y: np.ndarray, ysem: np.ndarray,
                  n_tail: int = N_TAIL) -> Tuple[float, float]:
    """Mean of the last n_tail S_R/N values; SEM combined in quadrature."""
    mu     = float(y[-n_tail:].mean())
    sem_mu = float(np.sqrt((ysem[-n_tail:] ** 2).sum()) / n_tail)
    return mu, sem_mu


# =====================================================================
# Step 2: 1/N extrapolation per r  (N <= DROP_N excluded from the fit)
# =====================================================================

def linear_fit_1overN(N_vals: np.ndarray, s_vals: np.ndarray) -> dict:
    """
    Fit s_plateau = slope*(1/N) + intercept on N > DROP_N only.
    intercept = s_inf; its SEM comes from the polyfit covariance.
    """
    keep = N_vals > DROP_N
    Nf, sf = N_vals[keep], s_vals[keep]
    if Nf.size < 2:
        return {"s_inf": float("nan"), "s_inf_sem": float("nan"),
                "slope": float("nan"), "r2": float("nan"),
                "n_points": int(Nf.size)}
    x = 1.0 / Nf
    if Nf.size >= 3:
        coef, cov = np.polyfit(x, sf, 1, cov=True)
        slope, intercept = float(coef[0]), float(coef[1])
        intercept_sem = float(np.sqrt(cov[1, 1]))
    else:
        coef = np.polyfit(x, sf, 1)
        slope, intercept = float(coef[0]), float(coef[1])
        intercept_sem = float("nan")
    yhat   = slope * x + intercept
    ss_res = float(np.sum((sf - yhat) ** 2))
    ss_tot = float(np.sum((sf - sf.mean()) ** 2)) or 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"s_inf": intercept, "s_inf_sem": intercept_sem,
            "slope": slope, "r2": r2, "n_points": int(Nf.size)}


# =====================================================================
# Step 3: figure
# =====================================================================

def make_figure(by_r: dict, fits: dict, outpath: Path) -> None:
    """Left: extrapolation for REP_RS (N<=8 open grey).  Right: s_inf(r)."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))

    cmap = plt.cm.viridis(np.linspace(0, 1, len(REP_RS)))
    for color, r_target in zip(cmap, REP_RS):
        r = min(by_r.keys(), key=lambda x: abs(x - r_target))
        pts = sorted(by_r[r])                          # [(N, s_plat, sem)]
        Ns     = np.array([p[0] for p in pts], float)
        s_plat = np.array([p[1] for p in pts], float)
        s_sem  = np.array([p[2] for p in pts], float)
        fit_mask = Ns > DROP_N

        axL.errorbar(1.0 / Ns[fit_mask], s_plat[fit_mask],
                     yerr=s_sem[fit_mask], marker="o", lw=0,
                     elinewidth=1.2, capsize=2, color=color, ms=5,
                     label=f"$r={r:.2f}$ (fit)")
        if (~fit_mask).any():
            axL.errorbar(1.0 / Ns[~fit_mask], s_plat[~fit_mask],
                         yerr=s_sem[~fit_mask], marker="o", lw=0,
                         elinewidth=1.0, capsize=2, mfc="none",
                         mec="grey", ecolor="grey", ms=6)
        f = fits[r]
        xs = np.linspace(0.0, 1.0 / Ns[fit_mask].min() * 1.05, 50)
        axL.plot(xs, f["slope"] * xs + f["s_inf"], "-",
                 color=color, lw=1.2, alpha=0.8)
        axL.plot([0.0], [f["s_inf"]], marker="*", ms=14, color=color,
                 mec="black", mew=0.6)

    axL.plot([], [], "o", mfc="none", mec="grey",
             label=f"$N\\leq{DROP_N}$ (excluded)")
    axL.set_xlabel(r"$1/N$")
    axL.set_ylabel(r"$s(N,r) = \langle S_R(t_{\rm late})\rangle / N$")
    axL.set_title(r"Finite-size extrapolation:  star $=s_\infty(r)$")
    axL.legend(fontsize=8); axL.set_xlim(left=0.0)

    rs     = sorted(fits.keys())
    s_infs = np.array([fits[r]["s_inf"]     for r in rs])
    s_sems = np.array([fits[r]["s_inf_sem"] for r in rs])
    axR.errorbar(rs, s_infs, yerr=s_sems, marker="o", lw=1.4, ms=5,
                 capsize=3, color="C0")
    axR.axhline(0, color="k", lw=0.8)
    axR.set_xlabel(r"measurement rate $r$")
    axR.set_ylabel(r"$s_\infty(r)$  (thermodynamic-limit plateau)")
    axR.set_title(r"Order parameter $s_\infty(r)$ vs $r$")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outpath / f"plateau_extrapolation.{ext}", dpi=200)
    plt.close(fig)


# =====================================================================
# Step 4: write results.json
# =====================================================================

def save_results(plateaus: dict, fits: dict, outpath: Path) -> None:
    out = {
        "config": {"checkpoint": str(CHECKPOINT), "n_tail": N_TAIL,
                   "drop_N": DROP_N, "rep_rs": REP_RS},
        "s_plateau": {
            str(N): {f"{r:.4f}": {"s": float(s), "sem": float(sem)}
                     for (N2, r), (s, sem) in plateaus.items() if N2 == N}
            for N in sorted({n for (n, _) in plateaus})
        },
        "s_inf": {f"{r:.4f}": fits[r] for r in sorted(fits)},
    }
    with open(outpath / "results.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {CHECKPOINT}")
    data = load_checkpoint(CHECKPOINT)
    print(f"  {len(data)} (N, r) cells loaded")

    plateaus = {}                                  # {(N, r) -> (s_plat, sem)}
    for (N, r), d in data.items():
        plateaus[(N, r)] = plateau_value(d["y"], d["ysem"])

    by_r: dict = {}                                # {r -> [(N, s_plat, sem)]}
    for (N, r), (s, sem) in plateaus.items():
        by_r.setdefault(r, []).append((N, s, sem))

    fits = {}                                      # {r -> {s_inf, ...}}
    for r, pts in by_r.items():
        pts = sorted(pts)
        N_arr = np.array([p[0] for p in pts], float)
        s_arr = np.array([p[1] for p in pts], float)
        fits[r] = linear_fit_1overN(N_arr, s_arr)

    print(f"Writing figure to {OUTDIR}/plateau_extrapolation.{{png,pdf}}")
    make_figure(by_r, fits, OUTDIR)
    print(f"Writing {OUTDIR}/results.json")
    save_results(plateaus, fits, OUTDIR)

    print()
    print(f"  {'r':>6}  {'s_inf':>9}  {'± sem':>8}  {'R^2':>5}  n")
    for r in sorted(fits):
        f = fits[r]
        print(f"  {r:>6.3f}  {f['s_inf']:>9.4f}  "
              f"{f['s_inf_sem']:>8.4f}  {f['r2']:>5.3f}  {f['n_points']}")


if __name__ == "__main__":
    main()
