"""
Fit the order parameter near the transition to the power law

    S_R(t_late)/N = A * (r_c - r)^nu      (for r < r_c, else 0)

and extract r_c and nu. Fits each large N separately so you can see the
finite-size drift. 
"""

# Imports
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

REPO_ROOT  = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "checkpoint.json"
OUTDIR     = REPO_ROOT / "results" / "data" / "powerlaw_fit"

N_TAIL   = 5            # steady value = mean of last N_TAIL samples
S_MIN    = 1e-3         # only fit points with s above this (nonzero side)
N_FIT    = [128, 256]   # system sizes to fit (skip small N: not in scaling regime)


def load(path):
    raw = json.load(open(path))
    data = {}
    for N_str, r_dict in raw.items():
        N = int(N_str)
        for r_str, cell in r_dict.items():
            r = float(r_str)
            S = np.asarray(cell["mean"], float)
            data[(N, r)] = S / N
    return data


def plateau(data, N, r):
    return float(data[(N, r)][-N_TAIL:].mean())


def powerlaw(r, A, rc, nu):
    return A * np.clip(rc - r, 0, None) ** nu


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load(CHECKPOINT)
    Ns_present = sorted({N for (N, _) in data})

    fig, ax = plt.subplots(figsize=(8, 6))
    results = {}

    for N in N_FIT:
        if N not in Ns_present:
            print(f"  N={N} not in checkpoint, skipping")
            continue
        rs = np.array(sorted(r for (n, r) in data if n == N))
        ss = np.array([plateau(data, N, r) for r in rs])

        # fit only the nonzero side
        mask = ss > S_MIN
        rf, sf = rs[mask], ss[mask]
        if len(rf) < 4:
            print(f"  N={N}: too few points ({len(rf)}) to fit")
            continue

        try:
            popt, pcov = curve_fit(powerlaw, rf, sf,
                                   p0=[1.0, 0.5, 2.0], maxfev=20000)
            perr = np.sqrt(np.diag(pcov))
            A, rc, nu = popt
            dA, drc, dnu = perr
            results[N] = {"A": A, "r_c": rc, "nu": nu,
                          "r_c_err": drc, "nu_err": dnu, "n_points": int(len(rf))}
            print(f"  N={N:>4}: r_c = {rc:.3f} ± {drc:.3f},  "
                  f"nu = {nu:.2f} ± {dnu:.2f},  ({len(rf)} points)")

            xs = np.linspace(rf.min(), rc, 300)
            line, = ax.plot(rs, ss, "o", ms=5, label=f"N={N} data")
            ax.plot(xs, powerlaw(xs, *popt), "-", lw=2, color=line.get_color(),
                    label=fr"N={N} fit: $r_c$={rc:.3f}, $\nu$={nu:.2f}")
        except Exception as e:
            print(f"  N={N}: fit failed ({e})")

    ax.set_xlabel("r", fontsize=13)
    ax.set_ylabel(r"$S_R(t)/N$", fontsize=13)
    ax.set_title("Power-law fit", fontsize=13)
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "powerlaw_fit.png", dpi=150)
    plt.close(fig)

    json.dump(results, open(OUTDIR / "powerlaw_results.json", "w"), indent=2)
    print(f"\n  wrote {OUTDIR}/powerlaw_fit.png and powerlaw_results.json")



if __name__ == "__main__":
    main()
