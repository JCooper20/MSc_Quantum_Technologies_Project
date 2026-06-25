"""
relaxation time tau(r, N) to the steady state, and its scaling with N.
For each (r, N): tau = time for S_R(t)/N to come within 5% of its steady value.
Then plot ln(tau) vs N for each r.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT  = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "checkpoint.json"
OUTDIR     = REPO_ROOT / "results" / "data" / "tau_scaling"
EPS, N_TAIL = 0.05, 5


def tau_of(t, y):
    """
    Time for y to close to within EPS of its steady value (mean of last N_TAIL).
    """
    s_inf = y[-N_TAIL:].mean()
    gap0  = abs(y[0] - s_inf)
    if gap0 < 1e-6:                       # never moves (r=0, or already pure)
        return None
    below = np.where(np.abs(y - s_inf) <= EPS * gap0)[0]
    if below.size == 0:                  # never reached steady state in the window
        return None
    return float(t[below[0]])


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw = json.load(open(CHECKPOINT))

    # tau for every (N, r)
    tau = {}                              # r -> [(N, tau)]
    for N_str, r_dict in raw.items():
        N = int(N_str)
        for r_str, cell in r_dict.items():
            r = float(r_str)
            t = np.asarray(cell["times"], float)
            y = np.asarray(cell["mean"], float) / N
            val = tau_of(t, y)
            if val is not None and val > 0:
                tau.setdefault(r, []).append((N, val))

    # plot ln(tau) vs N, one line per r
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in sorted(tau):
        pts = sorted(tau[r])
        if len(pts) < 2:
            continue
        N  = [p[0] for p in pts]
        lt = [np.log(p[1]) for p in pts]
        ax.plot(N, lt, "-o", ms=4, lw=1.2, label=f"r={r:.2f}")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\ln\,\tau$")
    ax.set_title(r"Relaxation time to steady state: $\ln\tau$ vs $N$")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTDIR / "tau_vs_N.png", dpi=200)
    plt.close(fig)

    # save the numbers
    json.dump({f"{r:.4f}": sorted(tau[r]) for r in sorted(tau)},
              open(OUTDIR / "tau.json", "w"), indent=2)


if __name__ == "__main__":
    main()
