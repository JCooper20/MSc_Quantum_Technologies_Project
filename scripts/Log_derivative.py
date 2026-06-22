"""
Logarithmic time derivative of the reference entropy
(Gullans & Huse, PRX 10, 041020, Fig. 6(b) inset).

Quantity:   -d S_R / d(ln t)  ~  -ΔS / Δ(ln t)  via np.gradient.

Physics:
  In the MIXED phase, S_R(t)/N ≈ s_inf - c·ln(t), so -dS/d(ln t) -> const
  at late times, independent of N.  A constant late-time value signals an
  exponentially long purification time τ ~ exp(a·N).  Plotting against
  t / N^z and scanning z probes the dynamical exponent (z=1 in 1D; unknown
  for all-to-all).

Source: results/checkpoints/checkpoint.json (nested {str(N): {f"{r:.4f}":
cell}}).

Pipeline:
  1. for each (N, r): compute S_R/N and  -ΔS/Δ(ln t)
  2. panel A: entangling r,  curves vs t/N^z, log-log
  3. panel B: transition/pure r, for contrast
  4. dump computed derivatives to results.json
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT  = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "checkpoint.json"
OUTDIR     = REPO_ROOT / "results" / "data" / "log_derivative"

Z_EXPONENT    = 1.0                          # x-axis scaling: t / N^z
RS_ENTANGLING = [0.10, 0.20, 0.30, 0.40]     # panel A
RS_PURE       = [0.50, 0.60, 0.70, 0.90]     # panel B


# =====================================================================
# Step 1: load
# =====================================================================

def load_checkpoint(path: Path) -> dict:
    """Return {(N, r) -> {'t', 'y'}} with y = S_R/N."""
    raw = json.load(open(path))
    out: dict = {}
    for N_str, r_dict in raw.items():
        N = int(N_str)
        for r_str, cell in r_dict.items():
            r = float(r_str)
            S = np.asarray(cell["mean"], dtype=float)
            out[(N, r)] = {"t": np.asarray(cell["times"], dtype=float),
                           "y": S / N}
    return out


# =====================================================================
# Step 2: logarithmic time derivative
# =====================================================================

def log_derivative(t: np.ndarray, y: np.ndarray):
    """
    Return (t_pos, -dY/d ln t).  Drops t<=0 (ln undefined) and uses
    np.gradient on y w.r.t. ln(t).  The negation makes the decaying-S
    signal positive.
    """
    pos = t > 0
    tp, yp = t[pos], y[pos]
    if tp.size < 3:
        return tp, np.full_like(tp, np.nan)
    dydlnt = np.gradient(yp, np.log(tp))
    return tp, -dydlnt


# =====================================================================
# Step 3: figure
# =====================================================================

def make_figure(data: dict, outpath: Path) -> dict:
    """Two panels: entangling (A) and pure/transition (B).  Returns the
    computed derivatives for JSON dump."""
    Ns = sorted({N for (N, _) in data})
    computed: dict = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    panels = [(axes[0], RS_ENTANGLING, "entangling"),
              (axes[1], RS_PURE,       "transition / pure")]

    for ax, r_targets, label in panels:
        for r_target in r_targets:
            for N in Ns:
                if (N, r_target) not in data:
                    continue
                d = data[(N, r_target)]
                tp, deriv = log_derivative(d["t"], d["y"])
                if tp.size < 3:
                    continue
                x = tp / (N ** Z_EXPONENT)
                ax.plot(x, deriv, "-o", ms=3, lw=1.2,
                        label=f"$N={N}, r={r_target:.2f}$")
                computed.setdefault(f"N{N}_r{r_target:.4f}",
                                    {"t": tp.tolist(), "x": x.tolist(),
                                     "deriv": deriv.tolist()})
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(fr"$t / N^{{{Z_EXPONENT:g}}}$")
        ax.set_ylabel(r"$-\,\Delta S_R / \Delta \ln t$")
        ax.set_title(f"Log-derivative — {label} phase")
        ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outpath / f"log_derivative.{ext}", dpi=200)
    plt.close(fig)
    return computed


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {CHECKPOINT}")
    data = load_checkpoint(CHECKPOINT)
    print(f"  {len(data)} (N, r) cells loaded")

    print(f"Writing figure to {OUTDIR}/log_derivative.{{png,pdf}}")
    computed = make_figure(data, OUTDIR)

    print(f"Writing {OUTDIR}/results.json")
    with open(OUTDIR / "results.json", "w") as fh:
        json.dump({"config": {"checkpoint": str(CHECKPOINT),
                              "z_exponent": Z_EXPONENT},
                   "log_derivative": computed}, fh, indent=2)

    print("\nDone. Check whether the entangling-phase curves (panel A) "
          "flatten to a constant at late t/N^z — that confirms the "
          "exponentially long purification time.")


if __name__ == "__main__":
    main()
