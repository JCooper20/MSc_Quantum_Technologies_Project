"""
Appendix C, Table C.1: the reduced-model sweep, all 51 cells.

The archived fit_N*.npz predate the measured-plateau estimator, so
nothing is read from them: every cell is refit from its stored
20-trajectory anchors with src/analysis/plateau.py.

RNG contract. The stream default_rng(20260819) is consumed in exactly
the order figures/fig_shadow_order_parameter.py consumes it — per size,
per ascending rate, one roll then one 400-draw bootstrap for each cell
whose exact ensemble still has a non-zero mean. Cells the figure omits
are rolled from a side stream keyed on (N, r), leaving the main stream
byte-identical. So s_model here is bitwise equal to the figure's
plotted value, not merely consistent with it.

Read-only. Run from the project root:
    PYTHONPATH=. python scripts/table_model.py
"""

# Imports
import os

import numpy as np

from src.analysis.plateau import fit_and_roll

ANCHORS = os.path.expanduser("~/Downloads/shadow_sweep_out")
EXACT   = os.path.expanduser("~/scaling_results/raw")
SIZES   = (384, 512, 768)
RATES   = [round(0.42 + 0.005 * k, 4) for k in range(17)]
SEED    = 20260819


def sci(v):
    """Scientific notation, 3 s.f.; exact zero prints as 0."""
    if v == 0:
        return "0"
    m, e = f"{v:.2e}".split("e")
    return rf"${m}\times 10^{{{int(e)}}}$"


def lam(info):
    """
    A cell with no fit carries the sentinel 1e-9, a placeholder the roll
    consumes. Printing it as a number would read as an escape nine
    decades slower than the real ones, so it prints as n/a.
    """
    body = sci(info["lam"]) if info["fitted"] else "n/a"
    return body + (r"\,$^{\dagger}$" if info["lambda_unreliable"] else "")


def main():
    rng = np.random.default_rng(SEED)      # the figure's stream
    tab = {}

    for N in SIZES:
        for r in RATES:
            z = np.load(f"{ANCHORS}/stairs_N{N}_r{r:.4f}_anchor20.npz")
            sr = np.load(f"{EXACT}/cell_N{N}_r{r:.4f}.npz")["sr"]
            on_fig = (sr / N).mean(0)[-1] > 0
            cell_rng = rng if on_fig else np.random.default_rng(
                [SEED, N, int(round(r * 10000))])
            curve, info = fit_and_roll(z, cell_rng)
            info["s_model"] = float(curve[-1])
            if on_fig:                     # the figure's bootstrap draws
                for _ in range(400):
                    rng.integers(0, len(sr), len(sr))
            tab[(N, r)] = info

    for r in RATES:
        cells = []
        for N in SIZES:
            x = tab[(N, r)]
            kp = "--" if x["plateau_fallback"] else f"{x['k_plateau']:.2f}"
            cells += [sci(x["s_model"]), lam(x), kp]
        print(f"{r:g} & " + " & ".join(cells) + r" \\")

    v = list(tab.values())
    n = len(v)
    print(f"\n{n} cells refit.  "
          f"lambda-unreliable {sum(x['lambda_unreliable'] for x in v)}/{n}, "
          f"plateau fallback {sum(x['plateau_fallback'] for x in v)}/{n}, "
          f"s_model exactly zero {sum(x['s_model'] == 0 for x in v)}/{n}")
    unfit = [x for x in v if not x["fitted"]]
    print(f"sentinel (n/a) in {len(unfit)} cells, all also flagged: "
          f"{all(x['lambda_unreliable'] for x in unfit)}")
    for x in v:
        if x["lambda_unreliable"] and x["fitted"]:
            print(f"  flagged with a finite lambda = {x['lam']:.3e}: "
                  f"{x['n_escapes']} escapes, {x['n_pts']} survival "
                  f"points — the flag fires on the point count")


if __name__ == "__main__":
    main()
