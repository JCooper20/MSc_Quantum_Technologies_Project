"""
Appendix B, Table B.2: the ceiling test per cell.

Reads out/ceiling/race_results.npz, keys meta|{target}|{w}|{L}|{N}|{r}
= [NLL_base, NLL_challenger, gap, replicate spread, n_val], where

    gap = NLL_base - NLL_challenger

so a POSITIVE gap favours the full state; and posctl_results.npz, keys
posctl|{N}|{r} = [NLL_head, NLL_null, pseudo-R^2].

Each of the 18 cells has 10 configurations: 5 history windows x 2
depths. The baseline ignores the embedding, so its NLL is independent
of depth and the median over 10 equals the median over 5.

Each gap is divided by ITS OWN baseline before the median is taken.
Next drop is a 5-class problem and remaining time a 7-class one, with
different baseline scales; one shared denominator would misstate one of
them.

Run from the project root:
    PYTHONPATH=. python scripts/table_ceiling.py
"""

# Imports
import os
from math import floor, log10

import numpy as np

OUT = "out/ceiling"


def sig3(x):
    """Three significant figures, trailing zeros kept."""
    if not np.isfinite(x):
        return "--"
    d = 2 - int(floor(log10(abs(x)))) if x else 2
    return f"{x:.{max(d, 0)}f}"


def main():
    z = np.load(os.path.join(OUT, "race_results.npz"))
    pc = np.load(os.path.join(OUT, "posctl_results.npz"))
    keys = sorted({k.split("|", 1)[1] for k in z.files
                   if k.startswith("meta|")})
    cells = sorted({(int(k.split("|")[3]), float(k.split("|")[4]))
                    for k in keys})

    rows = []
    for N, r in cells:
        rec = {"N": N, "r": r, "ctl": float(pc[f"posctl|{N}|{r:g}"][2])}
        for tgt in ("next_drop", "remaining"):
            sub = [k for k in keys if k.startswith(tgt)
                   and (int(k.split("|")[3]), float(k.split("|")[4]))
                   == (N, r)]
            nb = np.array([z[f"meta|{k}"][0] for k in sub])
            gap = np.array([z[f"meta|{k}"][2] for k in sub])
            rec[f"base_{tgt}"] = float(np.median(nb))
            rec[f"gap_{tgt}"] = float(np.median(100.0 * gap / nb))
            rec[f"n_{tgt}"] = len(sub)
        rows.append(rec)

    prev = None
    for x in rows:
        if prev is not None and x["N"] != prev:
            print(r"\addlinespace")
        prev = x["N"]
        g1, g2 = x["gap_next_drop"], x["gap_remaining"]
        print(f"{x['N']} & {x['r']:g} & {sig3(x['base_next_drop'])} & "
              f"{'+' if g1 > 0 else '-'}{sig3(abs(g1))} & "
              f"{'+' if g2 > 0 else '-'}{sig3(abs(g2))} & "
              f"{sig3(x['ctl'])} \\\\")

    g1 = np.array([x["gap_next_drop"] for x in rows])
    g2 = np.array([x["gap_remaining"] for x in rows])
    ctl = np.array([x["ctl"] for x in rows])
    n = len(rows)
    print(f"\n{n} cells, "
          f"{sorted({x['n_next_drop'] for x in rows})} configs each")
    print(f"next drop:  median {np.median(g1):+.3g}%, "
          f"range [{g1.min():+.3g}, {g1.max():+.3g}], "
          f"positive {int((g1 > 0).sum())}/{n}")
    print(f"remaining:  median {np.median(g2):+.3g}%, "
          f"range [{g2.min():+.3g}, {g2.max():+.3g}], "
          f"positive {int((g2 > 0).sum())}/{n}")
    print(f"control pseudo-R2 in [{ctl.min():.3g}, {ctl.max():.3g}], "
          f"bar 0.05, all pass {bool((ctl > 0.05).all())}")


if __name__ == "__main__":
    main()
