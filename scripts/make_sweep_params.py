"""
Generate scripts/sweep_params.txt for the Myriad FSS sweep:
one line per cell, "task_id N r", task ids 1-based.

Ordering: sizes ascending, rates ascending within each size — so the
first tasks are the cheapest cells (N=64), which is what the test
array (tasks 1-3) relies on.

Run from the project root:
    python scripts/make_sweep_params.py
"""

import numpy as np

SIZES = [64, 96, 128, 192, 256, 384, 512, 768, 1024]
R_VALUES = [round(r, 4) for r in np.arange(0.25, 0.5001, 0.005)]

if __name__ == "__main__":
    lines = []
    tid = 1
    for N in SIZES:
        for r in R_VALUES:
            lines.append(f"{tid} {N} {r:.4f}")
            tid += 1
    with open("scripts/sweep_params.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote scripts/sweep_params.txt: {tid - 1} cells "
          f"({len(SIZES)} sizes x {len(R_VALUES)} rates)")
