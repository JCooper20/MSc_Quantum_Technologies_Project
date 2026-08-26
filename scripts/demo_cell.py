"""
Minimal end-to-end demo: one (N, r) cell, Bell pairs to order parameter.

Builds the Bell-paired initial state, runs the monitored all-to-all
circuit, records S_R on the campaign's sample schedule, and reports

    s(N, r) = (1/M) sum_m (1/5N) sum_{j=18}^{22} S_R^(m)(t_j)

with its trajectory standard error. Also asserts the two properties the
whole analysis rests on: S_R is monotone non-increasing along every
trajectory, and stays in [0, N].

Needs only stim and numpy. Seconds at N = 16.

Run from the project root:
    PYTHONPATH=. python scripts/demo_cell.py
"""

# Imports
import time

import numpy as np
import stim

from src.analysis.entropy import stabiliser_entropy_region

N       = 16
R       = 0.35
M       = 20      # trajectories
DEPTH   = 8       # t_max = DEPTH * N sweeps


# =====================================
#  Sample times: 14 geometric + 10 linear
# =====================================

def sample_times(t_max):
    """Geometric points resolve the descent, linear ones the plateau."""
    t = np.unique(np.round(np.concatenate([
        np.geomspace(1, t_max, 14), np.linspace(1, t_max, 10)])))
    return [int(x) for x in t if 1 <= x <= t_max]


# =====================================
#  One trajectory
# =====================================

def run(N, r, times, t_max, seed):
    """
    System S = {0..N-1}, reference R = {N..2N-1}. Per operation: with
    probability r a Z-measurement on a random system qubit, else a
    random 2-qubit Clifford on a random distinct pair. One sweep is N
    operations. Only the system is touched; S_R is read off R.
    """
    rng = np.random.default_rng(seed)
    sim = stim.TableauSimulator(seed=seed)
    sim.set_num_qubits(2 * N)
    for i in range(N):
        sim.h(N + i)
        sim.cnot(N + i, i)

    ref = list(range(N, 2 * N))
    want = {t: i for i, t in enumerate(times)}
    out = np.full(len(times), np.nan)

    for step in range(1, t_max + 1):
        for _ in range(N):
            if rng.random() < r:
                sim.measure(int(rng.integers(0, N)))
            else:
                a = int(rng.integers(0, N))
                b = int(rng.integers(0, N - 1))
                if b >= a:                 # ensures b != a
                    b += 1
                sim.do_tableau(stim.Tableau.random(2), [a, b])
        if step in want:
            out[want[step]] = stabiliser_entropy_region(sim, 2 * N, ref)
    return out


# =====================================
#  Cell
# =====================================

def main():
    t_max = DEPTH * N
    times = sample_times(t_max)
    t0 = time.time()
    sr = np.stack([run(N, R, times, t_max, m) for m in range(M)])
    elapsed = time.time() - t0

    violations = int((np.diff(sr, axis=1) > 0).sum())
    assert violations == 0, (
        "S_R increased: the reference entropy is outcome-independent "
        "and monotone non-increasing, so this is a bug, not physics")
    assert 0 <= sr.min() and sr.max() <= N, "S_R left [0, N]"

    per_traj = sr[:, -5:].mean(1) / N
    se = per_traj.std(ddof=1) / np.sqrt(M)

    print(f"N = {N}, r = {R}, {M} trajectories, t_max = {t_max} sweeps")
    print(f"s({N}, {R}) = {per_traj.mean():.5f} +- {se:.5f}")
    print(f"purified within horizon: {int((sr[:, -1] == 0).sum())}/{M}")
    print(f"monotonicity violations: {violations}")
    print(f"elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
