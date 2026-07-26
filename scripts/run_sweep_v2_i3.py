"""
All-to-all monitored Clifford sweep at the 12N converged protocol,
with per-trajectory storage. S_R only — no I3 in this sweep.

Derived from scripts/run_sweep_v2_i3.py with exactly these changes:
protocol constants (12N, n_log=30, n_lin=150), I3 removed everywhere,
per-size dynamic trajectory counts, manifest updated to match, and
default outdir outputs_12N (the 12N arrays have ~172-176 sample times
and must never collide with the 22-sample v2 cells).

TRAJECTORY PATHWAY: the fully seed-deterministic variant (numpy-seeded
structure, Cliffords numpy-indexed into the cached 2-qubit Clifford
group, seeded TableauSimulator) — statistically equivalent to the
published pathway (scripts/diagnostics/verify_physics_equivalence.py).
S_R is computed by the UNCHANGED entropy.py:stabiliser_entropy_region,
the same code that produced the published curves.

STORAGE: one compressed npz per (N, r) cell:
    sr          (n_traj, n_times)  raw S_R in bits
    seeds       (n_traj,)
    meas_counts (n_traj,)
    times       (n_times,)
plus manifest.json and an aggregate checkpoint_12N.json in the legacy
format ({"N{N}_r{r:.4f}": {times, mean, sem, n_traj}}, raw bits).
Cells are written atomically and skipped if present (crash loses at
most one cell; restart resumes).

SANITY GUARD: S_R must be monotonically non-increasing along a
trajectory; violations are counted and logged as warnings.

SEEDS: seed(N, r, j) = (N * 1000 + round(100 r)) * 100000 + j —
unchanged from the v2 sweep. NOTE: unique on the default 0.01 grid;
a 0.005 grid would collide (round(100 r) cannot distinguish e.g.
0.300 from 0.305) — change the formula before using a finer grid.

Usage (from the project root):
    PYTHONPATH=. python scripts/run_sweep_12N.py \
        [--workers 22] [--sizes 64 128 256 512] \
        [--rs 0.30 0.50 0.01] [--outdir outputs_12N]
"""

# Imports
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from src.simulators.all_to_all_stim import make_sample_times
from src.analysis.entropy import stabiliser_entropy_region

N_LOG, N_LIN, T_MAX_FACTOR = 30, 150, 12         # 12N converged protocol

N_TRAJ_BY_SIZE = {64: 300, 128: 150, 256: 100, 512: 100}

_CLIFFORD_CACHE = None


def _two_qubit_cliffords():
    """All 11,520 two-qubit Cliffords, enumerated once per worker.

    Self-contained (the published all_to_all_stim.py does not carry
    this helper). stim.Tableau.iter_all(2) has a fixed deterministic
    order, so indexing into it with a seeded RNG gives exact uniform
    sampling and full reproducibility within a stim version.
    """
    global _CLIFFORD_CACHE
    if _CLIFFORD_CACHE is None:
        import stim
        _CLIFFORD_CACHE = tuple(stim.Tableau.iter_all(2))
    return _CLIFFORD_CACHE


def cell_seed(N, r, j):
    return (N * 1000 + int(round(100 * r))) * 100000 + j


def run_one(args):
    """One trajectory: (S_R(t), meas_count, seed, n_warn)."""
    N, r, seed = args
    import stim                                   # per-worker import
    t_max = T_MAX_FACTOR * N
    times = make_sample_times(t_max, N_LOG, N_LIN)
    rng = np.random.default_rng(seed)
    cliffords = _two_qubit_cliffords()
    sim = stim.TableauSimulator(seed=seed)
    sim.set_num_qubits(2 * N)
    for i in range(N):
        sim.h(N + i)
        sim.cnot(N + i, i)
    reference = list(range(N, 2 * N))
    sample_set = set(times)
    idx_of = {t: i for i, t in enumerate(times)}
    sr = np.full(len(times), np.nan)
    meas_count = 0
    for step in range(1, t_max + 1):
        for _ in range(N):
            if rng.random() < r:
                q = int(rng.integers(0, N))
                sim.measure(q)
                meas_count += 1
            else:
                a = int(rng.integers(0, N))
                b = int(rng.integers(0, N - 1))
                if b >= a:
                    b += 1
                c = int(rng.integers(0, len(cliffords)))
                sim.do_tableau(cliffords[c], [a, b])
        if step in sample_set:
            k = idx_of[step]
            sr[k] = stabiliser_entropy_region(sim, 2 * N, reference)
    n_warn = int(np.sum(np.diff(sr) > 1e-9))      # S_R must not rise
    return sr, meas_count, seed, n_warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[64, 128, 256, 512])
    ap.add_argument("--n-traj", type=int, default=300,
                    help="fallback only; per-size dict takes "
                         "precedence")
    ap.add_argument("--rs", type=float, nargs=3,
                    default=[0.30, 0.50, 0.01],
                    metavar=("START", "STOP", "STEP"))
    ap.add_argument("--outdir", default="outputs_12N")
    args = ap.parse_args()

    r_values = [round(r, 4) for r in np.arange(
        args.rs[0], args.rs[1] + 1e-9, args.rs[2])]
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "raw"), exist_ok=True)

    manifest = {
        "protocol": {"t_max": "12*N", "n_log": N_LOG, "n_lin": N_LIN,
                     "recovered_from":
                         "scripts/run_critical_sweep.py"},
        "grid": {"sizes": args.sizes, "r_values": r_values,
                 "n_traj_by_size": N_TRAJ_BY_SIZE},
        "pathway": "seed-deterministic (numpy-indexed cliffords, "
                   "seeded TableauSimulator); S_R via unchanged "
                   "stabiliser_entropy_region; no I3",
        "seed_formula": "(N*1000 + round(100 r)) * 100000 + j",
        "sr_units": "raw bits (matching existing checkpoint)",
    }
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    ckpt_path = os.path.join(args.outdir, "checkpoint_12N.json")
    ckpt = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) \
        else {}

    with Pool(args.workers) as pool:
        for N in args.sizes:
            n_traj = N_TRAJ_BY_SIZE[N]
            times = make_sample_times(T_MAX_FACTOR * N, N_LOG, N_LIN)
            for r in r_values:
                key = f"N{N}_r{r:.4f}"
                path = os.path.join(args.outdir, "raw",
                                    f"cell_{key}.npz")
                if os.path.exists(path):
                    print(f"[skip] {key} (exists)", flush=True)
                    continue
                t0 = time.time()
                jobs = [(N, r, cell_seed(N, r, j))
                        for j in range(n_traj)]
                sr = np.zeros((n_traj, len(times)))
                seeds = np.zeros(n_traj, dtype=np.int64)
                mcounts = np.zeros(n_traj, dtype=np.int64)
                warns = 0
                for k, (s, mc, sd, nw) in enumerate(
                        pool.imap(run_one, jobs, chunksize=1)):
                    sr[k] = s
                    seeds[k], mcounts[k] = sd, mc
                    warns += nw
                if warns:
                    print(f"[WARNING] {key}: {warns} S_R increases "
                          "detected — investigate before trusting",
                          flush=True)
                tmp = path + ".tmp.npz"
                np.savez_compressed(tmp, sr=sr, seeds=seeds,
                                    meas_counts=mcounts,
                                    times=np.array(times))
                os.replace(tmp, path)
                ckpt[key] = {
                    "times": [int(t) for t in times],
                    "mean": sr.mean(axis=0).tolist(),
                    "sem": (sr.std(axis=0, ddof=1)
                            / np.sqrt(n_traj)).tolist(),
                    "n_traj": n_traj,
                }
                with open(ckpt_path + ".tmp", "w") as f:
                    json.dump(ckpt, f)
                os.replace(ckpt_path + ".tmp", ckpt_path)
                dt = time.time() - t0
                print(f"[done] {key}: {dt:.0f}s "
                      f"({dt/n_traj:.2f}s/traj on "
                      f"{args.workers} workers)", flush=True)


if __name__ == "__main__":
    main()

