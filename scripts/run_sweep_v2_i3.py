"""
All-to-all monitored Clifford sweep, v2: per-trajectory storage + I3.

Reproduces the published protocol exactly and adds (a) full
per-trajectory storage for bootstrapping and (b) the tripartite mutual
information I3 as a second, threshold-free r_c locator.

PROTOCOL (verified against outputs_full/checkpoint.json, NOT the
later-edited sweep-script constants): t_max = 4N, sample_times =
make_sample_times(4N, n_log=14, n_lin=10). The verification harness
re-derives (n_log, n_lin) from the checkpoint's stored times for every
N before this script is trusted.

TRAJECTORY PATHWAY: the fully seed-deterministic variant (numpy-seeded
structure, Cliffords numpy-indexed into the cached 2-qubit Clifford
group, seeded TableauSimulator) — statistically equivalent to the
published pathway, as established by
scripts/diagnostics/verify_physics_equivalence.py (pathway means agree
with each other and with the checkpoint within error). S_R itself is
computed by the UNCHANGED entropy.py:stabiliser_entropy_region — the
same code that produced the published curves. I3 uses the fast exact
path in src/analysis/entropy_fast.py (verified bit-identical rank).
NOTE: each sample time therefore extracts canonical_stabilizers()
twice (once inside the untouched published S_R path, once for I3's
shared seven-entropy extraction); accepted cost of leaving the
published path byte-identical.

STORAGE: one compressed npz per (N, r) cell:
    sr        (n_traj, n_times)  raw S_R in bits (NOT divided by N,
                                 matching the existing checkpoint)
    i3        (n_traj, n_times)
    seeds     (n_traj,)
    meas_counts (n_traj,) realised measurement counts
    times     (n_times,)
plus manifest.json (protocol, grid, counts, code note) and an
aggregate checkpoint_v2.json in the exact format of the existing
checkpoint ({"N{N}_r{r:.4f}": {times, mean, sem, n_traj}}, mean/sem of
RAW S_R) so downstream analysis scripts work unchanged.

Cells are written atomically (tmp + rename) and skipped if present:
a crash loses at most one cell; restart resumes.

SANITY GUARD: S_R must be monotonically non-increasing along a
trajectory (references are never acted on). Violations are counted and
logged as warnings — they indicate a bug, not physics. I3 has no such
guarantee and trajectories are never truncated early.

SEEDS: seed(N, r, j) = (N * 1000 + round(100 r)) * 100000 + j —
unique across the grid, fully reproducible (within a stim version;
record pip freeze alongside the outputs).

Usage (from the project root):
    PYTHONPATH=. python scripts/run_sweep_v2_i3.py \
        [--workers 22] [--sizes 64 128 256 512] [--n-traj 300] \
        [--rs 0.30 0.50 0.01] [--outdir outputs_v2]
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
from src.analysis.entropy_fast import tripartite_I3

N_LOG, N_LIN, T_MAX_FACTOR = 14, 10, 4          # verified protocol

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
    """One trajectory: (S_R(t), I3(t), meas_count, seed, n_warn)."""
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
    i3 = np.full(len(times), np.nan)
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
            i3[k] = tripartite_I3(sim, N)
    n_warn = int(np.sum(np.diff(sr) > 1e-9))      # S_R must not rise
    return sr, i3, meas_count, seed, n_warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[64, 128, 256, 512])
    ap.add_argument("--n-traj", type=int, default=300)
    ap.add_argument("--rs", type=float, nargs=3,
                    default=[0.30, 0.50, 0.01],
                    metavar=("START", "STOP", "STEP"))
    ap.add_argument("--outdir", default="outputs_v2")
    args = ap.parse_args()

    r_values = [round(r, 4) for r in np.arange(
        args.rs[0], args.rs[1] + 1e-9, args.rs[2])]
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "raw"), exist_ok=True)

    manifest = {
        "protocol": {"t_max": "4*N", "n_log": N_LOG, "n_lin": N_LIN,
                     "verified_against": "outputs_full/checkpoint.json"},
        "grid": {"sizes": args.sizes, "r_values": r_values,
                 "n_traj": args.n_traj},
        "pathway": "seed-deterministic (numpy-indexed cliffords, "
                   "seeded TableauSimulator); S_R via unchanged "
                   "stabiliser_entropy_region; I3 via entropy_fast",
        "seed_formula": "(N*1000 + round(100 r)) * 100000 + j",
        "sr_units": "raw bits (matching existing checkpoint)",
    }
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    ckpt_path = os.path.join(args.outdir, "checkpoint_v2.json")
    ckpt = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) \
        else {}

    with Pool(args.workers) as pool:
        for N in args.sizes:
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
                        for j in range(args.n_traj)]
                sr = np.zeros((args.n_traj, len(times)))
                i3 = np.zeros((args.n_traj, len(times)))
                seeds = np.zeros(args.n_traj, dtype=np.int64)
                mcounts = np.zeros(args.n_traj, dtype=np.int64)
                warns = 0
                for k, (s, t3, mc, sd, nw) in enumerate(
                        pool.imap(run_one, jobs, chunksize=1)):
                    sr[k], i3[k] = s, t3
                    seeds[k], mcounts[k] = sd, mc
                    warns += nw
                if warns:
                    print(f"[WARNING] {key}: {warns} S_R increases "
                          "detected — investigate before trusting",
                          flush=True)
                tmp = path + ".tmp.npz"
                np.savez_compressed(tmp, sr=sr, i3=i3, seeds=seeds,
                                    meas_counts=mcounts,
                                    times=np.array(times))
                os.replace(tmp, path)
                ckpt[key] = {
                    "times": [int(t) for t in times],
                    "mean": sr.mean(axis=0).tolist(),
                    "sem": (sr.std(axis=0, ddof=1)
                            / np.sqrt(args.n_traj)).tolist(),
                    "n_traj": args.n_traj,
                    "i3_mean": i3.mean(axis=0).tolist(),
                    "i3_sem": (i3.std(axis=0, ddof=1)
                               / np.sqrt(args.n_traj)).tolist(),
                }
                with open(ckpt_path + ".tmp", "w") as f:
                    json.dump(ckpt, f)
                os.replace(ckpt_path + ".tmp", ckpt_path)
                dt = time.time() - t0
                print(f"[done] {key}: {dt:.0f}s "
                      f"({dt/args.n_traj:.2f}s/traj on "
                      f"{args.workers} workers)", flush=True)


if __name__ == "__main__":
    main()
