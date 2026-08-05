"""
Layer-1 verification gate for the graph-state engine + adapter
(src/graphstate/). Produces the evidence reported in VERIFICATION.md.

  A. S_R(t) match vs the stim pathway across >= 20 independent
     trajectories: N in {8, 16, 32} x r in {0.2, 0.35, 0.5} x seeds,
     identical circuits by construction (same numpy draw stream),
     every sample time compared against the published
     stabiliser_entropy_region.
  B. Full state equality at small N (canonical stabilizers, signs
     included) at every sample time, with measurement outcomes
     synchronised via postselect (outcomes are random and independent
     between engines; S_R is outcome-independent, but state equality
     is not).
  C. Physical sanity: S_R starts at N, monotone non-increasing,
     non-negative, reaches 0 deep in the purifying phase.
  D. LC non-uniqueness demonstration: apply a local complementation
     to a recorded graph; show the edge set changes while both S_R
     (cut-rank) and the reconstructed stabilizer state are unchanged.
  E. Timing at N = 8, 16, 32, 64 and stored npz size.

Run from the project root:
    PYTHONPATH=. python scripts/verify_graphstate.py
"""

# Imports
import os
import time

import numpy as np
import stim

from src.simulators.all_to_all_stim import make_sample_times
from src.analysis.entropy import stabiliser_entropy_region
from src.analysis.entropy_fast import gf2_rank_bitpacked
from src.graphstate.adapter import (GraphStateTrajectory,
                                    run_trajectory, save_trajectory,
                                    two_qubit_cliffords)


def stim_reference_sr(N, r, sample_times, t_max, seed):
    """The verified stim pathway, identical draw stream."""
    rng = np.random.default_rng(seed)
    cl = two_qubit_cliffords()
    sim = stim.TableauSimulator(seed=seed)
    sim.set_num_qubits(2 * N)
    for i in range(N):
        sim.h(N + i)
        sim.cnot(N + i, i)
    out = np.full(len(sample_times), np.nan)
    idx_of = {t: i for i, t in enumerate(sample_times)}
    for step in range(1, t_max + 1):
        for _ in range(N):
            if rng.random() < r:
                sim.measure(int(rng.integers(0, N)))
            else:
                a = int(rng.integers(0, N))
                b = int(rng.integers(0, N - 1))
                if b >= a:
                    b += 1
                c = int(rng.integers(0, len(cl)))
                sim.do_tableau(cl[c], [a, b])
        if step in idx_of:
            out[idx_of[step]] = stabiliser_entropy_region(
                sim, 2 * N, list(range(N, 2 * N)))
    return out


def main():
    print("=" * 66)
    print("A. S_R(t) match vs stim (identical circuits by construction)")
    print("=" * 66)
    n_traj = 0
    n_match = 0
    for N in (8, 16, 32):
        t_max = 4 * N
        times = make_sample_times(t_max, 14, 10)
        for r in (0.2, 0.35, 0.5):
            for seed in (1, 2, 3):
                gs = run_trajectory(N, r, times, t_max, seed)
                st = stim_reference_sr(N, r, times, t_max, seed)
                ok = np.array_equal(gs["sr"], st)
                n_traj += 1
                n_match += int(ok)
                if not ok:
                    k = int(np.argmax(gs["sr"] != st))
                    print(f"  MISMATCH N={N} r={r} seed={seed} at "
                          f"t={times[k]}: graph {gs['sr'][k]} vs "
                          f"stim {st[k]}")
    print(f"  {n_match}/{n_traj} trajectories: S_R(t) identical at "
          "every sample time")
    assert n_match == n_traj, "S_R GATE FAILED"

    print("=" * 66)
    print("B. full state equality (outcome-synchronised, small N)")
    print("=" * 66)
    cl = two_qubit_cliffords()
    checks = 0
    for N, r, seed in [(4, 0.35, 7), (6, 0.2, 3), (6, 0.5, 21),
                       (8, 0.35, 5)]:
        rng = np.random.default_rng(seed)
        traj = GraphStateTrajectory(N, seed=seed)
        sim = stim.TableauSimulator(seed=seed)
        sim.set_num_qubits(2 * N)
        for i in range(N):
            sim.h(N + i)
            sim.cnot(N + i, i)
        for step in range(1, 8 * N + 1):
            if rng.random() < r:
                q = int(rng.integers(0, N))
                m = traj.measure_z(q)
                sim.postselect_z(q, desired_value=bool(m))
            else:
                a = int(rng.integers(0, N))
                b = int(rng.integers(0, N - 1))
                if b >= a:
                    b += 1
                c = int(rng.integers(0, len(cl)))
                traj.apply_clifford(c, a, b)
                sim.do_tableau(cl[c], [a, b])
            assert traj.to_stim().canonical_stabilizers() == \
                sim.canonical_stabilizers(), (N, r, seed, step)
            checks += 1
    print(f"  {checks} operations verified state-identical "
          "(canonical stabilizers incl. signs)")

    print("=" * 66)
    print("C. physical sanity")
    print("=" * 66)
    for N, r in [(16, 0.2), (16, 0.5), (32, 0.5)]:
        t_max = 4 * N
        times = make_sample_times(t_max, 14, 10)
        gs = run_trajectory(N, r, times, t_max, seed=13)
        sr = gs["sr"]
        assert (np.diff(sr) <= 1e-9).all(), "S_R increased!"
        assert (sr >= 0).all(), "negative S_R!"
        note = ""
        if r == 0.5:
            assert sr[-1] == 0, "high-r trajectory failed to purify"
            note = " -> reaches 0 (purifying)"
        print(f"  N={N} r={r}: S_R[0]={sr[0]:.0f} (starts near N), "
              f"monotone, non-negative, final={sr[-1]:.0f}{note}")

    print("=" * 66)
    print("D. local-complementation non-uniqueness (concrete)")
    print("=" * 66)
    N = 8
    t_max = 16
    times = [8, 16]
    gs = run_trajectory(N, 0.3, times, t_max, seed=4)
    traj = gs["traj"]
    edges_before, vops_before = traj.snapshot()
    sr_before = traj.s_r()
    state_before = traj.to_stim().canonical_stabilizers()
    # complement at a vertex that has neighbours
    target = int(edges_before[0][0])
    traj.g.local_complementation(target)
    edges_after, vops_after = traj.snapshot()
    sr_after = traj.s_r()
    state_after = traj.to_stim().canonical_stabilizers()
    print(f"  LC at vertex {target}:")
    print(f"    edges before ({len(edges_before)}): "
          f"{[tuple(e) for e in edges_before]}")
    print(f"    edges after  ({len(edges_after)}): "
          f"{[tuple(e) for e in edges_after]}")
    print(f"    graphs differ: "
          f"{not np.array_equal(edges_before, edges_after)}")
    print(f"    S_R unchanged: {sr_before == sr_after} "
          f"({sr_before} -> {sr_after})")
    print(f"    physical state unchanged: "
          f"{state_before == state_after}")
    assert not np.array_equal(edges_before, edges_after)
    assert sr_before == sr_after and state_before == state_after

    print("=" * 66)
    print("E. timing and storage")
    print("=" * 66)
    outdir = "/tmp/graphstate_verify"
    os.makedirs(outdir, exist_ok=True)
    for N in (8, 16, 32, 64):
        t_max = 4 * N
        times = make_sample_times(t_max, 14, 10)
        t0 = time.time()
        gs = run_trajectory(N, 0.35, times, t_max, seed=17)
        dt = time.time() - t0
        path = os.path.join(outdir, f"traj_N{N}.npz")
        save_trajectory(path, gs)
        kb = os.path.getsize(path) / 1024
        # reload check
        z = np.load(path)
        assert np.array_equal(z["sr"], gs["sr"])
        print(f"  N={N:3d}: {dt:6.2f} s/trajectory "
              f"({len(times)} samples), stored npz {kb:7.1f} KB, "
              "reloads OK")

    print("\nALL LAYER-1 VERIFICATION GATES PASSED")


if __name__ == "__main__":
    main()
