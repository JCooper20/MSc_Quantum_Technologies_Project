"""
Monitored all-to-all Clifford circuit with a purification probe.
Stim stabiliser-tableau backend.

WHAT THIS SIMULATES
-------------------
The all-to-all measurement-induced phase transition (MIPT) of
Nahum, Roy, Skinner & Ruhman, PRX Quantum 2, 010352 (2021).

There is no spatial structure -- it is a "bag" of N qubits where any
qubit can interact with any other. The dynamics is built from single
ELEMENTARY OPERATIONS, applied one at a time:

    with probability r      -> measure one random system qubit (Z basis)
    with probability (1 - r) -> apply a random 2-qubit Clifford to one
                                random pair of distinct system qubits

One TIME STEP is defined as N such elementary operations (so each qubit
is "touched" about once per step on average). The measurement rate r in
[0, 1] is the only knob: small r = few measurements (entangling phase),
large r = many measurements (disentangling phase).

`forced=True` switches to forced measurements (always project onto |0>),
which is the FMPT variant; the default is the Born-rule MPT.

This is NOT the same as the layered fully-connected circuit in
fully_connected.py. There, every qubit is paired up once per layer (L/2
gates at once). Here, only ONE pair (or one measurement) happens per
elementary op, and time is counted in units of N ops -- the convention
Nahum uses.

HOW WE MEASURE THE PHASE (the purification probe)
-------------------------------------------------
All-to-all circuits have no "left half vs right half", so the usual
spatial half-chain entropy is meaningless. Instead we use the
Gullans-Huse purification idea:

    1. Add a REFERENCE register R of N extra qubits and Bell-pair each
       reference qubit to one system qubit. This makes the system start
       MAXIMALLY MIXED, and the reference entropy is S_R = N bits.
    2. Run the circuit on the SYSTEM qubits only -- the reference is
       never touched again.
    3. Track S_R, the entropy of the reference register, over time.

S_R measures how much of the initial information the system still holds:

    entangling phase (small r): S_R/N stays high for an exponentially
        long time (a plateau) -- the system "remembers".
    disentangling phase (large r): S_R/N decays quickly to 0 -- the
        measurements purify the system and erase its memory.

This single quantity reproduces both Nahum's operator entanglement and
the Gullans-Huse purification curve.

HOW S_R IS COMPUTED
-------------------
S_R is just an entanglement entropy, so we use the SAME shared routine
as every other circuit in this project: stabiliser_entropy_region from
src/analysis/entropy.py. The only difference from the 1D/2D circuits is
WHICH qubits we take the entropy of:

    1D/2D/FC : a spatial region of the system (e.g. the first L/2 qubits)
    all-to-all: the reference register R = {N, ..., 2N-1}

The formula underneath is identical in all cases:
    S(A) = rank_GF(2)(M_B) - |B|,   B = complement of A.

(Because the full 2N-qubit state stays pure, S(reference) = S(system);
we take the reference here as it reads most directly as "memory of the
initial state".)

Author: Joseph Cooper, MSc Quantum Technologies, UCL.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import stim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.entropy import stabiliser_entropy_region


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class Config:
    """All sweep parameters live here. Edit before calling main()."""

    # ---- System sizes. N counts SYSTEM qubits only; the reference
    #      register doubles this, so the tableau holds 2 * N qubits. ----
    N_values: List[int] = field(default_factory=lambda:
        [8, 16, 32, 64, 128, 256, 512])

    # ---- Measurement rates r to sweep. Denser near the expected
    #      transition. Nahum's Haar values are r_c ~ 0.749 (quantum) and
    #      0.8 (classical); the Clifford value differs, so scan broadly.
    r_values: List[float] = field(default_factory=lambda: [
        0.10, 0.20, 0.30, 0.40, 0.50,
        0.55, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70,
        0.72, 0.74, 0.76, 0.78, 0.80,
        0.85, 0.90,
    ])

    # ---- How many trajectories to average per (N, r). Fewer at large N
    #      because each trajectory is more expensive. -------------------
    n_traj_small:  int = 200   # N <= 32
    n_traj_medium: int = 80    # N <= 128
    n_traj_large:  int = 30    # N <= 256
    n_traj_xl:     int = 12    # N = 512

    # ---- How long to run: t_max = t_max_factor * N time steps. In the
    #      entangling phase the plateau lasts exponentially long, so at
    #      large N the system will NOT fully purify within the window --
    #      that is expected physics, not a bug. ------------------------
    t_max_factor: int = 4

    # ---- When to record S_R: a mix of log- and linearly-spaced steps,
    #      so both the early decay and the late plateau are sampled. ----
    n_log_times: int = 14
    n_lin_times: int = 10

    # ---- Born-rule measurements (MPT) by default; True = forced (FMPT).
    forced: bool = False

    # ---- Fixed seed so runs are reproducible. ------------------------
    seed: int = 12345

    # ---- Output ------------------------------------------------------
    outdir: str = "outputs"
    save_raw: bool = False       # also dump per-trajectory arrays if True
    heatmap_N_max: int = 256     # largest N to include in the (r,t) heatmap

    def n_traj_for(self, N: int) -> int:
        """Pick the trajectory budget for a given system size."""
        if N <= 32:  return self.n_traj_small
        if N <= 128: return self.n_traj_medium
        if N <= 256: return self.n_traj_large
        return self.n_traj_xl


# =====================================================================
# Step 1: prepare the initial state (system + reference)
# =====================================================================

def build_initial_bell_state(N: int) -> stim.TableauSimulator:
    """
    Create the starting state: N Bell pairs linking each system qubit to
    its reference partner.

    Qubit layout in the tableau:
        system    S = {0,    ..., N-1}
        reference R = {N,    ..., 2N-1}

    For each i we apply H to R_i then CNOT(R_i -> S_i), giving the Bell
    pair (|00> + |11>)/sqrt(2). Once all N pairs are made:
        - the system on its own is maximally mixed,
        - the reference entropy is exactly S_R = N bits.

    This is the maximally-mixed starting point whose purification we
    track.
    """
    sim = stim.TableauSimulator()
    sim.set_num_qubits(2 * N)
    for i in range(N):
        sim.h(N + i)
        sim.cnot(N + i, i)
    return sim


# =====================================================================
# Step 2: choose when to record S_R
# =====================================================================

def make_sample_times(t_max: int, n_log: int, n_lin: int) -> List[int]:
    """
    Build the list of time steps at which to record S_R.

    Combines logarithmically-spaced steps (to resolve the fast early-time
    behaviour) with linearly-spaced steps (to cover the long plateau),
    then de-duplicates and clips to [1, t_max].
    """
    log_t = np.geomspace(1, t_max, max(n_log, 2))
    lin_t = np.linspace(1, t_max,  max(n_lin, 2))
    times = np.unique(np.round(np.concatenate([log_t, lin_t])).astype(int))
    times = times[(times >= 1) & (times <= t_max)]
    return [int(t) for t in times]


# =====================================================================
# Step 3: run one trajectory of the circuit
# =====================================================================

def run_single_trajectory(N: int, r: float, sample_times: List[int],
                          t_max: int, seed: int,
                          forced: bool = False) -> np.ndarray:
    """
    Run one full trajectory and return S_R at each sample time.

    The reference qubits R = {N, ..., 2N-1} are prepared once and then
    left alone -- only the system qubits S = {0, ..., N-1} get gates and
    measurements. At each requested time step we read off S_R as the
    entanglement entropy of the reference register.

    Parameters:
    N            = number of system qubits
    r            = measurement rate in [0, 1]
    sample_times = time steps at which to record S_R
    t_max        = total number of time steps to run
    seed         = RNG seed (makes this trajectory reproducible)
    forced       = if True, force measurements onto |0> (FMPT)

    Returns:
    out = array of S_R values (in bits), one per entry of sample_times.
    """
    rng = np.random.default_rng(seed)
    sim = build_initial_bell_state(N)

    reference = list(range(N, 2 * N))     # the qubits whose entropy = S_R
    sample_set = set(sample_times)
    idx_of = {t: i for i, t in enumerate(sample_times)}
    out = np.full(len(sample_times), np.nan, dtype=float)

    # Each time step is N elementary operations on the system.
    for step in range(1, t_max + 1):
        for _ in range(N):
            if rng.random() < r:
                # --- measurement on a random system qubit ---
                q = int(rng.integers(0, N))
                if forced:
                    # FMPT: force the outcome to |0>. Measure (Born), then
                    # flip if it came out 1. For stabiliser states this
                    # gives the same density matrix as projecting onto |0>
                    # and never fails (unlike true postselection, which
                    # would break when the |0> amplitude is exactly zero).
                    result = sim.measure(q)
                    if result:
                        sim.x(q)
                else:
                    sim.measure(q)
            else:
                # --- random 2-qubit Clifford on a random distinct pair ---
                a = int(rng.integers(0, N))
                b = int(rng.integers(0, N - 1))
                if b >= a:           # ensure b != a (pick a distinct partner)
                    b += 1
                sim.do_tableau(stim.Tableau.random(2), [a, b])

        # Record S_R = entropy of the reference register at this step.
        if step in sample_set:
            out[idx_of[step]] = stabiliser_entropy_region(
                sim, 2 * N, reference)

    return out


# =====================================================================
# Sanity checks (run before any sweep)
# =====================================================================

def validate(verbose: bool = True) -> None:
    """
    Quick correctness checks on the entropy and the dynamics. These catch
    the common ways the setup can be wrong before spending hours on a
    sweep.
    """
    if verbose:
        print("Validation:")

    # 1. One Bell pair -> reference entropy is exactly 1 bit.
    sim = build_initial_bell_state(1)
    S = stabiliser_entropy_region(sim, 2, [1])
    assert abs(S - 1.0) < 1e-9, f"Bell pair: expected 1, got {S}"
    if verbose:
        print(f"  [PASS] N=1 Bell pair         S_R = {S:.6f}   (expected 1)")

    # 2. Unentangled product state -> entropy 0.
    sim = stim.TableauSimulator(); sim.set_num_qubits(2)
    S = stabiliser_entropy_region(sim, 2, [1])
    assert abs(S) < 1e-9, f"Product state: expected 0, got {S}"
    if verbose:
        print(f"  [PASS] product state         S_R = {S:.6f}   (expected 0)")

    # 3. N=4 Bell-paired register -> S_R = 4 bits (maximally mixed system).
    sim = build_initial_bell_state(4)
    S = stabiliser_entropy_region(sim, 8, list(range(4, 8)))
    assert abs(S - 4.0) < 1e-9, f"N=4 Bell: expected 4, got {S}"
    if verbose:
        print(f"  [PASS] N=4 Bell pairs        S_R = {S:.6f}   (expected 4)")

    # 4. GHZ on 4 qubits -> any single-qubit cut has entropy 1.
    sim = stim.TableauSimulator(); sim.set_num_qubits(4)
    sim.h(0); sim.cnot(0, 1); sim.cnot(0, 2); sim.cnot(0, 3)
    S = stabiliser_entropy_region(sim, 4, [0])
    assert abs(S - 1.0) < 1e-9, f"GHZ_4 single cut: expected 1, got {S}"
    if verbose:
        print(f"  [PASS] GHZ_4 1-qubit cut     S    = {S:.6f}   (expected 1)")

    # 5. r=0 (no measurements) -> nothing purifies, S_R stays at N.
    out = run_single_trajectory(N=8, r=0.0, sample_times=[1, 4, 8],
                                t_max=8, seed=1)
    assert np.allclose(out, 8.0, atol=1e-9), \
        f"r=0: S_R should stay = 8, got {out}"
    if verbose:
        print(f"  [PASS] r=0  (N=8)            S_R(t) = {out.tolist()}  (all 8)")

    # 6. r=1 (measure constantly) -> system purifies fast, S_R -> 0.
    out = run_single_trajectory(N=8, r=1.0, sample_times=[2, 4, 8],
                                t_max=8, seed=1)
    assert out[-1] < 1.0, f"r=1: S_R should decay to ~0, got {out[-1]}"
    if verbose:
        print(f"  [PASS] r=1  (N=8)            S_R(t) = {out.tolist()}  (-> 0)")

    if verbose:
        print("  All validation tests passed.\n")


# =====================================================================
# Sweep over (N, r) with on-disk checkpointing
# =====================================================================

def _key_str(N: int, r: float) -> str:
    """Stable string key for one (N, r) cell, used in the checkpoint."""
    return f"N{N}_r{r:.4f}"


def _save_json(path: str, obj: dict) -> None:
    """Write JSON atomically (write to .tmp, then rename) so a crash
    mid-write cannot corrupt the checkpoint."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _load_json_if_exists(path: str) -> dict:
    """Load a checkpoint if present, else return an empty dict."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def sweep(cfg: Config) -> Dict[Tuple[int, float], dict]:
    """
    Run the full (N, r) grid. For each cell, average S_R(t) over
    cfg.n_traj_for(N) trajectories and store the mean and standard error.

    The checkpoint is rewritten after every cell, and already-completed
    cells are skipped on restart, so a long N=512 run can be stopped and
    resumed safely.

    Returns a dict: (N, r) -> {'times', 'mean', 'sem', 'n_traj'}.
    """
    os.makedirs(cfg.outdir, exist_ok=True)
    ckpt_path = os.path.join(cfg.outdir, "checkpoint.json")
    raw_dir   = os.path.join(cfg.outdir, "raw")
    if cfg.save_raw:
        os.makedirs(raw_dir, exist_ok=True)

    # Reload any previously-completed cells.
    stored = _load_json_if_exists(ckpt_path)
    if stored:
        print(f"Resumed checkpoint with {len(stored)} cells from {ckpt_path}")

    results: Dict[Tuple[int, float], dict] = {}
    for skey, cell in stored.items():
        N_s, r_s = skey.split("_")
        results[(int(N_s[1:]), float(r_s[1:]))] = cell

    for N in cfg.N_values:
        t_max = cfg.t_max_factor * N
        sample_times = make_sample_times(t_max, cfg.n_log_times, cfg.n_lin_times)
        n_traj = cfg.n_traj_for(N)
        for r in cfg.r_values:
            r = float(r)
            key = (N, r)
            if key in results:          # already done -> skip
                continue
            t0 = time.time()

            # Run n_traj independent trajectories for this (N, r).
            traj = np.full((n_traj, len(sample_times)), np.nan, dtype=float)
            for j in range(n_traj):
                # Distinct, reproducible seed per (N, r, trajectory).
                seed = (cfg.seed
                        + 1009 * N
                        + 7919 * int(round(r * 10000))
                        + j)
                traj[j] = run_single_trajectory(
                    N=N, r=r, sample_times=sample_times,
                    t_max=t_max, seed=seed, forced=cfg.forced,
                )

            # Aggregate: mean and standard error across trajectories.
            mean = traj.mean(0)
            sem  = (traj.std(0, ddof=1) / np.sqrt(n_traj)
                    if n_traj > 1 else np.zeros_like(mean))
            results[key] = {
                "times":  sample_times,
                "mean":   mean.tolist(),
                "sem":    sem.tolist(),
                "n_traj": n_traj,
            }
            if cfg.save_raw:
                np.save(os.path.join(raw_dir,
                                      f"raw_N{N}_r{r:.4f}.npy"), traj)

            # Checkpoint after every cell.
            _save_json(ckpt_path,
                       {_key_str(*k): v for k, v in results.items()})
            elapsed = time.time() - t0
            print(f"[done]  N={N:>4}  r={r:.3f}  n_traj={n_traj:>3}  "
                  f"S_R(final)/N = {mean[-1] / N:.4f}   ({elapsed:.1f}s)")

    return results


# =====================================================================
# Figures
# =====================================================================

def _save_fig(fig, outdir: str, stem: str) -> None:
    """Save a figure as both PNG and PDF, then close it."""
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"), dpi=200)
    plt.close(fig)


def make_plots(cfg: Config, results: Dict[Tuple[int, float], dict]) -> None:
    """
    Produce the standard figures:
        fig1 : S_R/N vs t at small r  -> the entangling-phase plateau
        fig2 : S_R/N vs t at large r  -> the disentangling-phase decay
        fig3 : late-time S_R/N vs r   -> the phase boundary / crossing
        fig4 : (optional) S_R/N heatmap over (r, t) at the largest N
    """
    outdir = cfg.outdir
    os.makedirs(outdir, exist_ok=True)

    # Use the smallest / largest r actually present in the results.
    r_done = sorted({r for (_, r) in results})
    r_small = min(r_done)
    r_large = max(r_done)

    # ---- Figs 1 & 2: S_R/N vs t at fixed r, one curve per N ----------
    def fig_time_series(r_target, stem, phase_label):
        fig, ax = plt.subplots(figsize=(8, 5))
        for N in cfg.N_values:
            key = (N, r_target)
            if key not in results:
                continue
            d = results[key]
            t = np.asarray(d["times"], dtype=float)
            m = np.asarray(d["mean"]) / N      # entropy per system qubit
            s = np.asarray(d["sem"]) / N
            ax.plot(t, m, "-o", ms=3, lw=1.4, label=f"$N={N}$")
            ax.fill_between(t, m - s, m + s, alpha=0.2)
        ax.set_xlabel(r"time step $t$  (units of $N$ elementary ops)")
        ax.set_ylabel(r"$S_R(t) / N$   (bits per system qubit)")
        ax.set_title(fr"All-to-all Clifford MIPT — {phase_label}  "
                     fr"($r={r_target:.2f}$)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="best")
        _save_fig(fig, outdir, stem)

    fig_time_series(r_small, "fig1_plateau_smallr",
                    "entangling-phase plateau")
    fig_time_series(r_large, "fig2_decay_larger",
                    "disentangling-phase decay")

    # ---- Fig 3: late-time S_R/N vs r, one curve per N ----------------
    #      Curves fan out (ordered by N) in the entangling phase and
    #      collapse to 0 in the disentangling phase; the separation marks
    #      the transition.
    fig, ax = plt.subplots(figsize=(8, 5))
    for N in cfg.N_values:
        rs, ms, ss = [], [], []
        for r in cfg.r_values:
            r = float(r)
            key = (N, r)
            if key not in results:
                continue
            d = results[key]
            rs.append(r)
            ms.append(d["mean"][-1] / N)   # last sampled time
            ss.append(d["sem"][-1] / N)
        if not rs:
            continue
        rs = np.array(rs); ms = np.array(ms); ss = np.array(ss)
        ax.errorbar(rs, ms, yerr=ss, marker="o", ms=4, lw=1.4,
                    capsize=2, label=f"$N={N}$")
    ax.set_xlabel(r"measurement rate $r$")
    ax.set_ylabel(r"$S_R(t_{\rm late}) / N$")
    ax.set_title("All-to-all Clifford MIPT — late-time $S_R/N$ vs $r$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="best")
    _save_fig(fig, outdir, "fig3_late_vs_r")

    # ---- Fig 4 (optional): S_R/N heatmap over (r, t) at the largest N -
    feasible_N = [N for N in cfg.N_values if N <= cfg.heatmap_N_max
                  and any((N, float(r)) in results for r in cfg.r_values)]
    if feasible_N:
        N = max(feasible_N)
        rs_done = sorted([r for r in cfg.r_values
                          if (N, float(r)) in results])
        if rs_done:
            times = results[(N, float(rs_done[0]))]["times"]
            Z = np.full((len(rs_done), len(times)), np.nan)
            for i, r in enumerate(rs_done):
                Z[i] = np.asarray(results[(N, float(r))]["mean"]) / N
            fig, ax = plt.subplots(figsize=(9, 5))
            im = ax.imshow(Z, aspect="auto", origin="lower",
                           extent=[times[0], times[-1],
                                   rs_done[0], rs_done[-1]],
                           cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xlabel(r"time step $t$")
            ax.set_ylabel(r"measurement rate $r$")
            ax.set_title(fr"$S_R(t) / N$ heatmap, $N={N}$")
            fig.colorbar(im, ax=ax, label=r"$S_R / N$")
            _save_fig(fig, outdir, f"fig4_heatmap_N{N}")


# =====================================================================
# Main entry point
# =====================================================================

def main(cfg: Config | None = None) -> None:
    """Validate, run the sweep, and produce the figures."""
    cfg = cfg or Config()
    os.makedirs(cfg.outdir, exist_ok=True)
    print("=" * 70)
    print("All-to-all monitored Clifford MIPT  (purification probe, Stim)")
    print("=" * 70)
    print(f"output dir : {cfg.outdir}")
    print(f"N values   : {cfg.N_values}")
    print(f"r values   : {cfg.r_values}")
    print(f"t_max      : {cfg.t_max_factor} * N (time steps; each = N ops)")
    print(f"forced     : {cfg.forced}")
    print(f"seed       : {cfg.seed}\n")

    validate(verbose=True)

    print("Sweep:")
    t0 = time.time()
    results = sweep(cfg)
    print(f"\nSweep complete: {len(results)} (N, r) cells in "
          f"{(time.time() - t0) / 60:.1f} min.")

    print("Plotting:")
    make_plots(cfg, results)
    print(f"Figures written to {cfg.outdir}/")


if __name__ == "__main__":
    main()
