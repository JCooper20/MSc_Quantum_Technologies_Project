"""
Graph-state trajectory adapter for the all-to-all monitored Clifford
model. Drives the Anders-Briegel engine through the exact trajectories 
of the verified stim pathway.
"""

# Imports
import random as _pyrandom
import numpy as np
import stim
from src.graphstate.engine import GraphState
from src.graphstate.vop_map import VOP_TABLEAUS
from src.analysis.entropy_fast import gf2_rank_bitpacked

_CLIFFORD_CACHE = None
_DECOMP_CACHE = {}

_GATE_MAP = {
    "H": "h", "S": "s", "S_DAG": "s_dagger", "X": "x", "Y": "y",
    "Z": "z", "CX": "cx", "ZCX": "cx", "CZ": "cz", "ZCZ": "cz",
    "I": "id",
}


def two_qubit_cliffords():
    global _CLIFFORD_CACHE
    if _CLIFFORD_CACHE is None:
        _CLIFFORD_CACHE = tuple(stim.Tableau.iter_all(2))
    return _CLIFFORD_CACHE


def clifford_gate_list(index):
    """
    Decompose 2q Clifford #index into engine-supported gates.
    Returns a list of (method_name, qubit_indices) with qubit indices
    in {0, 1}; raises on any gate outside the mapped set.
    """
    if index not in _DECOMP_CACHE:
        circuit = two_qubit_cliffords()[index].to_circuit(
            method="elimination")
        ops = []
        for inst in circuit:
            name = inst.name
            if name not in _GATE_MAP:
                raise ValueError(
                    f"decomposition produced unsupported gate {name!r}")
            targets = [t.value for t in inst.targets_copy()]
            meth = _GATE_MAP[name]
            if meth in ("cx", "cz"):
                for k in range(0, len(targets), 2):
                    ops.append((meth, (targets[k], targets[k + 1])))
            else:
                for t in targets:
                    ops.append((meth, (t,)))
        _DECOMP_CACHE[index] = ops
    return _DECOMP_CACHE[index]


class GraphStateTrajectory:
    """
    One trajectory of the model in the graph-state representation.
    """

    def __init__(self, N, seed=None):
        self.N = N
        self.g = GraphState(2 * N)
        if seed is not None:
            _pyrandom.seed(seed)     # engine's measurement-outcome RNG
        for i in range(N):
            self.g.h(N + i)
            self.g.cx(N + i, i)

    # dynamics
    def apply_clifford(self, index, a, b):
        qmap = (a, b)
        for meth, targets in clifford_gate_list(index):
            getattr(self.g, meth)(*(qmap[t] for t in targets))

    def measure_z(self, q):
        return self.g.measure(q, basis="Z")

    # readout 
    def adjacency(self):
        n = 2 * self.N
        A = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in self.g.vertices[i].neighbors:
                A[i, j] = True
        return A

    def s_r(self):
        A = self.adjacency()
        return float(gf2_rank_bitpacked(
            np.ascontiguousarray(A[self.N:, :self.N])))

    def snapshot(self):
        edges = sorted(
            (i, j)
            for i in range(2 * self.N)
            for j in self.g.vertices[i].neighbors if i < j)
        vops = np.array([v.vop_code for v in self.g.vertices],
                        dtype=np.uint8)
        return np.array(edges, dtype=np.int32).reshape(-1, 2), vops

    def to_stim(self):
        """
        Reconstruct the represented stabilizer state in stim (for
        verification): |G> from H + CZ per edge, then the VOPs.
        """
        sim = stim.TableauSimulator()
        sim.set_num_qubits(2 * self.N)
        for q in range(2 * self.N):
            sim.h(q)
        for i in range(2 * self.N):
            for j in self.g.vertices[i].neighbors:
                if i < j:
                    sim.cz(i, j)
        for q in range(2 * self.N):
            sim.do_tableau(VOP_TABLEAUS[self.g.vertices[q].vop_code],
                           [q])
        return sim


def run_trajectory(N, r, sample_times, t_max, seed):
    """
    Seed-deterministic trajectory; identical circuit structure to
    the stim pathway for the same seed. Returns a dict with sr,
    snapshots (list of (edges, vops)), meas_count.
    """
    rng = np.random.default_rng(seed)
    n_cliff = len(two_qubit_cliffords())
    traj = GraphStateTrajectory(N, seed=seed)
    sample_set = set(sample_times)
    idx_of = {t: i for i, t in enumerate(sample_times)}
    sr = np.full(len(sample_times), np.nan)
    snapshots = [None] * len(sample_times)
    meas_count = 0
    for step in range(1, t_max + 1):
        for _ in range(N):
            if rng.random() < r:
                q = int(rng.integers(0, N))
                traj.measure_z(q)
                meas_count += 1
            else:
                a = int(rng.integers(0, N))
                b = int(rng.integers(0, N - 1))
                if b >= a:
                    b += 1
                c = int(rng.integers(0, n_cliff))
                traj.apply_clifford(c, a, b)
        if step in sample_set:
            k = idx_of[step]
            sr[k] = traj.s_r()
            snapshots[k] = traj.snapshot()
    return {"sr": sr, "snapshots": snapshots,
            "meas_count": meas_count, "seed": seed, "N": N,
            "times": list(sample_times), "traj": traj}


def save_trajectory(path, result):
    """
    Store the evolving-graph trajectory target-agnostically.
    """
    N = result["N"]
    snaps = result["snapshots"]
    offsets = np.zeros(len(snaps) + 1, dtype=np.int64)
    all_edges = []
    vops = np.zeros((len(snaps), 2 * N), dtype=np.uint8)
    for k, (edges, v) in enumerate(snaps):
        all_edges.append(edges)
        offsets[k + 1] = offsets[k] + len(edges)
        vops[k] = v
    node_type = np.zeros(2 * N, dtype=np.uint8)
    node_type[N:] = 1
    np.savez_compressed(
        path, sr=result["sr"], times=np.array(result["times"]),
        vops=vops,
        edges_flat=(np.concatenate(all_edges) if all_edges
                    else np.zeros((0, 2), dtype=np.int32)),
        edge_offsets=offsets, node_type=node_type,
        seed=result["seed"], meas_count=result["meas_count"], N=N)
