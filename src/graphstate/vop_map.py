"""
Bridge between the Anders-Briegel engine's 24 vertex-operator (VOP)
codes and stim single-qubit Clifford tableaus.

Rather than hand-deriving the engine's sign conventions (a classic
source of subtle errors), the mapping is DERIVED AND PROVEN at import
time:
  - candidate generators: the two sqrt(X)-type tableaus (X->X,
    Z->+/-Y) and the two sqrt(Z)-type tableaus (Z->Z, X->+/-Y);
  - each VOP code's tableau is composed from the engine's own
    decomposition_table strings ('X'/'Z' factors) under a candidate
    (sqrtX sign, sqrtZ sign, string order, composition order);
  - a candidate assignment is accepted only if it satisfies ALL of:
      anchors: code 0 = I, 1 = X, 2 = Y, 3 = Z, 10 = H, 6 = S or
               S_DAG (paired consistently with 5),
      the FULL 24 x 24 group homomorphism against the engine's
      multiplication_table:
          map[mult[a, b]] == map[a] after map[b]   (or the reverse
          order, fixed by the same search),
  - the search must yield a unique map (tableaus are phase-free, so
    global phases cannot hide differences). Anything else raises.

This makes the bridge self-verifying: if the vendored lookup tables
ever changed, import would fail loudly rather than silently corrupt
the verification gate.
"""

# Imports
import numpy as np
import stim

from src.graphstate.engine.lookup_tables import (multiplication_table,
                                                 decomposition_table)


def _candidate_generators():
    xs, zs = [], []
    for sign in (1, -1):
        t = stim.Tableau.from_conjugated_generators(
            xs=[stim.PauliString("X")],
            zs=[stim.PauliString("Y") * sign])
        xs.append(t)
        t = stim.Tableau.from_conjugated_generators(
            xs=[stim.PauliString("Y") * sign],
            zs=[stim.PauliString("Z")])
        zs.append(t)
    return xs, zs


def _build_map(sqrt_x, sqrt_z, reverse_string, compose_ab):
    """VOP code -> tableau under one candidate convention."""
    ident = stim.Tableau(1)
    out = []
    for code in range(24):
        s = decomposition_table[code]
        if reverse_string:
            s = s[::-1]
        t = ident
        for ch in s:
            g = sqrt_x if ch == "X" else sqrt_z
            t = t.then(g) if compose_ab else g.then(t)
        out.append(t)
    return out


def _anchors_ok(m):
    I = stim.Tableau(1)
    X = stim.Tableau.from_named_gate("X")
    Y = stim.Tableau.from_named_gate("Y")
    Z = stim.Tableau.from_named_gate("Z")
    H = stim.Tableau.from_named_gate("H")
    S = stim.Tableau.from_named_gate("S")
    SD = stim.Tableau.from_named_gate("S_DAG")
    return (m[0] == I and m[1] == X and m[2] == Y and m[3] == Z
            and m[10] == H
            # the engine's API names code 6 "s": adopt that I/O
            # convention strictly (all-real dynamics is invariant
            # under global complex conjugation, so internal checks
            # alone cannot distinguish S from S_DAG — the trajectory
            # state-equality gate confirms this choice end to end)
            and m[6] == S and m[5] == SD)


def _homomorphism_ok(m):
    for a in range(24):
        for b in range(24):
            # engine's apply(): new state VOP = mult[new, old],
            # meaning operator NEW acts AFTER operator OLD:
            # map[mult[a, b]] must equal map[b] followed by map[a].
            if m[multiplication_table[a, b]] != m[b].then(m[a]):
                return False
    return True


def _cz_table_ok(m):
    """Dynamics disambiguator + vendored-table validation: the map
    must make every one of the 2 x 24 x 24 cz_table entries TRUE in
    stim on an isolated pair:
        CZ (U_a x U_b) CZ^h |++>  ==  (U_a' x U_b') CZ^e |++>.
    Anchors + homomorphism alone admit the complex-conjugate map pair
    (S <-> S_DAG); only the engine's dynamics distinguishes them."""
    from src.graphstate.engine.lookup_tables import cz_table

    def state(vop_a, vop_b, edge, extra_cz):
        sim = stim.TableauSimulator()
        sim.set_num_qubits(2)
        sim.h(0)
        sim.h(1)
        if edge:
            sim.cz(0, 1)
        sim.do_tableau(m[vop_a], [0])
        sim.do_tableau(m[vop_b], [1])
        if extra_cz:
            sim.cz(0, 1)
        return sim.canonical_stabilizers()

    for h in (0, 1):
        for a in range(24):
            for b in range(24):
                e, va, vb = cz_table[h, a, b]
                if state(a, b, h, True) != state(va, vb, e, False):
                    return False
    return True


def _derive():
    xs, zs = _candidate_generators()
    found = []
    for sx in xs:
        for sz in zs:
            for rev in (False, True):
                for order in (False, True):
                    m = _build_map(sx, sz, rev, order)
                    if _anchors_ok(m) and _homomorphism_ok(m) and \
                            _cz_table_ok(m):
                        found.append(m)
    if not found:
        raise RuntimeError(
            "VOP->stim map derivation failed: no candidate convention "
            "satisfies anchors + homomorphism + cz_table dynamics")
    first = found[0]
    for other in found[1:]:
        if any(first[k] != other[k] for k in range(24)):
            raise RuntimeError(
                "VOP->stim map derivation ambiguous: multiple distinct "
                "maps satisfy all constraints")
    return first


VOP_TABLEAUS = _derive()
