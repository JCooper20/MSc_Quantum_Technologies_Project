"""
Censoring-aware waiting-time kernels.

Per entropy level, the sweeps spent there before the next unit drop,
combined by the Kaplan-Meier product-limit estimator

    S(t) = prod_{t_i <= t} (1 - d_i / n_i)

for d_i events among n_i at risk. No binning, no smoothing: event times
are exact integer durations.

Censored residences (trajectory still at level k when the horizon ends)
enter through the risk sets only. Fitting on completed residences alone
is biased short — slow ones are preferentially cut off — and the walk
then over-absorbs.

Kaplan & Meier, J. Am. Stat. Assoc. 53, 457 (1958).
"""

# Imports
import collections

import numpy as np


# =====================================
#  Wilson score interval
# =====================================

def wilson(p, n, zc=1.96):
    """
    Interval for a proportion p from n trials. Preferred over the normal
    approximation, which misbehaves near p = 0 and 1 — where purified
    fractions live.

        centre = p + z^2/2n,  denom = 1 + z^2/n
        half   = z sqrt( p(1-p)/n + z^2/4n^2 )
    """
    d = 1 + zc * zc / n
    c = p + zc * zc / (2 * n)
    h = zc * np.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n))
    return (c - h) / d, (c + h) / d


# =====================================
#  Level visits
# =====================================

def visits(stairs, idx):
    """
    Yield (rung, duration, event, next_rung) for the selected
    trajectories; event 0 = censored, next_rung -1 = censored.

    Staircases are flat rungs/entries arrays cut by off offsets. Level 0
    is absorbing and is skipped.
    """
    z = stairs
    off, R, E = z["off"], z["rungs"], z["entries"]
    ab, tau, t_max = z["absorbed"], z["tau"], int(z["t_max"])
    for j in idx:
        Rr, Ee = R[off[j]:off[j + 1]], E[off[j]:off[j + 1]]
        for i in range(len(Rr)):
            k = int(Rr[i])
            if k == 0:
                continue
            if i + 1 < len(Rr):
                yield k, int(Ee[i + 1] - Ee[i]), 1, int(Rr[i + 1])
            elif ab[j]:
                yield k, int(tau[j] - Ee[i]), 1, 0
            else:
                yield k, int(t_max - Ee[i]), 0, -1


# =====================================
#  Kaplan-Meier fit
# =====================================

def km_fit(obs):
    """
    obs = [(duration, event)]. Returns (durs, F, tail, n_events,
    n_censored), F = 1 - S at each event time.

    tail is S after the last event: the mass Kaplan-Meier cannot resolve
    past the final observation.
    """
    durs = sorted({d for d, e in obs if e})
    n_tot = len(obs)
    S = 1.0
    F = []
    for t in durs:
        n_risk = sum(1 for d, e in obs if d >= t)
        d_t = sum(1 for d, e in obs if e and d == t)
        S *= (1 - d_t / n_risk) if n_risk else 1.0
        F.append(1 - S)
    n_ev = sum(e for _, e in obs)
    n_cs = n_tot - n_ev
    return (np.array(durs), np.array(F), float(S), n_ev, n_cs)


def build_km_kernels(stairs, fit_idx):
    """
    One kernel per level: {level: dict(durs, F, tail, n_ev, n_cs, nxt)}.

    Jump targets are pooled by (level, duration), so a walk drawing a
    duration draws a target from the visits that took that long —
    keeping the residence-jump pairing where it is observable.
    """
    obs = collections.defaultdict(list)
    nxt_by_dur = collections.defaultdict(list)
    for k, d, e, nx in visits(stairs, fit_idx):
        obs[k].append((d, e))
        if e:
            nxt_by_dur[(k, d)].append(nx)
    kern = {}
    for k, o in obs.items():
        durs, F, tail, n_ev, n_cs = km_fit(o)
        kern[k] = dict(durs=durs, F=F, tail=tail, n_ev=n_ev,
                       n_cs=n_cs, nxt=nxt_by_dur)
    return kern
