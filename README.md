# Many-Body Quantum States of Quantum Circuits and Neural Networks

**UCL MSc Quantum Technologies**

**Author:** Joseph Cooper 

**Supervisors:** Prof. Arijeet Pal · Prof. Andrew Fisher

**Academic Year:** 2025–26

---

## Overview

This project investigates whether neural network controllers trained on real-time measurement outcomes can learn to preserve and stabilise highly entangled many-body states in noisy, dynamically evolving quantum circuits.

Monitored quantum circuits, alternating layers of entangling unitary gates and stochastic projective measurements, exhibit a **measurement-induced phase transition (MIPT)** at a critical measurement rate *p*_c, separating a volume-law entangled phase from an area-law phase. A key insight from recent literature is that this phase boundary coincides exactly with a **learnability transition**: the point at which a classical observer can no longer decode a global property of the initial state from the measurement record alone. This equivalence connects quantum phase structure, quantum error correction, and classical machine learning — and raises the possibility that a network capable of *reading* the measurement record can be repurposed as a real-time **feedback controller** to actively stabilise entanglement.

---

## Research Plan

**Phase A — Scalable simulation & finite-size scaling**
Stabiliser-tableau backend; systematic sweeps over *p*_m for L = 8, 16, 32, 64; crossing analysis of S(L/2)/L to extract *p*_c; finite-size scaling collapse of I₃.

**Phase B — Temporal architectures for phase detection**
Replace spatial CNN with TCN and GRU; benchmark classification accuracy vs L and |*p*_m − *p*_c| at matched parameter counts; test whether temporal processing provides a quantitative advantage near criticality.

**Phase C — Adaptive feedback & entanglement stabilisation**
Use the best temporal architecture from Phase B as a policy backbone; implement adaptive feedback loop selecting measurement locations per circuit layer; target stabilisation of GHZ/cat states at rates that would otherwise drive area-law behaviour; compare to Sign-Colour Decoder baseline.

**Phase D — Multi-agent RL decomposition**
Partition chain into N = 4 or 8 local agents (CTDE paradigm); test whether local agents communicating with neighbours can match a global single-agent controller; assess scalability beyond L = 64.

---

## Key References

1. Skinner, Ruhman, Nahum — *Measurement-induced phase transitions in the dynamics of entanglement*, Phys. Rev. X 9, 031009 (2019)
2. Agrawal et al. — *Observing quantum measurement collapse as a learnability phase transition*, Phys. Rev. X 14, 041012 (2024)
3. Paszko, Szyniszewski, Pal — *Dynamic syndrome decoder in volume-law phases of hybrid quantum circuits*, arXiv:2508.13045 (2025)
4. Torlai, Melko — *Neural decoder for topological codes*, Phys. Rev. Lett. 119, 030501 (2017)
5. Krastanov, Jiang — *Deep neural network probabilistic decoder for stabilizer codes*, Sci. Rep. 7, 11003 (2017)
6. Dehghani, Lavasani, Hafezi — *Neural-network decoders for measurement-induced phase transitions*, Nat. Commun. 14, 2918 (2023)
7. Angelidi, Szyniszewski, Pal — *Stabilization of symmetry-protected long-range entanglement in stochastic quantum circuits*, Quantum (2024)
