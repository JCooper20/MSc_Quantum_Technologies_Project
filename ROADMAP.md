# Research Roadmap

## Status at submission of Progress Report (Feb 2026)

Stages 1–2 complete. Stage 3 (Stim backend) and Stage 4 (TCN/GRU) implemented in code. Stage 5 (TFIM neural controller) prototyped. Literature review finalised March 2026.

---

## Phase A — Scalable Simulation & Finite-Size Scaling

**Goal:** Replace Qiskit statevector backend (L ≲ 12) with Stim stabiliser-tableau backend to access L = 8, 16, 32, 64.

**Tasks:**
- [ ] Validate Stim entropy results against Qiskit statevector at L = 6, 8, 10
- [ ] Sweep *p*_m for both Ising-type and random Clifford ensembles at all target L
- [ ] Extract *p*_c from crossing analysis of S(L/2)/L curves
- [ ] Compute tripartite mutual information I₃ and attempt finite-size scaling collapse via ansatz: I₃(p, L) = f((p − p_c) · L^{1/ν})
- [ ] Estimate critical exponent ν

**Key question:** Does the Stim backend reproduce the crossover seen in statevector, and can we resolve a sharp transition at larger L?

---

## Phase B — Temporal Architectures for Phase Detection

**Goal:** Determine whether temporal neural architectures (TCN, GRU) outperform the spatial CNN near *p*_c.

**Tasks:**
- [ ] Standardise dataset format (two-channel spatiotemporal image: was_measured + outcome)
- [ ] Benchmark CNN, TCN, GRU at matched parameter counts across L = 8, 16, 32, 64
- [ ] Plot classification accuracy vs |*p*_m − *p*_c| for each architecture
- [ ] Check whether performance gap between temporal and spatial models increases near criticality (where long-range temporal correlations are expected)
- [ ] Compare against MLP baseline from Stage 2

**Key question:** Do temporal architectures that respect causal ordering of the measurement record provide a measurable advantage for phase detection near criticality?

---

## Phase C — Adaptive Feedback & Entanglement Stabilisation

**Goal:** Use the best temporal architecture from Phase B as a policy backbone for a real-time adaptive feedback controller.

**Tasks:**
- [ ] Define action space: at each circuit layer, for each qubit, decide whether to measure (binary per-qubit action)
- [ ] Define reward: half-chain entropy S(L/2) at end of trajectory (or running entropy integral)
- [ ] Implement adaptive feedback loop: controller reads measurement record sequentially and outputs measurement decisions
- [ ] Train with policy gradient (PPO or REINFORCE); compare to Sign-Colour Decoder (SCD) baseline where tractable
- [ ] Test on GHZ / cat-state stabilisation targets
- [ ] Measure correction gain ΔS/S_max vs uncontrolled circuit and greedy local decoder

**Key question:** Can a neural network trained solely on measurement outcomes (without access to the stabiliser tableau) match the Sign-Colour Decoder? Can it sustain volume-law entanglement above the passive *p*_c?

---

## Phase D — Multi-Agent RL Decomposition *(stretch goal)*

*Contingent on successful completion of Phase C.*

**Goal:** Test whether the control problem can be spatially decomposed using a multi-agent RL (MARL) framework.

**Tasks:**
- [ ] Partition chain into N = 4 and N = 8 agents, each observing a local window of the measurement record
- [ ] Each agent acts on a local subset of qubits; agents share a global entanglement reward
- [ ] Train under Centralised Training with Decentralised Execution (CTDE) paradigm
- [ ] Compare performance of N-agent system vs global single-agent controller
- [ ] Assess whether performance is sustained as L increases beyond single-agent reach

**Key question:** Does the light-cone structure of the brickwork circuit provide sufficient locality for the MARL decomposition to be lossless, and can local agents with only neighbour communication match a global controller?

---

## Open Questions (from Lit Review)

1. Do temporal architectures (TCN, GRU) provide a quantitative advantage over spatial CNNs near criticality, where long-range temporal correlations in the measurement record are expected to develop?
2. Can a neural network trained solely on measurement outcomes match the Sign-Colour Decoder of Paszko et al. (2025), without access to the stabiliser structure?
3. Can a learned RL policy sustain volume-law entanglement at measurement rates that a passive circuit cannot? Can it access dynamical phases inaccessible to any fixed-rule strategy?
4. Can a spatially decomposed multi-agent controller match a global single-agent policy at larger system sizes? Does the light-cone structure make this decomposition lossless?
