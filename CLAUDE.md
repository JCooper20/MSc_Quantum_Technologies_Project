# CLAUDE.md — Project Briefing for Claude Code
 
## Rules — Read First
 
These rules apply for every session. Do not deviate from them without
explicit permission from Joseph.
 
1. **Do NOT edit docstrings, comments, or variable names** — these have
   been carefully written and reviewed. Leave them exactly as they are.
2. **Do NOT refactor, restructure, or rename anything** without being
   explicitly asked to do so or suggesting.
3. **Do NOT make changes to files that are not directly related to the
   task** — if fixing a bug in stim_clifford.py, do not touch any other
   file unless it is directly necessary.
4. **Always ask before editing any source file** — show me the proposed
   change and wait for approval before applying it.
5. **Run experiments using the --save flag only when I say so** — default
   to dry runs (no --save) unless I explicitly say the run is worth keeping.
6. **Always run from the project root with PYTHONPATH=.**
   ```
   PYTHONPATH=. python scripts/run_stage4_stim.py
   ```
 
---
 
## Project Overview
 
**Title:** Many-Body Quantum States of Quantum Circuits and Neural Networks
**Degree:** MSc Quantum Technologies, UCL
**Author:** Joseph Cooper (Student No. 22004534)
**Supervisors:** Prof. Arijeet Pal, Prof. Sougato Bose
 
This project investigates whether neural network controllers trained on
real-time measurement outcomes can detect and stabilise entanglement in
1D monitored brickwork circuits exhibiting measurement-induced phase
transitions (MIPTs).
 
The central scientific questions are:
- Can classical neural networks learn to detect phase boundaries from
  measurement records alone (the learnability transition)?
- Can adaptive feedback driven by a neural controller stabilise
  volume-law entanglement at rates that would otherwise drive area-law
  behaviour?
---
 
## Current State of the Project
 
All six stages are implemented and the codebase has been fully refactored
into a clean package structure. The code has been fully documented with
mathematical docstrings throughout.
 
**Completed:**
- Stage 1: TFIM neural controller (Conv+LSTM, GHZ fidelity preservation)
- Stage 2: Toric code CNN decoder (L=6, 95.4% accuracy at p=0.08)
- Stage 3: Monitored brickwork statevector (Qiskit, L≤12, MIPT observed)
- Stage 4: Stim stabiliser backend (L up to 64, CNN vs MLP benchmarked)
- Stage 5: CNN/TCN/GRU architecture comparison implemented
- Stage 6: REINFORCE adaptive RL controller implemented
**Immediate priorities (Phase A):**
- Get all six scripts running cleanly end to end
- Run systematic MIPT sweep for L = 8, 16, 32, 64 via Stim backend
- Extract p_c from crossing analysis of S(L/2)/L curves
- Implement finite-size scaling collapse to extract critical exponent ν
---
 
## Codebase Structure
 
```
MSc_Quantum_Technologies_Project/
├── scripts/                    ← entry points, one per stage
│   ├── run_stage1_tfim.py
│   ├── run_stage2_toric.py
│   ├── run_stage3_brickwork.py
│   ├── run_stage4_stim.py
│   ├── run_stage5_architectures.py
│   └── run_stage6_rl.py
├── src/
│   ├── config.py               ← all dataclasses (TFIMConfig, StimConfig etc.)
│   ├── simulators/
│   │   ├── tfim.py             ← QiskitTFIMSimulator, Action enum
│   │   ├── statevector.py      ← MonitoredBrickworkCircuit (Qiskit, L≤12)
│   │   └── stim_clifford.py    ← run_trajectory_stim, run_mipt_sweep (Stim)
│   ├── codes/
│   │   └── toric_code.py       ← ToricCode, MWPMDecoder, CNNDecoder
│   ├── models/
│   │   ├── phase_classifiers.py ← CNN, TCN, GRU (JAX)
│   │   └── policy.py           ← RL policy network, episode runners
│   ├── training/
│   │   ├── adam.py             ← shared JAX AdamOptimizer
│   │   └── reinforce.py        ← supervised pretrain + REINFORCE
│   └── analysis/
│       ├── entropy.py          ← gf2_rank, stabiliser_entropy
│       └── encoders.py         ← encode_flat/2d/temporal, generate_data
├── results/
│   ├── checkpoints/            ← model weights (.pt files)
│   ├── figures/                ← plots (.png, .pdf)
│   ├── data/                   ← generated datasets (.npy, .json)
│   └── logs/                   ← training logs (.json)
├── data/                       ← raw input data (gitignored)
├── requirements.txt
├── ROADMAP.md
└── CLAUDE.md                   ← this file
```
 
---
 
## Key Technical Details
 
**Environment:**
- Python 3.11
- Virtual environment at `.venv/` — always activate before running
- Always run scripts with `PYTHONPATH=.` from the repo root
**Backends:**
- Qiskit Statevector — exact simulation, limited to L ≲ 12
- Stim stabiliser tableau — O(L²) scaling, used for L = 8, 16, 32, 64
- Never mix backends within a single experiment
**JAX models:**
- All JAX models use the shared AdamOptimizer from src/training/adam.py
- Parameters stored as pytree dicts, not nn.Module
- Use jit and grad from JAX for compiled gradients
**PyTorch models:**
- TFIMNeuralController and CNNDecoder use PyTorch nn.Module
- Trained with AdamW + cosine annealing scheduler
**Running scripts:**
```bash
# Dry run (no saving)
PYTHONPATH=. python scripts/run_stage4_stim.py
 
# Save results with label
PYTHONPATH=. python scripts/run_stage4_stim.py --save --tag phase_a_L16
 
# Results land in:
# results/checkpoints/stage4_phase_a_L16_*.pt
# results/figures/stage4_phase_a_L16_*.png
# results/logs/stage4_phase_a_L16_*.json
```
 
---
 
## Research Plan — Four Phases
 
### Phase A — Scalable Simulation & Finite-Size Scaling
Immediate priority. Run MIPT sweeps for L = 8, 16, 32, 64 using
stim_clifford.run_trajectory_stim. Extract p_c from crossing analysis
of S(L/2)/L. Fit finite-size scaling ansatz:
 
    S(L/2) = f((p_m - p_c) · L^{1/ν})
 
to estimate critical exponent ν.
 
New file needed: src/analysis/scaling.py
    - finite_size_collapse()
    - fit_scaling_ansatz()
    - tripartite_mutual_info()
 
### Phase B — Temporal Architecture Comparison
Benchmark CNN, TCN, GRU at matched parameter counts.
Plot classification accuracy vs |p_m - p_c| for each architecture.
Key question: do temporal architectures outperform CNN near criticality?
 
### Phase C — Adaptive Feedback & Entanglement Stabilisation
Use best Phase B architecture as policy backbone.
Train REINFORCE controller to select measurement locations.
Compare against Sign-Colour Decoder baseline.
Measure correction gain ΔS/S_max.
 
New file needed: src/training/ppo.py (more stable than REINFORCE)
 
### Phase D — Multi-Agent RL (stretch goal)
Partition chain into N=4 and N=8 local agents.
CTDE training with shared entanglement reward.
New package needed: src/agents/
 
---
 
## Plotting Standards
 
All figures for the thesis should:
- Use matplotlib with a clean style (no grid by default unless helpful)
- Have proper LaTeX axis labels: e.g. r'$p_m$', r'$S(L/2)$'
- Include error bars (sem_S from sweep results)
- Use a consistent colour scheme across figures
- Be saved as both .png (dpi=200) and .pdf for the thesis
- Be saved to results/figures/ with prefix final_ to be tracked by git
---
 
## Open Scientific Questions
 
1. Do temporal architectures (TCN, GRU) outperform CNN near p_c?
2. Can the neural controller match the Sign-Colour Decoder without
   access to the stabiliser tableau?
3. Can a learned RL policy sustain volume-law entanglement above p_c?
4. Does the MARL decomposition perform losslessly given light-cone structure?
---
