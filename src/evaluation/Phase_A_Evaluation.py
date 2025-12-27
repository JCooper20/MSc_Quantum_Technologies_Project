"""
Phase A: Evaluate Existing Quantum Controller
Add comprehensive metrics and baseline comparison
"""

import sys
sys.path.insert(0, '..')

import torch
from src.quantum.quantum_controller import (
    CircuitParams, 
    BrickworkCircuit, 
    QuantumControllerLSTM
)
from src.evaluation.metrics import evaluate_controller, evaluate_no_control_baseline

print("="*70)
print("PHASE A: EVALUATING QUANTUM CONTROLLER")
print("="*70)

# Load your trained models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

# Setup circuit
L = 6
params = CircuitParams(L=L, p_u=0.5, p_m=0.3)
circuit = BrickworkCircuit(params)

# Load Phase 1 model
print("\n" + "="*70)
print("LOADING PHASE 1 MODEL (MAINTENANCE)")
print("="*70)

model_phase1 = QuantumControllerLSTM(L=L, d_model=128, num_layers=3)
try:
    checkpoint = torch.load('best_model_maintain.pt', map_location=device)
    model_phase1.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Phase 1 model loaded")
except:
    print("⚠ Warning: Could not load Phase 1 model. Using untrained model.")

# Evaluate Phase 1
print("\n" + "="*70)
print("EVALUATING PHASE 1 (MAINTENANCE)")
print("="*70)

summary_p1, results_p1 = evaluate_controller(
    model_phase1,
    circuit,
    params,
    num_episodes=100,
    phase='maintain',
    device=device
)

print(f"\nPhase 1 Results:")
print(f"  Success Rate:           {summary_p1['success_rate']:.1%}")
print(f"  Mean Final Overlap:     {summary_p1['mean_final_overlap']:.4f} ± {summary_p1['std_final_overlap']:.4f}")
print(f"  Mean Stabilization Time: {summary_p1['mean_stabilization_time']:.1f} steps")
print(f"  Intervention Rate:      {summary_p1['intervention_rate']:.3f} actions/step")

# Baseline comparison
print("\n" + "="*70)
print("NO-CONTROL BASELINE (MAINTENANCE)")
print("="*70)

baseline_p1 = evaluate_no_control_baseline(circuit, params, num_episodes=100, phase='maintain')

print(f"\nBaseline Results:")
print(f"  Mean Final Overlap:     {baseline_p1['mean_final_overlap']:.4f} ± {baseline_p1['std_final_overlap']:.4f}")
print(f"  Success Rate:           {baseline_p1['success_rate']:.1%}")

print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")
print(f"Controller vs No-Control:")
print(f"  Overlap Improvement:    {summary_p1['mean_final_overlap'] - baseline_p1['mean_final_overlap']:+.4f}")
print(f"  Success Rate Gain:      {summary_p1['success_rate'] - baseline_p1['success_rate']:+.1%}")

# Load Phase 2 model
print("\n" + "="*70)
print("LOADING PHASE 2 MODEL (CREATION)")
print("="*70)

model_phase2 = QuantumControllerLSTM(L=L, d_model=128, num_layers=3)
try:
    checkpoint = torch.load('best_model_create.pt', map_location=device)
    model_phase2.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Phase 2 model loaded")
except:
    print("⚠ Warning: Could not load Phase 2 model. Using untrained model.")

# Evaluate Phase 2
print("\n" + "="*70)
print("EVALUATING PHASE 2 (CREATION)")
print("="*70)

summary_p2, results_p2 = evaluate_controller(
    model_phase2,
    circuit,
    params,
    num_episodes=100,
    phase='create',
    device=device
)

print(f"\nPhase 2 Results:")
print(f"  Success Rate:           {summary_p2['success_rate']:.1%}")
print(f"  Mean Final Overlap:     {summary_p2['mean_final_overlap']:.4f} ± {summary_p2['std_final_overlap']:.4f}")
print(f"  Mean Stabilization Time: {summary_p2['mean_stabilization_time']:.1f} steps")
print(f"  Intervention Rate:      {summary_p2['intervention_rate']:.3f} actions/step")

# Baseline comparison
print("\n" + "="*70)
print("NO-CONTROL BASELINE (CREATION)")
print("="*70)

baseline_p2 = evaluate_no_control_baseline(circuit, params, num_episodes=100, phase='create')

print(f"\nBaseline Results:")
print(f"  Mean Final Overlap:     {baseline_p2['mean_final_overlap']:.4f} ± {baseline_p2['std_final_overlap']:.4f}")
print(f"  Success Rate:           {baseline_p2['success_rate']:.1%}")

print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")
print(f"Controller vs No-Control:")
print(f"  Overlap Improvement:    {summary_p2['mean_final_overlap'] - baseline_p2['mean_final_overlap']:+.4f}")
print(f"  Success Rate Gain:      {summary_p2['success_rate'] - baseline_p2['success_rate']:+.1%}")

print("\n" + "="*70)
print("PHASE A EVALUATION COMPLETE!")
print("="*70)
print("\n✓ You now have quantitative metrics for your controller!")
print("✓ Next: Add visualization (plotting.py)")
print("✓ Then: Start Phase B (Ising model)")
