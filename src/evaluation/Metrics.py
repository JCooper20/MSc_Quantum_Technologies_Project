"""
Evaluation metrics for quantum controller performance
Phase A: Add comprehensive evaluation to existing quantum controller
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from qiskit.quantum_info import Statevector


def evaluate_controller(
    model,
    circuit,
    params,
    num_episodes=50,
    phase='maintain',
    device='cpu'
):
    """
    Comprehensive evaluation of trained controller
    
    Returns:
        summary (dict): Aggregated metrics
        results (dict): Per-episode results
    """
    
    from src.quantum.quantum_controller import OracleController
    
    model.eval()
    oracle = OracleController(target_type='GHZ')
    
    results = {
        'final_overlaps': [],
        'stabilization_times': [],
        'total_interventions': [],
        'trajectory_lengths': [],
        'success': []
    }
    
    with torch.no_grad():
        for episode in range(num_episodes):
            # Initialize state based on phase
            if phase == 'maintain':
                state = circuit.initialize_near_ghz(noise_level=0.15)
            else:
                state = circuit.initialize_plus_state()
            
            # Track trajectory
            measurements_seq = []
            syndromes_seq = []
            overlaps = []
            interventions = 0
            
            initial_overlap = oracle._compute_overlap(state, circuit.L)
            overlaps.append(initial_overlap)
            
            stabilization_time = None
            max_steps = 50
            
            for step in range(max_steps):
                # Run one timestep
                state_copy, record = circuit.run_single_timestep(state, step)
                measurements_seq.append(record['measurements'])
                syndromes_seq.append(record['syndrome'])
                
                # Prepare input for model
                meas_tensor = torch.zeros(1, len(measurements_seq), circuit.L)
                for t, meas_list in enumerate(measurements_seq):
                    for qubit, outcome in meas_list:
                        meas_tensor[0, t, qubit] = outcome
                
                synd_tensor = torch.zeros(1, len(syndromes_seq), circuit.L)
                for t, synd in enumerate(syndromes_seq):
                    synd_tensor[0, t] = torch.from_numpy(synd)
                
                params_tensor = torch.tensor([[params.p_u, params.p_m]])
                
                # Get model prediction
                actions_pred, qubits_pred, halt_pred = model(
                    meas_tensor.to(device),
                    synd_tensor.to(device),
                    params_tensor.to(device)
                )
                
                # Decode action
                action_idx = actions_pred[0, -1].argmax().item()
                qubit_idx = qubits_pred[0, -1].argmax().item()
                should_halt = halt_pred[0, -1].item() > 0.5
                
                # Apply action
                if action_idx == 1:  # Apply gate
                    i = qubit_idx
                    j = min(i + 1, circuit.L - 1)
                    state = circuit._apply_two_qubit_gate(state_copy, i, j)
                    interventions += 1
                elif action_idx == 2:  # Remeasure
                    state, _ = circuit.measure_x_basis(state_copy, qubit_idx)
                    interventions += 1
                else:  # Do nothing
                    state = state_copy
                
                # Track overlap
                current_overlap = oracle._compute_overlap(state, circuit.L)
                overlaps.append(current_overlap)
                
                # Check if stabilized
                if current_overlap >= 0.95 and stabilization_time is None:
                    stabilization_time = step + 1
                
                # Check halt
                if should_halt and step > 10:
                    break
            
            # Record results
            results['final_overlaps'].append(overlaps[-1])
            results['stabilization_times'].append(stabilization_time if stabilization_time else max_steps)
            results['total_interventions'].append(interventions)
            results['trajectory_lengths'].append(len(overlaps))
            results['success'].append(overlaps[-1] >= 0.95)
    
    # Compute summary
    summary = {
        'success_rate': np.mean(results['success']),
        'mean_final_overlap': np.mean(results['final_overlaps']),
        'std_final_overlap': np.std(results['final_overlaps']),
        'mean_stabilization_time': np.mean([t for t in results['stabilization_times'] if t < max_steps]),
        'mean_interventions': np.mean(results['total_interventions']),
        'intervention_rate': np.mean([i/l for i, l in zip(results['total_interventions'], 
                                                           results['trajectory_lengths'])])
    }
    
    return summary, results


def evaluate_no_control_baseline(circuit, params, num_episodes=100, phase='maintain'):
    """
    Baseline: Let circuit evolve with NO interventions
    """
    
    from src.quantum.quantum_controller import OracleController
    
    oracle = OracleController(target_type='GHZ')
    results = {
        'final_overlaps': [],
        'trajectory_lengths': []
    }
    
    for episode in range(num_episodes):
        if phase == 'maintain':
            state = circuit.initialize_near_ghz(noise_level=0.15)
        else:
            state = circuit.initialize_plus_state()
        
        initial_overlap = oracle._compute_overlap(state, circuit.L)
        
        max_steps = 50
        for step in range(max_steps):
            state, _ = circuit.run_single_timestep(state, step)
        
        final_overlap = oracle._compute_overlap(state, circuit.L)
        results['final_overlaps'].append(final_overlap)
    
    summary = {
        'mean_final_overlap': np.mean(results['final_overlaps']),
        'std_final_overlap': np.std(results['final_overlaps']),
        'success_rate': np.mean([o >= 0.95 for o in results['final_overlaps']])
    }
    
    return summary
