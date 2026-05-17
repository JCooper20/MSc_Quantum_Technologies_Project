"""
scripts/run_stage5_architectures.py
------------------------------------
Stage 5: CNN / TCN / GRU architecture comparison.
 
Benchmarks three temporal architectures against the spatial CNN and MLP
baseline as a function of L and |p_m - p_c|. Tests the hypothesis that
temporal models respecting causal ordering outperform spatial CNNs near
the critical point.
"""
 
import numpy as np
import json
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
 
from src.config import ArchConfig
from src.simulators.stim_clifford import run_trajectory_stim
from src.analysis.encoders import generate_data_all, encode_flat, encode_2d, encode_temporal
from src.models.phase_classifiers import train_cnn, cnn_predict_proba
from src.models.phase_classifiers import train_tcn
from src.models.phase_classifiers import train_gru
 
 
def run_delta_sweep(cfg: ArchConfig):
    """
    Train each architecture at (p_c - δ, p_c + δ) for varying δ.
    Produces a 'learnability gap' curve.
    """
    L     = cfg.L_main
    depth = 4 * L
    results = []
 
    for delta in cfg.delta_values:
        p_low  = max(0.01, cfg.p_c_estimate - delta)
        p_high = min(0.99, cfg.p_c_estimate + delta)
        print(f"\n  δ={delta:.2f}  ({p_low:.2f} vs {p_high:.2f})")
 
        data = generate_data_all(L, depth, p_low, p_high,
                                  cfg.n_traj_delta, cfg.meas_basis)
 
        # MLP
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=150,
                             early_stopping=True, random_state=42)
        mlp.fit(data['flat_train'], data['y_train'])
        mlp_va = accuracy_score(data['y_val'], mlp.predict(data['flat_val']))
 
        # CNN
        _, _, cnn_va, _ = train_cnn(
            data['img_train'], data['y_train'],
            data['img_val'],   data['y_val'],
            epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr)
 
        # TCN
        _, _, tcn_va, _ = train_tcn(
            data['seq_train'], data['y_train'],
            data['seq_val'],   data['y_val'],
            hidden=cfg.hidden, kernel_size=cfg.kernel_size,
            epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr)
 
        # GRU
        _, _, gru_va, _ = train_gru(
            data['seq_train'], data['y_train'],
            data['seq_val'],   data['y_val'],
            hidden=cfg.gru_hidden,
            epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr)
 
        results.append({
            'delta': delta, 'p_low': p_low, 'p_high': p_high,
            'mlp': mlp_va, 'cnn': cnn_va, 'tcn': tcn_va, 'gru': gru_va,
        })
        print(f"    MLP={mlp_va:.3f}  CNN={cnn_va:.3f}  "
              f"TCN={tcn_va:.3f}  GRU={gru_va:.3f}")
 
    return results
 
 
def main():
    cfg = ArchConfig()
    t0  = time.time()
 
    print("\nStage 5: Architecture comparison")
    delta_results = run_delta_sweep(cfg)
 
    save = {'delta_sweep': delta_results, 'config': {
        'L_main': cfg.L_main, 'p_c': cfg.p_c_estimate,
        'deltas': cfg.delta_values,
    }}
    with open('results/arch_comparison.json', 'w') as f:
        json.dump(save, f, indent=2)
 
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")
    print("Saved → results/arch_comparison.json")
 
 
if __name__ == '__main__':
    main()
