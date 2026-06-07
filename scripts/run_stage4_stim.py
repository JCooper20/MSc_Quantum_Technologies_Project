"""
Stage 4: Stim stabiliser backend — MIPT sweep, scaling, CNN vs MLP.
 
"""
# Imports 
import numpy as np
import json
import time
from src.config import StimConfig
from src.simulators.stim_clifford import run_mipt_sweep, run_scaling
from src.analysis.encoders import generate_data
from src.models.phase_classifiers import train_cnn, cnn_predict_proba
from src.analysis.encoders import encode_flat, encode_2d
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
 
def run_comparison(cfg: StimConfig):
    L     = cfg.L_main
    depth = 4 * L
 
    difficulties = [
        {'label': 'Easy',   'p_low': 0.02, 'p_high': 0.50, 'n': 300},
        {'label': 'Medium', 'p_low': 0.05, 'p_high': 0.25, 'n': 300},
        {'label': 'Hard',   'p_low': 0.08, 'p_high': 0.15, 'n': 400},
    ]
    sweep_pm = np.linspace(0.0, 0.60, 15)
    results  = []
 
    for d in difficulties:
        print(f"\n  [{d['label']}] {d['p_low']} vs {d['p_high']}")
        data = generate_data(L, depth, d['p_low'], d['p_high'],
                             d['n'], cfg.meas_basis)
 
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=150,
                             early_stopping=True, random_state=42)
        mlp.fit(data['flat_train'], data['y_train'])
        mlp_va = accuracy_score(data['y_val'], mlp.predict(data['flat_val']))
 
        cnn_params, cnn_tr, cnn_va, cnn_hist = train_cnn(
            data['img_train'], data['y_train'],
            data['img_val'],   data['y_val'],
            epochs=cfg.clf_epochs, batch_size=cfg.clf_batch, lr=cfg.clf_lr)
 
        mlp_sweep, cnn_sweep = [], []
        from src.simulators.stim_clifford import run_trajectory_stim
        for p_m in sweep_pm:
            mp, cp = [], []
            for _ in range(cfg.sweep_n):
                t  = run_trajectory_stim(L, depth, p_m, cfg.meas_basis)
                xf = encode_flat(t['measurement_record']).reshape(1, -1)
                mp.append(mlp.predict_proba(xf)[0, 1])
                cp.append(cnn_predict_proba(cnn_params,
                                             encode_2d(t['measurement_record'])))
            mlp_sweep.append(float(np.mean(mp)))
            cnn_sweep.append(float(np.mean(cp)))
 
        results.append({
            'label': d['label'], 'p_low': d['p_low'], 'p_high': d['p_high'],
            'mlp_val': mlp_va,   'cnn_val': cnn_va,
            'sweep_pm': list(sweep_pm),
            'mlp_sweep': mlp_sweep, 'cnn_sweep': cnn_sweep,
        })
    return results
 
 
def main():
    cfg = StimConfig()
    t0  = time.time()
 
    print("\nPart 1: MIPT sweep")
    mipt = run_mipt_sweep(cfg.L_main, 4 * cfg.L_main,
                           cfg.pm_fine, cfg.n_traj_mipt, cfg.meas_basis)
 
    print("\nPart 2: Scaling")
    scaling = run_scaling(cfg.L_values, cfg.scaling_pm,
                          cfg.n_traj_scaling, cfg.meas_basis)
 
    print("\nPart 3: CNN vs MLP")
    comparison = run_comparison(cfg)
 
    save = {
        'mipt': {'p_m': mipt['p_m'], 'mean_S': mipt['mean_S']},
        'scaling': {str(L): {'p_m': d['p_m'], 'S': d['mean_S']}
                    for L, d in scaling.items()},
        'comparison': comparison,
    }
    with open('results/stim_clifford_data.json', 'w') as f:
        json.dump(save, f, indent=2)
 
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")
    print("Saved → results/stim_clifford_data.json")
 
 
if __name__ == '__main__':
    main()
