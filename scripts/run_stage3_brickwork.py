"""
Stage 3: Monitored Brickwork Circuit — Qiskit Statevector
 
Runs the Ising-type monitored brickwork circuit using the exact Qiskit
statevector simulator. Limited to L ≲ 12 due to exponential memory cost,
but gives exact entropy values that serve as ground truth for validating
the Stim stabiliser backend in Stage 4.
 
Pipeline:
  1. MIPT sweep   — S(L/2) vs p_m for a single L
  2. Scaling study — S(L/2) vs p_m for L = 6, 8, 10, 12
  3. Classifier   — CNN vs MLP on measurement records near the transition
"""
 
import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
 
from src.config import BrickworkConfig
from src.simulators.statevector import MonitoredBrickworkCircuit
from src.analysis.encoders import encode_flat, encode_2d
from src.models.phase_classifiers import train_cnn, cnn_predict_proba
 
 
# ============================================================================
# PART 1 — MIPT SWEEP
# ============================================================================
 
def run_mipt_sweep(cfg: BrickworkConfig) -> dict:
    """
    Sweep p_m and compute trajectory-averaged S(L/2) for cfg.L.
    """
    print(f"\n{'='*60}")
    print(f"  Part 1: MIPT sweep  (L={cfg.L}, depth={cfg.depth})")
    print(f"{'='*60}")
 
    circuit = MonitoredBrickworkCircuit(
        L=cfg.L, p_u=cfg.p_u,
        depth=cfg.depth, meas_basis=cfg.meas_basis)
 
    results = {'p_m': [], 'mean_S': [], 'std_S': [], 'sem_S': [],
               'mean_history': []}
 
    for p_m in cfg.p_m_values:
        entropies  = []
        histories  = []
        t0 = time.time()
 
        for _ in range(cfg.n_trajectories):
            traj = circuit.run_trajectory(p_m)
            entropies.append(traj['final_entropy'])
            histories.append(traj['entropy_history'])
 
        mean = float(np.mean(entropies))
        std  = float(np.std(entropies))
        sem  = std / np.sqrt(cfg.n_trajectories)
 
        results['p_m'].append(p_m)
        results['mean_S'].append(mean)
        results['std_S'].append(std)
        results['sem_S'].append(sem)
        results['mean_history'].append(
            np.mean(histories, axis=0).tolist())
 
        print(f"  p_m={p_m:.2f}  S={mean:.3f} ± {sem:.3f}"
              f"  ({time.time()-t0:.1f}s)")
 
    return results
 
 
# ============================================================================
# PART 2 — FINITE-SIZE SCALING
# ============================================================================
 
def run_scaling(cfg: BrickworkConfig) -> dict:
    """
    Run the MIPT sweep for each L in cfg.L_values.
    Returns {L: mipt_dict}.
    """
    print(f"\n{'='*60}")
    print(f"  Part 2: Finite-size scaling  (L = {cfg.L_values})")
    print(f"{'='*60}")
 
    scaling = {}
    for L in cfg.L_values:
        print(f"\n  L = {L}  (2^{L} = {2**L} amplitudes)")
        circuit = MonitoredBrickworkCircuit(
            L=L, p_u=cfg.p_u,
            depth=cfg.depth, meas_basis=cfg.meas_basis)
 
        pm_list, mean_list, sem_list = [], [], []
        for p_m in cfg.p_m_values:
            entropies = [circuit.run_trajectory(p_m)['final_entropy']
                         for _ in range(cfg.n_trajectories)]
            m   = float(np.mean(entropies))
            sem = float(np.std(entropies) / np.sqrt(cfg.n_trajectories))
            pm_list.append(p_m)
            mean_list.append(m)
            sem_list.append(sem)
            print(f"    p_m={p_m:.2f}  S={m:.3f} ± {sem:.3f}")
 
        scaling[L] = {'p_m': pm_list, 'mean_S': mean_list, 'sem_S': sem_list}
 
    return scaling
 
 
# ============================================================================
# PART 3 — PHASE CLASSIFIER
# ============================================================================
 
def generate_classifier_data(cfg: BrickworkConfig) -> dict:
    """
    Generate labelled measurement records:
      label 0 = volume-law (p_m = p_low)
      label 1 = area-law   (p_m = p_high)
    """
    print(f"\n  Generating classifier data "
          f"({cfg.p_low} vs {cfg.p_high}, n={cfg.n_train_samples})...")
 
    circuit = MonitoredBrickworkCircuit(
        L=cfg.L, p_u=cfg.p_u,
        depth=cfg.depth, meas_basis=cfg.meas_basis)
 
    flat_X, img_X, labels = [], [], []
    n_each = cfg.n_train_samples // 2
 
    for p_m, label, name in [(cfg.p_low,  0, 'Volume-law'),
                               (cfg.p_high, 1, 'Area-law  ')]:
        print(f"    {name} (p_m={p_m}): ", end='', flush=True)
        for i in range(n_each):
            traj = circuit.run_trajectory(p_m)
            rec  = traj['measurement_record']
            flat_X.append(encode_flat(rec))
            img_X.append(encode_2d(rec))
            labels.append(label)
            if (i + 1) % 50 == 0:
                print(f"{i+1} ", end='', flush=True)
        print()
 
    flat_X = np.array(flat_X)
    img_X  = np.array(img_X)
    labels = np.array(labels)
 
    idx    = np.random.permutation(len(labels))
    flat_X, img_X, labels = flat_X[idx], img_X[idx], labels[idx]
 
    n_val  = len(labels) // 5
    return {
        'flat_train': flat_X[n_val:],  'flat_val': flat_X[:n_val],
        'img_train':  img_X[n_val:],   'img_val':  img_X[:n_val],
        'y_train':    labels[n_val:],  'y_val':    labels[:n_val],
    }
 
 
def run_classifier(cfg: BrickworkConfig) -> dict:
    """Train CNN and MLP classifiers and compare near the transition."""
    print(f"\n{'='*60}")
    print(f"  Part 3: Phase classifier  (L={cfg.L})")
    print(f"{'='*60}")
 
    data = generate_classifier_data(cfg)
 
    # MLP baseline
    print("\n  Training MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200,
                         early_stopping=True, random_state=42)
    mlp.fit(data['flat_train'], data['y_train'])
    mlp_val = accuracy_score(data['y_val'], mlp.predict(data['flat_val']))
    print(f"    MLP val accuracy: {mlp_val:.3f}")
 
    # CNN
    print("\n  Training CNN...")
    cnn_params, cnn_tr, cnn_val, cnn_hist = train_cnn(
        data['img_train'], data['y_train'],
        data['img_val'],   data['y_val'],
        epochs=cfg.clf_epochs, batch_size=32, lr=1e-3)
    print(f"    CNN val accuracy: {cnn_val:.3f}")
 
    # Learnability sweep across all p_m values
    print("\n  Learnability sweep...")
    circuit   = MonitoredBrickworkCircuit(
        L=cfg.L, p_u=cfg.p_u,
        depth=cfg.depth, meas_basis=cfg.meas_basis)
    n_sweep   = 40
    mlp_probs, cnn_probs = [], []
 
    for p_m in cfg.p_m_values:
        mp, cp = [], []
        for _ in range(n_sweep):
            traj = circuit.run_trajectory(p_m)
            rec  = traj['measurement_record']
            xf   = encode_flat(rec).reshape(1, -1)
            mp.append(mlp.predict_proba(xf)[0, 1])
            cp.append(cnn_predict_proba(cnn_params, encode_2d(rec)))
        mlp_probs.append(float(np.mean(mp)))
        cnn_probs.append(float(np.mean(cp)))
        print(f"    p_m={p_m:.2f}  MLP={np.mean(mp):.3f}  "
              f"CNN={np.mean(cp):.3f}")
 
    return {
        'mlp_val':    mlp_val,
        'cnn_val':    cnn_val,
        'mlp_probs':  mlp_probs,
        'cnn_probs':  cnn_probs,
        'cnn_hist':   cnn_hist,
    }
 
 
# ============================================================================
# PART 4 — PLOTS
# ============================================================================
 
def create_plots(mipt: dict, scaling: dict,
                 clf: dict, cfg: BrickworkConfig) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f'Monitored Brickwork Circuit — Qiskit Statevector  '
        f'(L={cfg.L}, depth={cfg.depth})',
        fontsize=13, fontweight='bold')
 
    BLUE, RED, GREEN = '#2E86AB', '#E94F37', '#27AE60'
 
    # 1. MIPT sweep
    ax = axes[0, 0]
    ax.errorbar(mipt['p_m'], mipt['mean_S'], yerr=mipt['sem_S'],
                fmt='o-', color=BLUE, lw=2, ms=6, capsize=3)
    ax.set_xlabel('Measurement rate $p_m$')
    ax.set_ylabel('$S(L/2)$')
    ax.set_title('Half-chain entropy vs $p_m$', fontweight='bold')
    ax.grid(True, alpha=0.3)
 
    # 2. Finite-size scaling
    ax = axes[0, 1]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(scaling)))
    for (L, d), c in zip(scaling.items(), cmap):
        ax.plot(d['p_m'], d['mean_S'], 'o-', color=c,
                lw=2, ms=5, label=f'L={L}')
    ax.set_xlabel('Measurement rate $p_m$')
    ax.set_ylabel('$S(L/2)$')
    ax.set_title('Finite-size scaling', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
 
    # 3. Learnability
    ax = axes[1, 0]
    ax.plot(cfg.p_m_values, clf['mlp_probs'], 's--',
            color=RED,  lw=2, ms=6, label=f'MLP ({clf["mlp_val"]:.3f})')
    ax.plot(cfg.p_m_values, clf['cnn_probs'], 'o-',
            color=BLUE, lw=2, ms=6, label=f'CNN ({clf["cnn_val"]:.3f})')
    ax.axhline(0.5, color='gray', ls=':', lw=1.5)
    ax.set_xlabel('Measurement rate $p_m$')
    ax.set_ylabel('P(area-law)')
    ax.set_title('Learnability transition', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
 
    # 4. CNN training curve
    ax = axes[1, 1]
    ax.plot(clf['cnn_hist']['val_acc'], color=GREEN, lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation accuracy')
    ax.set_title('CNN training', fontweight='bold')
    ax.set_ylim([0.5, 1.05])
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig('results/brickwork_results.png', dpi=200, bbox_inches='tight')
    plt.savefig('results/brickwork_results.pdf', bbox_inches='tight')
    print("\nSaved → results/brickwork_results.png / .pdf")
 
 
# ============================================================================
# MAIN
# ============================================================================
 
def main():
    t_start = time.time()
    cfg     = BrickworkConfig()
 
    print("\n" + "=" * 60)
    print("  STAGE 3: MONITORED BRICKWORK CIRCUIT (STATEVECTOR)")
    print("=" * 60)
    print(f"  L={cfg.L}  depth={cfg.depth}  basis={cfg.meas_basis}")
    print(f"  2^L = {2**cfg.L} amplitudes  |  L_values = {cfg.L_values}")
    print(f"  WARNING: L > 12 will be very slow (statevector scales as 2^L)")
 
    mipt    = run_mipt_sweep(cfg)
    scaling = run_scaling(cfg)
    clf     = run_classifier(cfg)
 
    create_plots(mipt, scaling, clf, cfg)
 
    save = {
        'config': {
            'L': cfg.L, 'depth': cfg.depth,
            'meas_basis': cfg.meas_basis,
            'L_values': cfg.L_values,
        },
        'mipt': {
            'p_m':    mipt['p_m'],
            'mean_S': mipt['mean_S'],
            'sem_S':  mipt['sem_S'],
        },
        'scaling': {
            str(L): {'p_m': d['p_m'], 'mean_S': d['mean_S']}
            for L, d in scaling.items()
        },
        'classifier': {
            'mlp_val':   clf['mlp_val'],
            'cnn_val':   clf['cnn_val'],
            'mlp_probs': clf['mlp_probs'],
            'cnn_probs': clf['cnn_probs'],
        },
    }
 
    with open('results/brickwork_data.json', 'w') as f:
        json.dump(save, f, indent=2)
 
    print(f"\nSaved → results/brickwork_data.json")
    print(f"\n{'='*60}")
    print(f"  ALL DONE — {(time.time()-t_start)/60:.1f} min")
    print(f"{'='*60}")
 
    # Quick summary
    p_c_idx = int(np.argmin(np.gradient(mipt['mean_S'])))
    print(f"\n  Estimated crossover near p_m ≈ "
          f"{mipt['p_m'][p_c_idx]:.2f}")
    print(f"  CNN val accuracy: {clf['cnn_val']:.3f}")
    print(f"  MLP val accuracy: {clf['mlp_val']:.3f}")
 
 
if __name__ == '__main__':
    main()
