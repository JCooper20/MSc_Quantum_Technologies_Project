"""
scripts/run_stage2_toric.py
---------------------------
Stage 2: CNN Toric Code Decoder
 
Trains a circular-padding ResNet to decode the L=6 toric code,
comparing against the MWPM baseline across a sweep of error rates.
"""
 
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
 
from src.config import ToricConfig
from src.codes.decoder import (
    ToricCode, MWPMDecoder, CNNDecoder,
    ToricDataset, ToricTrainer
)
 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
 
def evaluate_all(model: CNNDecoder, config: ToricConfig):
    """Sweep error rates and compare Neural vs MWPM failure rates."""
    print(f"\n{'='*60}\nCOMPREHENSIVE EVALUATION\n{'='*60}")
    code     = ToricCode(config.L)
    mwpm     = MWPMDecoder(code)
    model.eval()
 
    p_errors = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
    results  = {'p_error': p_errors, 'neural': [], 'mwpm': [], 'gap': []}
 
    for p in tqdm(p_errors, desc="Error sweep"):
        n_fail = m_fail = 0
        for _ in range(config.eval_samples_final):
            error    = code.generate_error(p)
            syndrome = code.get_syndrome(error)
            if not code.check_recovery(error, model.decode(syndrome, code)):
                n_fail += 1
            if not code.check_recovery(error, mwpm.decode(syndrome)):
                m_fail += 1
        neural = n_fail / config.eval_samples_final
        mwpm_r = m_fail / config.eval_samples_final
        results['neural'].append(neural)
        results['mwpm'].append(mwpm_r)
        results['gap'].append(neural - mwpm_r)
        print(f"  p={p:.2f}: Neural={neural:.3f}  MWPM={mwpm_r:.3f}  "
              f"Gap={neural-mwpm_r:+.3f}")
    return results
 
 
def main():
    config = ToricConfig()
    print(f"\nToric Code CNN Decoder — Stage 2  |  Device: {DEVICE}")
    print(f"L={config.L}  p_error={config.p_error}  "
          f"epochs={config.num_epochs}")
 
    code         = ToricCode(config.L)
    train_loader = DataLoader(
        ToricDataset(code, config.p_error, config.num_train_samples),
        batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(
        ToricDataset(code, config.p_error, config.num_val_samples),
        batch_size=config.batch_size)
 
    model   = CNNDecoder(config).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
 
    trainer = ToricTrainer(model, config, train_loader, val_loader)
    history = trainer.train()
    results = evaluate_all(model, config)
 
    # Save
    torch.save({'model': model.state_dict(),
                'history': history, 'eval': results},
               'results/toric_decoder.pt')
 
    with open('results/toric_results.json', 'w') as f:
        json.dump(results, f, indent=2)
 
    print(f"\nSaved → results/toric_decoder.pt, results/toric_results.json")
 
    idx = results['p_error'].index(0.08)
    print(f"\nAt p=0.08:")
    print(f"  Neural failure: {results['neural'][idx]:.3f}")
    print(f"  MWPM failure:   {results['mwpm'][idx]:.3f}")
    print(f"  Gap:            {results['gap'][idx]:+.3f}")
 
 
if __name__ == '__main__':
    main()
