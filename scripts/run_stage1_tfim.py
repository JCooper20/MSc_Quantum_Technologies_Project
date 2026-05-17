"""
scripts/run_stage1_tfim.py
--------------------------
Stage 1: TFIM Neural Controller
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
 
from src.config import TFIMConfig
from src.simulators.tfim import QiskitTFIMSimulator
from src.models.tfim_controller import TFIMNeuralController, TFIMDataset, TFIMTrainer
 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
def main():
    config = TFIMConfig()
    print(f"\nTFIM Neural Controller — Stage 1  |  Device: {DEVICE}")
 
    train_loader = DataLoader(
        TFIMDataset(config, config.num_train_trajectories),
        batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(
        TFIMDataset(config, config.num_val_trajectories),
        batch_size=config.batch_size)
 
    model   = TFIMNeuralController(config).to(DEVICE)
    trainer = TFIMTrainer(model, config, train_loader, val_loader)
    history = trainer.train()
    final   = trainer.evaluate_controller(config.num_eval_trajectories)
 
    print(f"\n  Controlled:   {final['controlled_fidelity']:.4f} ± {final['controlled_std']:.4f}")
    print(f"  Uncontrolled: {final['uncontrolled_fidelity']:.4f} ± {final['uncontrolled_std']:.4f}")
    print(f"  Improvement:  {final['improvement']:+.4f}")
 
    torch.save({'model_state_dict': model.state_dict(),
                'config': vars(config), 'history': history,
                'final_eval': final}, 'results/tfim_controller.pt')
    print("Saved → results/tfim_controller.pt")
 
if __name__ == '__main__':
    main()
