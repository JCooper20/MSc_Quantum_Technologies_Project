"""
scripts/run_stage1_tfim.py
--------------------------
Stage 1: TFIM Neural Controller

Trains a Conv+LSTM controller to preserve GHZ-state fidelity
under Trotter evolution, stochastic measurements, and bit-flip errors.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm.auto import tqdm

from src.config import TFIMConfig
from src.simulators.tfim import QiskitTFIMSimulator, Action

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Re-import model and trainer from the original monolith until refactored
# (TFIMNeuralController and TFIMTrainer are PyTorch — candidates for
#  models/tfim_controller.py in a future PR)
from src.msc_project import TFIMNeuralController, TFIMDataset, TFIMTrainer
from src.msc_project import run_mipt_analysis, run_parameter_sweep, create_plots


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    config = TFIMConfig()
    print(f"\nTFIM Neural Controller — Stage 1")
    print(f"Device: {DEVICE}")

    train_dataset = TFIMDataset(config, config.num_train_trajectories)
    val_dataset   = TFIMDataset(config, config.num_val_trajectories)
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size,
                                shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=config.batch_size)

    model   = TFIMNeuralController(config)
    trainer = TFIMTrainer(model, config, train_loader, val_loader)
    history = trainer.train()

    final_eval    = trainer.evaluate_controller(config.num_eval_trajectories)
    mipt_results  = run_mipt_analysis(model, config)
    param_results = run_parameter_sweep(model, config)
    create_plots(history, mipt_results, param_results, config)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config':           vars(config),
        'history':          history,
        'final_eval':       final_eval,
    }, 'results/tfim_controller.pt')
    print("\nDone. Saved → results/tfim_controller.pt")


if __name__ == '__main__':
    main()
