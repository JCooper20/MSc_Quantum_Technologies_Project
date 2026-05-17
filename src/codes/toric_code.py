"""
src/codes/toric_code.py
-----------------------
Toric Code implementation with:
  ToricCode    — geometry, syndrome computation, logical operators
  MWPMDecoder  — greedy minimum-weight perfect matching baseline
  CNNDecoder   — circular-padding ResNet decoder (best Stage 2 result)
  ToricDataset — dataset generator
  Trainer      — training loop with MWPM comparison

The CNNDecoder uses circular padding throughout to respect the toric
(periodic) boundary conditions, and two separate output heads for
horizontal and vertical qubits.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import time

from src.config import ToricConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# TORIC CODE
# ============================================================================

class ToricCode:
    """L×L toric code — syndrome measurement and logical operator checks."""

    def __init__(self, L: int):
        self.L           = L
        self.n_qubits    = 2 * L * L
        self.n_plaquettes = L * L

        self.plaquette_qubits        = self._build_plaquettes()
        self.logical_z1, self.logical_z2 = self._build_logical_operators()
        self.H = self._build_parity_check_matrix()

    def _build_plaquettes(self) -> np.ndarray:
        L = self.L
        p = np.zeros((L * L, 4), dtype=np.int32)
        for row in range(L):
            for col in range(L):
                idx       = row * L + col
                p[idx, 0] = row * L + col
                p[idx, 1] = ((row + 1) % L) * L + col
                p[idx, 2] = L * L + row * L + col
                p[idx, 3] = L * L + row * L + (col + 1) % L
        return p

    def _build_logical_operators(self):
        L  = self.L
        z1 = np.zeros(self.n_qubits, dtype=np.int8)
        z2 = np.zeros(self.n_qubits, dtype=np.int8)
        for col in range(L):
            z1[col] = 1
        for row in range(L):
            z2[L * L + row * L] = 1
        return z1, z2

    def _build_parity_check_matrix(self) -> np.ndarray:
        H = np.zeros((self.n_plaquettes, self.n_qubits), dtype=np.int8)
        for p_idx, qubits in enumerate(self.plaquette_qubits):
            H[p_idx, qubits] = 1
        return H

    def generate_error(self, p: float) -> np.ndarray:
        return (np.random.random(self.n_qubits) < p).astype(np.int8)

    def get_syndrome(self, error: np.ndarray) -> np.ndarray:
        return (self.H @ error) % 2

    def get_syndrome_2d(self, error: np.ndarray) -> np.ndarray:
        return self.get_syndrome(error).reshape(self.L, self.L)

    def get_homology_class(self, chain: np.ndarray) -> Tuple[int, int]:
        return (int(np.dot(chain, self.logical_z2) % 2),
                int(np.dot(chain, self.logical_z1) % 2))

    def check_recovery(self, error: np.ndarray, recovery: np.ndarray) -> bool:
        h1, h2 = self.get_homology_class((error + recovery) % 2)
        return h1 == 0 and h2 == 0


# ============================================================================
# MWPM DECODER (baseline)
# ============================================================================

class MWPMDecoder:
    """Greedy minimum-weight perfect matching decoder with precomputed distances."""

    def __init__(self, code: ToricCode):
        self.code = code
        self.L    = code.L
        self._precompute_distances()

    def _precompute_distances(self):
        L = self.L
        n = L * L
        self.distances = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            r1, c1 = i // L, i % L
            for j in range(i + 1, n):
                r2, c2 = j // L, j % L
                dr = min(abs(r1 - r2), L - abs(r1 - r2))
                dc = min(abs(c1 - c2), L - abs(c1 - c2))
                self.distances[i, j] = self.distances[j, i] = dr + dc

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        flat    = syndrome.flatten()
        defects = list(np.where(flat == 1)[0])
        if len(defects) == 0:
            return np.zeros(self.code.n_qubits, dtype=np.int8)
        if len(defects) % 2 != 0:
            defects = defects[:-1]

        recovery  = np.zeros(self.code.n_qubits, dtype=np.int8)
        remaining = list(defects)

        while len(remaining) >= 2:
            best_d, best_i, best_j = float('inf'), 0, 1
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    d = self.distances[remaining[i], remaining[j]]
                    if d < best_d:
                        best_d, best_i, best_j = d, i, j

            for q in self._get_path(remaining[best_i], remaining[best_j]):
                recovery[q] ^= 1
            remaining = [r for k, r in enumerate(remaining)
                         if k not in [best_i, best_j]]

        return recovery

    def _get_path(self, p1: int, p2: int) -> List[int]:
        L       = self.L
        r1, c1  = p1 // L, p1 % L
        r2, c2  = p2 // L, p2 % L
        path    = []
        c, r    = c1, r1

        dc = (c2 - c1) % L
        if dc > L // 2:
            dc -= L
        for _ in range(abs(dc)):
            if dc > 0:
                path.append(L * L + r * L + (c + 1) % L)
                c = (c + 1) % L
            else:
                path.append(L * L + r * L + c)
                c = (c - 1) % L

        dr = (r2 - r1) % L
        if dr > L // 2:
            dr -= L
        for _ in range(abs(dr)):
            if dr > 0:
                path.append(((r + 1) % L) * L + c)
                r = (r + 1) % L
            else:
                path.append(r * L + c)
                r = (r - 1) % L

        return path


# ============================================================================
# CNN DECODER
# ============================================================================

class CircularPad2d(nn.Module):
    """Circular (toric) padding for 2D convolutions."""
    def __init__(self, padding: int):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, [self.padding] * 4, mode='circular')


class ResBlock(nn.Module):
    """Residual block with circular padding and BatchNorm."""
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CircularPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            CircularPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class CNNDecoder(nn.Module):
    """
    CNN decoder with circular padding for toric topology.

    Two output heads — one for horizontal qubits, one for vertical —
    each outputting per-qubit error probabilities.
    """

    def __init__(self, config: ToricConfig):
        super().__init__()
        self.L        = config.L
        self.n_qubits = config.n_qubits
        C             = config.base_channels

        self.initial = nn.Sequential(
            CircularPad2d(1),
            nn.Conv2d(1, C, 3),
            nn.BatchNorm2d(C),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[
            ResBlock(C, config.dropout)
            for _ in range(config.num_res_blocks)
        ])
        self.h_head = nn.Sequential(
            CircularPad2d(1), nn.Conv2d(C, C // 2, 3), nn.ReLU(),
            nn.Conv2d(C // 2, 1, 1))
        self.v_head = nn.Sequential(
            CircularPad2d(1), nn.Conv2d(C, C // 2, 3), nn.ReLU(),
            nn.Conv2d(C // 2, 1, 1))

    def forward(self, syndrome):
        if syndrome.dim() == 2:
            x = syndrome.unsqueeze(0).unsqueeze(0)
        elif syndrome.dim() == 3:
            x = syndrome.unsqueeze(1)
        else:
            x = syndrome

        batch = x.shape[0]
        x = self.initial(x)
        x = self.res_blocks(x)
        h = torch.sigmoid(self.h_head(x)).view(batch, -1)
        v = torch.sigmoid(self.v_head(x)).view(batch, -1)
        return torch.cat([h, v], dim=1)

    def decode(self, syndrome: np.ndarray, code: ToricCode) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            syn_2d = (syndrome.reshape(self.L, self.L)
                      if syndrome.ndim == 1 else syndrome)
            x     = torch.FloatTensor(syn_2d).unsqueeze(0).unsqueeze(0).to(DEVICE)
            probs = self.forward(x).cpu().numpy().flatten()

        recovery = (probs > 0.5).astype(np.int8)
        return self._fix_syndrome(recovery, syndrome.flatten(), code, probs)

    def _fix_syndrome(self, recovery, target, code, probs):
        current = code.get_syndrome(recovery)
        target  = target.flatten()
        for _ in range(50):
            if np.array_equal(current, target):
                break
            diff = np.where(current != target)[0]
            if len(diff) == 0:
                break
            qubits = code.plaquette_qubits[diff[0]]
            scores = [probs[q] if recovery[q] == 0 else 1 - probs[q]
                      for q in qubits]
            recovery[qubits[np.argmax(scores)]] ^= 1
            current = code.get_syndrome(recovery)
        return recovery


# ============================================================================
# DATASET
# ============================================================================

class ToricDataset(Dataset):
    def __init__(self, code: ToricCode, p_error: float, num_samples: int):
        self.L = code.L
        print(f"Generating {num_samples:,} samples...")
        self.syndromes = np.zeros((num_samples, self.L, self.L), dtype=np.float32)
        self.errors    = np.zeros((num_samples, code.n_qubits), dtype=np.float32)
        for i in tqdm(range(num_samples), desc="Generating"):
            error             = code.generate_error(p_error)
            self.syndromes[i] = code.get_syndrome_2d(error)
            self.errors[i]    = error
        print("✓ Done")

    def __len__(self):
        return len(self.syndromes)

    def __getitem__(self, idx):
        return {
            'syndrome': torch.FloatTensor(self.syndromes[idx]),
            'error':    torch.FloatTensor(self.errors[idx]),
        }


# ============================================================================
# TRAINER
# ============================================================================

class ToricTrainer:
    """Training loop with per-epoch MWPM comparison."""

    def __init__(self, model: CNNDecoder, config: ToricConfig,
                 train_loader: DataLoader, val_loader: DataLoader):
        self.model        = model.to(DEVICE)
        self.config       = config
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.code         = ToricCode(config.L)
        self.mwpm         = MWPMDecoder(self.code)

        self.optimizer = optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs)
        self.history   = defaultdict(list)
        self.best_gap  = float('inf')

    def _train_epoch(self):
        self.model.train()
        total = 0.0
        for batch in self.train_loader:
            syndrome = batch['syndrome'].to(DEVICE)
            error    = batch['error'].to(DEVICE)
            pred     = self.model(syndrome)
            loss     = F.binary_cross_entropy(pred, error)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
        return total / len(self.train_loader)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total, correct, n = 0.0, 0, 0
        for batch in self.val_loader:
            syndrome = batch['syndrome'].to(DEVICE)
            error    = batch['error'].to(DEVICE)
            pred     = self.model(syndrome)
            total   += F.binary_cross_entropy(pred, error).item()
            correct += ((pred > 0.5).float() == error).sum().item()
            n       += error.numel()
        return total / len(self.val_loader), correct / n

    def evaluate(self, num_samples: int = 300) -> Dict:
        self.model.eval()
        n_fail = m_fail = 0
        for _ in range(num_samples):
            error    = self.code.generate_error(self.config.p_error)
            syndrome = self.code.get_syndrome(error)
            if not self.code.check_recovery(
                    error, self.model.decode(syndrome, self.code)):
                n_fail += 1
            if not self.code.check_recovery(
                    error, self.mwpm.decode(syndrome)):
                m_fail += 1
        return {'neural': n_fail / num_samples, 'mwpm': m_fail / num_samples}

    def train(self) -> Dict:
        print(f"\n{'='*60}\nTRAINING CNN TORIC CODE DECODER\n{'='*60}")
        print(f"L={self.config.L}, p={self.config.p_error}, "
              f"epochs={self.config.num_epochs}")
        start = time.time()

        for epoch in range(self.config.num_epochs):
            t0         = time.time()
            train_loss = self._train_epoch()
            val_loss, val_acc = self._validate()
            self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if (epoch + 1) % 5 == 0:
                res = self.evaluate(self.config.eval_samples_training)
                self.history['neural_failure'].append(res['neural'])
                self.history['mwpm_failure'].append(res['mwpm'])
                gap    = res['neural'] - res['mwpm']
                marker = "✓ BEST" if gap < self.best_gap else ""
                if gap < self.best_gap:
                    self.best_gap = gap
                    torch.save(self.model.state_dict(),
                               'results/best_toric_decoder.pt')
                print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | "
                      f"Acc: {val_acc:.3f} | Neural: {res['neural']:.3f} | "
                      f"MWPM: {res['mwpm']:.3f} | Gap: {gap:+.3f} | "
                      f"{time.time()-t0:.1f}s {marker}")
            else:
                print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | "
                      f"Acc: {val_acc:.3f} | {time.time()-t0:.1f}s")

        self.model.load_state_dict(
            torch.load('results/best_toric_decoder.pt'))
        print(f"\nTraining time: {(time.time()-start)/60:.1f} min")
        return dict(self.history)
