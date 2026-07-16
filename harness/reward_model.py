"""
Pixel reward model — GI use-case #3 made concrete (code truth → pixels).

The code-vs-pixel *illustration* (eval.py) uses a hand-tuned per-tile detector — honest but
constructed. This module does the real thing: a tiny CNN trained ONLY on code-truth labels
(the exact, label-free supervision the environment emits for free) that predicts a code-defined
event — "has the can been picked up?" — from a rendered frame, with the HUD cropped off so it
cannot cheat by reading the predicate ticks.

The honest outcome is whatever it is: on a clean scene a full-frame model can approximate the
code label well (low held-out disagreement), which IS the point — code truth is exact, label-free
supervision a pixel reward model only approximates. torch is imported lazily (rl extra only).
"""

from __future__ import annotations

import numpy as np


def frame_to_input(frame_pil, size: int = 48) -> np.ndarray:
    img = frame_pil.convert("RGB").resize((size, size))
    return (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)  # CHW


class PixelRewardModel:
    def __init__(self, size: int = 48):
        self.size = size
        self.net = None

    def _build(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(32 * 16, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def fit(self, frames, labels, epochs: int = 60, seed: int = 0):
        import torch, torch.nn as nn
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        y = np.array(labels, dtype=np.float32)
        # class-balance by resampling to the minority count (avoids the collapse that a large
        # pos_weight causes) so the model must actually learn to separate the two classes
        pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
        if len(pos_idx) and len(neg_idx):
            k = min(len(pos_idx), len(neg_idx))
            idx = np.concatenate([rng.choice(pos_idx, k, replace=len(pos_idx) < k),
                                  rng.choice(neg_idx, k, replace=len(neg_idx) < k)])
        else:
            idx = np.arange(len(frames))
        X = torch.tensor(np.stack([frame_to_input(frames[i], self.size) for i in idx]))
        yt = torch.tensor(y[idx]).unsqueeze(1)
        self.net = self._build()
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            loss_fn(self.net(X), yt).backward()
            opt.step()
        return self

    def predict(self, frames) -> np.ndarray:
        import torch
        X = torch.tensor(np.stack([frame_to_input(f, self.size) for f in frames]))
        with torch.no_grad():
            return (self.net(X).squeeze(1) > 0).numpy()
