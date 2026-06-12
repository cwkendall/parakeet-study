"""
Tiny end-to-end demonstration: train a 3-layer CNN with your CTC loss.

Run after the unit tests pass:

    python train_toy_classifier.py

This deliberately uses your `ctc_loss_and_grad` from starter.py — not PyTorch's
loss — so you're seeing your own implementation drive a training loop.

The "task" is synthetic: per-frame 16-d acoustic features whose labels are
short 3-letter words chosen from a tiny vocabulary. Each "letter" has its
own random Gaussian feature distribution; the CNN has to figure out which
letter is active at each frame and the CTC loss does the alignment.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from starter import ctc_loss_and_grad, ctc_greedy_decode


# ------------------------------------------------------------------
# Synthetic data: per-letter Gaussian feature distributions
# ------------------------------------------------------------------
VOCAB = "abcdefghijklmnopqrstuvwxyz"  # 26 letters
BLANK = len(VOCAB)                    # blank index
WORDS = ["cat", "dog", "bird", "fish", "lion", "tiger", "bear", "wolf",
         "deer", "fox", "hare", "moth", "frog", "duck", "owl", "snake"]


def _make_letter_centres(seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((len(VOCAB), 16)) * 2.0


CENTRES = _make_letter_centres()


def synth_example(word: str, frames_per_letter: int = 6, noise: float = 0.5,
                  rng: np.random.Generator | None = None):
    """Return (features [T, 16], target [U]) for one word."""
    if rng is None:
        rng = np.random.default_rng()
    target = [VOCAB.index(c) for c in word]
    feats = []
    for c in target:
        n = frames_per_letter + rng.integers(-2, 3)  # jitter
        for _ in range(max(1, n)):
            feats.append(CENTRES[c] + rng.standard_normal(16) * noise)
    return np.stack(feats, 0), np.array(target)


# ------------------------------------------------------------------
# A tiny CNN encoder
# ------------------------------------------------------------------
class TinyEncoder(nn.Module):
    def __init__(self, n_in=16, n_hidden=64, n_classes=27):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in, n_hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(n_hidden, n_hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(n_hidden, n_classes, 5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 16]
        out = self.net(x.transpose(1, 2)).transpose(1, 2)  # [B, T, n_classes]
        return out


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def train(n_epochs=30, examples_per_epoch=64, lr=1e-2):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = TinyEncoder(n_classes=BLANK + 1)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        total_loss = 0.0
        for _ in range(examples_per_epoch):
            word = WORDS[rng.integers(len(WORDS))]
            feats_np, target = synth_example(word, rng=rng)
            feats = torch.from_numpy(feats_np).float().unsqueeze(0)
            logits_t = model(feats)[0]  # [T, V+1]
            logits_np = logits_t.detach().numpy().astype(np.float64)

            # Our CTC loss & gradient
            loss, grad_np = ctc_loss_and_grad(logits_np, target.tolist(), BLANK)
            total_loss += loss

            # Push gradient through PyTorch's autograd by hand.
            grad_t = torch.from_numpy(grad_np).float()
            model.zero_grad()
            logits_t.backward(grad_t)
            optim.step()

        # End-of-epoch eval on a few words
        eval_word = WORDS[epoch % len(WORDS)]
        feats_np, target = synth_example(eval_word, rng=np.random.default_rng(123 + epoch))
        feats = torch.from_numpy(feats_np).float().unsqueeze(0)
        with torch.no_grad():
            log_probs = torch.log_softmax(model(feats)[0], dim=-1).numpy().astype(np.float64)
        decode = ctc_greedy_decode(log_probs, BLANK)
        decode_str = "".join(VOCAB[i] for i in decode if i < len(VOCAB))
        print(f"Epoch {epoch:2d}: avg loss = {total_loss/examples_per_epoch:6.2f}  "
              f"sample decode = {decode_str!r:>10}  (truth: {eval_word!r})")


if __name__ == "__main__":
    train()
