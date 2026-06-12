"""
Exercise 3 — RNN-T forward-backward from scratch (starter).

Read README.md before starting. Fill each TODO. Run the matching test.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


NEG_INF = -1e30


def logsumexp(*xs: float) -> float:
    m = max(xs)
    if m == NEG_INF:
        return NEG_INF
    s = sum(np.exp(x - m) for x in xs)
    return m + float(np.log(s))


# -----------------------------------------------------------------------------
# TODO 1: extract blank and target log-probabilities from the joint tensor
# -----------------------------------------------------------------------------
# joint_log_probs: [T, U+1, V+1]   already-log-softmaxed along the V+1 axis
# target:          length-U sequence of vocab indices (no blanks)
# blank:           the blank token index (typically V)
# Returns:
#   log_p_blank:  [T, U+1]   joint_log_probs[t, u, blank]
#   log_p_target: [T, U]     joint_log_probs[t, u, target[u]]
def extract_blank_and_label_log_probs(
    joint_log_probs: np.ndarray,
    target: Sequence[int],
    blank: int,
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("TODO 1")


# -----------------------------------------------------------------------------
# TODO 2: forward variable in log space
# -----------------------------------------------------------------------------
# log_p_blank:  [T, U+1]
# log_p_target: [T, U]
# Returns log_alpha of shape [T+1, U+1] with log_alpha[0, 0] = 0.
def rnnt_forward(log_p_blank: np.ndarray, log_p_target: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO 2")


# -----------------------------------------------------------------------------
# TODO 3: backward variable in log space
# -----------------------------------------------------------------------------
# Returns log_beta of shape [T+1, U+1] with log_beta[T, U] = 0.
def rnnt_backward(log_p_blank: np.ndarray, log_p_target: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO 3")


# -----------------------------------------------------------------------------
# TODO 4: loss and gradient w.r.t. joint_log_probs
# -----------------------------------------------------------------------------
# Returns:
#   loss: scalar = -log P(y | x)
#   grad: [T, U+1, V+1] = d loss / d joint_log_probs
def rnnt_loss_and_grad(
    joint_log_probs: np.ndarray,
    target: Sequence[int],
    blank: int,
) -> tuple[float, np.ndarray]:
    raise NotImplementedError("TODO 4")


# -----------------------------------------------------------------------------
# TODO 5: greedy decoder (precomputed joint version)
# -----------------------------------------------------------------------------
# Walks the lattice from (0, 0) to (T, U_emitted) using argmax at each cell.
def rnnt_greedy_decode(
    joint_log_probs: np.ndarray,
    blank: int,
    max_symbols_per_step: int = 10,
) -> list[int]:
    raise NotImplementedError("TODO 5")
