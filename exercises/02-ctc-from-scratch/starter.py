"""
Exercise 2 — CTC from scratch (starter).

Fill in each TODO. Run the matching pytest after each one. Read README.md first.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


NEG_INF = -1e30  # we represent log(0) as a very negative number


def logsumexp(*xs: float) -> float:
    """Numerically stable log-sum-exp of a few scalars."""
    m = max(xs)
    if m == NEG_INF:
        return NEG_INF
    s = sum(np.exp(x - m) for x in xs)
    return m + float(np.log(s))


# -----------------------------------------------------------------------------
# TODO 1: build the extended target [B, y_1, B, y_2, B, ..., y_U, B]
# -----------------------------------------------------------------------------
def extend_target(target: Sequence[int], blank: int) -> list[int]:
    """
    target = [3, 1, 20], blank = 0  ->  [0, 3, 0, 1, 0, 20, 0]
    Length of the result is 2*len(target) + 1.
    """
    raise NotImplementedError("TODO 1: interleave blanks into the target")


# -----------------------------------------------------------------------------
# TODO 2: forward variable α(t, s) in log space
# -----------------------------------------------------------------------------
# log_probs has shape [T, V+1]: log of the per-frame probability distribution.
# Returns log_alpha of shape [T, S] where S = 2*U + 1.
def ctc_forward(
    log_probs: np.ndarray,
    target: Sequence[int],
    blank: int,
) -> np.ndarray:
    raise NotImplementedError("TODO 2: implement the CTC forward in log space")


# -----------------------------------------------------------------------------
# TODO 3: backward variable β(t, s) in log space
# -----------------------------------------------------------------------------
# Returns log_beta of shape [T, S].
def ctc_backward(
    log_probs: np.ndarray,
    target: Sequence[int],
    blank: int,
) -> np.ndarray:
    raise NotImplementedError("TODO 3: implement the CTC backward in log space")


# -----------------------------------------------------------------------------
# TODO 4: loss and analytical gradient w.r.t. the input logits
# -----------------------------------------------------------------------------
# logits has shape [T, V+1]. Returns:
#   loss = -log P(y | x)  (scalar)
#   grad = dL / dlogits   (shape [T, V+1])
#
# Algorithm:
#   1. log_probs = log_softmax(logits, axis=-1)
#   2. probs = exp(log_probs)
#   3. log_alpha = ctc_forward(...)
#   4. log_beta  = ctc_backward(...)
#   5. log_P = logsumexp(log_alpha[T-1, S-1], log_alpha[T-1, S-2])
#   6. For each (t, k):
#        grad[t, k] = probs[t, k] - (1 / P) * sum_{s : ext[s] == k} exp(log_alpha[t, s] + log_beta[t, s] - log p[t, k])
#      Equivalently in log space:
#        sum_{s : ext[s] == k} exp(log_alpha[t, s] + log_beta[t, s] - log p[t, k] - log_P)
#      gives the occupancy ratio. Subtract from probs[t, k].
def ctc_loss_and_grad(
    logits: np.ndarray,
    target: Sequence[int],
    blank: int,
) -> tuple[float, np.ndarray]:
    raise NotImplementedError("TODO 4: implement CTC loss and analytical gradient")


# -----------------------------------------------------------------------------
# TODO 5: greedy decoder
# -----------------------------------------------------------------------------
def ctc_greedy_decode(log_probs: np.ndarray, blank: int) -> list[int]:
    """argmax per frame, then collapse repeats, then strip blanks."""
    raise NotImplementedError("TODO 5: implement greedy CTC decoding")


# -----------------------------------------------------------------------------
# TODO 6: prefix beam search
# -----------------------------------------------------------------------------
# log_probs has shape [T, V+1]. Returns a list of token IDs (no blanks).
# beam_size is the number of prefixes retained between time steps.
def ctc_prefix_beam_search(
    log_probs: np.ndarray,
    blank: int,
    beam_size: int = 10,
) -> list[int]:
    raise NotImplementedError("TODO 6: implement prefix beam search")
