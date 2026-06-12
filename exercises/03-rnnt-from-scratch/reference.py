"""Exercise 3 — reference solution."""
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


def extract_blank_and_label_log_probs(joint_log_probs, target, blank):
    T = joint_log_probs.shape[0]
    U = len(target)
    log_p_blank = joint_log_probs[:, :, blank]  # [T, U+1]
    log_p_target = np.zeros((T, U), dtype=joint_log_probs.dtype)
    for u in range(U):
        log_p_target[:, u] = joint_log_probs[:, u, target[u]]
    return log_p_blank, log_p_target


def rnnt_forward(log_p_blank: np.ndarray, log_p_target: np.ndarray) -> np.ndarray:
    """
    Forward variable. Convention:
      - log_p_blank:  [T, U+1]   joint_log_probs[t, u, blank]
      - log_p_target: [T, U]     joint_log_probs[t, u, target[u]]
      - log_alpha:    [T+1, U+1] log_alpha[t, u] = log prob of being at lattice
                                 cell (t, u) = consumed t frames, emitted u tokens.
      - log_alpha[T, U] is the full loss target (P(y|x), no extra blank step).

    Transitions reaching (t, u):
      - From (t-1, u) by emitting blank using joint[t-1, u, blank]
      - From (t, u-1) by emitting target token using joint[t, u-1, target[u-1]]
        (only valid when t < T -- there is no joint frame T)
    """
    T, U_plus_1 = log_p_blank.shape
    U = U_plus_1 - 1
    log_alpha = np.full((T + 1, U + 1), NEG_INF, dtype=np.float64)
    log_alpha[0, 0] = 0.0

    # Top row (u=0): only blank transitions
    for t in range(1, T + 1):
        log_alpha[t, 0] = log_alpha[t - 1, 0] + log_p_blank[t - 1, 0]

    # Left column (t=0): only target-emission transitions
    for u in range(1, U + 1):
        log_alpha[0, u] = log_alpha[0, u - 1] + log_p_target[0, u - 1]

    # Interior
    for t in range(1, T + 1):
        for u in range(1, U + 1):
            blank_term = log_alpha[t - 1, u] + log_p_blank[t - 1, u]
            if t < T:
                tok_term = log_alpha[t, u - 1] + log_p_target[t, u - 1]
                log_alpha[t, u] = logsumexp(blank_term, tok_term)
            else:
                # at t == T no joint frame exists, so no token emissions allowed.
                log_alpha[t, u] = blank_term

    return log_alpha


def rnnt_backward(log_p_blank: np.ndarray, log_p_target: np.ndarray) -> np.ndarray:
    """
    Backward variable. Convention:
      log_beta[t, u] = log prob of reaching (T, U) from (t, u).
      Transitions leaving (t, u):
        - emit blank: (t, u) -> (t+1, u) using log_p_blank[t, u]   (valid when t < T)
        - emit target token: (t, u) -> (t, u+1) using log_p_target[t, u]  (valid when t < T, u < U)
    """
    T, U_plus_1 = log_p_blank.shape
    U = U_plus_1 - 1
    log_beta = np.full((T + 1, U + 1), NEG_INF, dtype=np.float64)
    log_beta[T, U] = 0.0

    # Rightmost column (u=U): only blanks
    for t in range(T - 1, -1, -1):
        log_beta[t, U] = log_p_blank[t, U] + log_beta[t + 1, U]

    # Bottom row at t=T: cannot emit further (no frames left, and tokens need a frame too).
    # By convention, log_beta[T, u<U] = NEG_INF (unreachable from (T, u) -- we'd
    # need to emit target tokens but there is no joint frame). So leave as NEG_INF.

    # Interior
    for t in range(T - 1, -1, -1):
        for u in range(U - 1, -1, -1):
            blank_term = log_p_blank[t, u] + log_beta[t + 1, u]
            tok_term = log_p_target[t, u] + log_beta[t, u + 1]
            log_beta[t, u] = logsumexp(blank_term, tok_term)

    return log_beta


def rnnt_loss_and_grad(
    joint_log_probs: np.ndarray, target: Sequence[int], blank: int,
) -> tuple[float, np.ndarray]:
    T, U_plus_1, V_plus_1 = joint_log_probs.shape
    U = U_plus_1 - 1
    assert len(target) == U

    log_p_blank, log_p_target = extract_blank_and_label_log_probs(joint_log_probs, target, blank)
    log_alpha = rnnt_forward(log_p_blank, log_p_target)
    log_beta = rnnt_backward(log_p_blank, log_p_target)

    log_P = log_alpha[T, U]
    loss = -log_P

    grad = np.zeros_like(joint_log_probs)

    # Blank emissions contribute at every (t, u) for t in [0, T):
    #   transition (t, u) --blank--> (t+1, u)
    for t in range(T):
        for u in range(U + 1):
            log_g = log_alpha[t, u] + log_p_blank[t, u] + log_beta[t + 1, u] - log_P
            if log_g > -700:
                grad[t, u, blank] = -np.exp(log_g)

    # Target emissions contribute at every (t, u) for u in [0, U), t in [0, T):
    #   transition (t, u) --y_{u+1}--> (t, u+1)
    for t in range(T):
        for u in range(U):
            log_g = log_alpha[t, u] + log_p_target[t, u] + log_beta[t, u + 1] - log_P
            if log_g > -700:
                grad[t, u, target[u]] = -np.exp(log_g)

    return loss, grad


def rnnt_greedy_decode(joint_log_probs, blank, max_symbols_per_step: int = 10):
    T = joint_log_probs.shape[0]
    U_max = joint_log_probs.shape[1] - 1
    t = u = 0
    hyp = []
    while t < T and u <= U_max:
        symbols_this_t = 0
        while True:
            v = int(np.argmax(joint_log_probs[t, u]))
            if v == blank:
                t += 1
                break
            hyp.append(v)
            u += 1
            symbols_this_t += 1
            if u > U_max:
                return hyp
            if symbols_this_t >= max_symbols_per_step:
                t += 1
                break
    return hyp
