"""
Exercise 2 — reference solution.
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


def extend_target(target: Sequence[int], blank: int) -> list[int]:
    out = [blank]
    for t in target:
        out.append(int(t))
        out.append(blank)
    return out


def ctc_forward(log_probs: np.ndarray, target: Sequence[int], blank: int) -> np.ndarray:
    T = log_probs.shape[0]
    ext = extend_target(target, blank)
    S = len(ext)

    log_alpha = np.full((T, S), NEG_INF, dtype=np.float64)
    log_alpha[0, 0] = log_probs[0, blank]
    if S > 1:
        log_alpha[0, 1] = log_probs[0, ext[1]]

    for t in range(1, T):
        for s in range(S):
            terms = [log_alpha[t - 1, s]]
            if s >= 1:
                terms.append(log_alpha[t - 1, s - 1])
            # We can only jump 2 (skip a blank) when:
            #   * we're past the boundary (s >= 2)
            #   * ext[s] is a label (not blank)
            #   * ext[s] != ext[s-2]  (otherwise we'd merge identical labels)
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                terms.append(log_alpha[t - 1, s - 2])
            log_alpha[t, s] = logsumexp(*terms) + log_probs[t, ext[s]]
    return log_alpha


def ctc_backward(log_probs: np.ndarray, target: Sequence[int], blank: int) -> np.ndarray:
    T = log_probs.shape[0]
    ext = extend_target(target, blank)
    S = len(ext)

    log_beta = np.full((T, S), NEG_INF, dtype=np.float64)
    log_beta[T - 1, S - 1] = 0.0
    if S >= 2:
        log_beta[T - 1, S - 2] = 0.0

    for t in range(T - 2, -1, -1):
        for s in range(S):
            terms = [log_beta[t + 1, s] + log_probs[t + 1, ext[s]]]
            if s + 1 < S:
                terms.append(log_beta[t + 1, s + 1] + log_probs[t + 1, ext[s + 1]])
            if s + 2 < S and ext[s + 2] != blank and ext[s] != ext[s + 2]:
                terms.append(log_beta[t + 1, s + 2] + log_probs[t + 1, ext[s + 2]])
            log_beta[t, s] = logsumexp(*terms)
    return log_beta


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return logits - m - np.log(e.sum(axis=-1, keepdims=True))


def ctc_loss_and_grad(
    logits: np.ndarray, target: Sequence[int], blank: int,
) -> tuple[float, np.ndarray]:
    T, V = logits.shape
    log_probs = _log_softmax(logits)
    probs = np.exp(log_probs)

    ext = extend_target(target, blank)
    S = len(ext)

    log_alpha = ctc_forward(log_probs, target, blank)
    log_beta = ctc_backward(log_probs, target, blank)

    log_P = logsumexp(log_alpha[T - 1, S - 1], log_alpha[T - 1, S - 2])
    loss = -log_P

    # Marginal occupancy: γ(t, s) = α(t, s) * β(t, s) / P(y|x), but we only need
    # the per-label sum at each frame.
    grad = np.zeros_like(logits)
    for t in range(T):
        # marginal per label k: sum over s where ext[s] == k of exp(α + β - logP)
        for k in range(V):
            occ_terms = []
            for s in range(S):
                if ext[s] == k:
                    occ_terms.append(log_alpha[t, s] + log_beta[t, s] - log_P)
            if occ_terms:
                occupancy = np.exp(logsumexp(*occ_terms))
            else:
                occupancy = 0.0
            grad[t, k] = probs[t, k] - occupancy

    return loss, grad


def ctc_greedy_decode(log_probs: np.ndarray, blank: int) -> list[int]:
    raw = np.argmax(log_probs, axis=1).tolist()
    collapsed = [raw[0]]
    for c in raw[1:]:
        if c != collapsed[-1]:
            collapsed.append(c)
    return [c for c in collapsed if c != blank]


def ctc_prefix_beam_search(
    log_probs: np.ndarray, blank: int, beam_size: int = 10,
) -> list[int]:
    T, V = log_probs.shape
    probs = np.exp(log_probs)

    # beam: dict mapping prefix-tuple to (p_blank, p_nonblank) in linear space
    beam: dict[tuple, tuple[float, float]] = {(): (1.0, 0.0)}

    for t in range(T):
        new_beam: dict[tuple, tuple[float, float]] = {}
        for prefix, (pb, pnb) in beam.items():
            for s in range(V):
                p = probs[t, s]
                if s == blank:
                    new_pb, new_pnb = new_beam.get(prefix, (0.0, 0.0))
                    new_pb += (pb + pnb) * p
                    new_beam[prefix] = (new_pb, new_pnb)
                else:
                    ext_prefix = prefix + (s,)
                    new_pb_e, new_pnb_e = new_beam.get(ext_prefix, (0.0, 0.0))
                    if len(prefix) > 0 and s == prefix[-1]:
                        # repeated label: only the pb of the *previous* state contributes
                        new_pnb_e += pb * p
                        # and we can stay on the same prefix (no extension) by repeating without blank
                        new_pb_s, new_pnb_s = new_beam.get(prefix, (0.0, 0.0))
                        new_pnb_s += pnb * p
                        new_beam[prefix] = (new_pb_s, new_pnb_s)
                    else:
                        new_pnb_e += (pb + pnb) * p
                    new_beam[ext_prefix] = (new_pb_e, new_pnb_e)
        beam = dict(
            sorted(new_beam.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:beam_size]
        )
    best = max(beam.items(), key=lambda kv: kv[1][0] + kv[1][1])[0]
    return list(best)
