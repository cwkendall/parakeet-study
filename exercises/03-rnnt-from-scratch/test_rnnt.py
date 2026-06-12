"""Tests for exercise 3 — RNN-T from scratch."""
import numpy as np
import pytest
import torch

import torchaudio.functional as taF

from starter import (
    extract_blank_and_label_log_probs,
    rnnt_forward,
    rnnt_backward,
    rnnt_loss_and_grad,
    rnnt_greedy_decode,
)


def _make_joint(T, U, V, seed=0):
    """Return a [T, U+1, V+1] log-prob tensor (log-softmaxed along last axis)."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((T, U + 1, V + 1))
    m = logits.max(axis=-1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))


def _torchaudio_loss(joint_log_probs, target, blank):
    """torchaudio's rnnt_loss expects logits (not log-probs) of shape [B, T, U+1, V+1]."""
    logits = torch.from_numpy(joint_log_probs).float().unsqueeze(0).contiguous()
    tgt = torch.tensor([target], dtype=torch.int32)
    T = torch.tensor([joint_log_probs.shape[0]], dtype=torch.int32)
    U = torch.tensor([len(target)], dtype=torch.int32)
    return taF.rnnt_loss(logits, tgt, T, U, blank=blank, reduction="none").item()


def test_extract():
    T, U, V = 5, 3, 4
    blank = V  # blank = 4 (last index)
    target = [1, 2, 3]
    j = _make_joint(T, U, V)
    lpb, lpt = extract_blank_and_label_log_probs(j, target, blank)
    assert lpb.shape == (T, U + 1)
    assert lpt.shape == (T, U)
    for t in range(T):
        for u in range(U + 1):
            assert lpb[t, u] == j[t, u, blank]
        for u in range(U):
            assert lpt[t, u] == j[t, u, target[u]]


def test_forward_endpoint():
    T, U, V = 8, 3, 4
    blank = V
    target = [1, 2, 1]
    j = _make_joint(T, U, V, seed=11)
    lpb, lpt = extract_blank_and_label_log_probs(j, target, blank)

    log_alpha = rnnt_forward(lpb, lpt)
    # log P(y | x) = log_alpha[T, U]
    log_P_ours = log_alpha[T, U]

    neg_log_P_ref = _torchaudio_loss(j, target, blank)
    np.testing.assert_allclose(log_P_ours, -neg_log_P_ref, atol=1e-4)


def test_forward_backward_consistency():
    T, U, V = 7, 4, 3
    blank = V
    target = [1, 2, 0, 1]
    j = _make_joint(T, U, V, seed=22)
    lpb, lpt = extract_blank_and_label_log_probs(j, target, blank)

    log_alpha = rnnt_forward(lpb, lpt)
    log_beta = rnnt_backward(lpb, lpt)

    # The total log P should be obtainable from either end.
    log_P_fwd = log_alpha[T, U]
    log_P_bwd = log_beta[0, 0]
    np.testing.assert_allclose(log_P_fwd, log_P_bwd, atol=1e-5)


def test_loss_matches_torchaudio():
    T, U, V = 12, 5, 6
    blank = V
    target = [1, 4, 2, 3, 2]
    j = _make_joint(T, U, V, seed=33)
    loss_ours, _ = rnnt_loss_and_grad(j, target, blank)
    loss_ref = _torchaudio_loss(j, target, blank)
    np.testing.assert_allclose(loss_ours, loss_ref, atol=1e-4)


def test_gradient_matches_torchaudio():
    T, U, V = 8, 3, 4
    blank = V
    target = [1, 2, 3]

    # Use raw logits this time so we can compare gradients consistently
    rng = np.random.default_rng(44)
    logits = rng.standard_normal((T, U + 1, V + 1)).astype(np.float64)
    # log-softmax to make joint_log_probs
    m = logits.max(axis=-1, keepdims=True)
    log_probs = logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))

    loss_ours, grad_log_probs_ours = rnnt_loss_and_grad(log_probs, target, blank)

    # torchaudio: get gradient wrt logits, then convert to gradient wrt log-probs
    # via the chain rule. Easier route: compute torchaudio gradient wrt LOGITS,
    # and check ours via the same chain.
    logits_t = torch.from_numpy(logits).float().unsqueeze(0).contiguous().requires_grad_()
    tgt = torch.tensor([target], dtype=torch.int32)
    T_t = torch.tensor([T], dtype=torch.int32)
    U_t = torch.tensor([len(target)], dtype=torch.int32)
    loss_t = taF.rnnt_loss(logits_t, tgt, T_t, U_t, blank=blank, reduction="sum")
    loss_t.backward()
    grad_logits_pt = logits_t.grad.numpy()[0].astype(np.float64)

    # Convert our gradient-wrt-log-probs to gradient-wrt-logits.
    # log_softmax(logits) = logits - logsumexp(logits)
    # d log_softmax / d logits = I - softmax(logits)
    # So grad_logits = grad_log_probs - softmax(logits) * grad_log_probs.sum(-1, keepdims=True)
    probs = np.exp(log_probs)
    grad_logits_ours = grad_log_probs_ours - probs * grad_log_probs_ours.sum(axis=-1, keepdims=True)

    np.testing.assert_allclose(grad_logits_ours, grad_logits_pt, atol=1e-4)


def test_greedy_decode():
    # Construct a tiny joint where the greedy path emits "1, 2"
    # T=4, U+1=3 (so vocab is {blank=2, 0, 1, ...}; we set V=3 -> tokens are 0,1,2, blank=3)
    T, U_plus_1, V_plus_1 = 4, 3, 4
    blank = V_plus_1 - 1
    j = np.full((T, U_plus_1, V_plus_1), -10.0)
    # at (0,0) argmax should be token 1
    j[0, 0, 1] = 1.0
    # at (0,1) argmax should be token 2
    j[0, 1, 2] = 1.0
    # at (0,2) argmax should be blank -> advance t
    j[0, 2, blank] = 1.0
    # at (1, 2) blank
    j[1, 2, blank] = 1.0
    j[2, 2, blank] = 1.0
    j[3, 2, blank] = 1.0

    out = rnnt_greedy_decode(j, blank)
    assert out == [1, 2]
