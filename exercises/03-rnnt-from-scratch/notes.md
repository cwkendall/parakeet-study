# Exercise 3 — extension problems

### 1. Numerical gradient check (easy)
Independent of `torchaudio.functional.rnnt_loss`, perturb each entry of `joint_log_probs` by ±ε and confirm your analytical gradient matches `(L(+ε) - L(-ε)) / (2ε)` to within ~1e-4. This catches bugs that pass the torchaudio cross-check by coincidence.

### 2. Beam-search variants (hard)
Implement the three transducer beam-search variants mentioned in chapter 26:
- **TSD** — time-synchronous decoding: at each frame, expand all beams by every possible (blank or token) action.
- **ALSD** — alignment-length-synchronous decoding: group beams by alignment length to keep GPU utilisation high.
- **MAES** — modified adaptive expansion search: adaptive per-beam expansion limit.

Compare WER on synthetic data with varying beam sizes (1, 2, 4, 8). Plot the speed/quality tradeoff.

### 3. Pruned RNN-T (k2-style) (hard)
Only compute α(t, u) inside a band around the diagonal alignment. Concretely, find the rough alignment first (e.g. by a fast CTC pass) and then only update cells within ±K of it. Measure memory savings — should be roughly T/(2K+1)×.

### 4. Build a full end-to-end transducer (capstone)
Combine exercise 2's TinyEncoder with:
- a one-layer LSTM prediction net
- a small joint network: `joint(f_t, g_u) = W_o · tanh(W_e f_t + W_p g_u + b)`
- your RNN-T loss

Train on the synthetic data from exercise 2 and verify it converges to higher accuracy than CTC alone. This is your tracer-bullet path to understanding the actual NeMo training loop.

### 5. Profile the joint network (medium)
For T=200, U=30, V=1024 (a typical short utterance), how big is the joint tensor in MB? In fp32, fp16, fp8? Compare to NeMo's `fused_batch_size` setting, which subdivides the batch through the joint network for memory reasons. Why is this needed?

### 6. Compare to CTC convergence (medium)
On the same synthetic data, train two models — one with your CTC loss, one with your RNN-T loss + a one-layer LSTM prednet. Plot loss curves. RNN-T usually trains slower at first but reaches lower final loss. Why?
