# Exercise 5 — extension problems

### 1. Beam search vs greedy (medium)
Change `decoding.strategy` to `beam` and `decoding.beam.beam_size: 4`. Re-run `evaluate.py`. Plot WER and decoding time at beam sizes 1, 2, 4, 8. You should see WER improve by ~0.1–0.3 at beam 4 with a 2–4× decoding slowdown.

### 2. FastEmit (medium)
FastEmit (Yu et al. 2021) is a regularisation trick that reduces emission latency by adding a small penalty for blank emissions. In the config, set:
```yaml
loss:
  warprnnt_numba_kwargs:
    fastemit_lambda: 0.001
```
Re-train. Measure both WER and *first-token latency* (how many encoder frames before the first non-blank token is emitted on average). Lower is better for streaming applications.

### 3. Alignment-restricted RNN-T (hard)
Use NeMo's `alignment_restricted_rnnt` config. It prunes the lattice to a band around the forced alignment, cutting joint memory by ~10×. Try training with `fused_batch_size=128` (much higher than before) and measure the speedup.

### 4. Stateful streaming decoder (capstone)
Modify `EncDecRNNTBPEModel.transcribe` to consume audio in 320-ms chunks, maintaining the prediction-net hidden state and any cached encoder state across chunks. Compare to the non-streaming inference on the same files and observe the end-to-end emission latency.

### 5. Joint network ablations (medium)
Try `joint_hidden: 160` and `joint_hidden: 640`. Plot the parameter count, training memory, and WER at each. The sweet spot for small models is usually `≈ pred_hidden`.

### 6. KenLM shallow fusion (medium)
Same as exercise 4: train a KenLM 4-gram on transcripts and apply shallow fusion. RNN-T's gain from external LM is smaller than CTC's (because the prednet already models output dependencies) — quantify how much smaller.
