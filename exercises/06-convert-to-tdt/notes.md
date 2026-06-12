# Exercise 6 — extension problems

### 1. Hyperparameter sweep on σ and ω (medium)
Run training (from scratch or fine-tuned) with a grid:
- σ ∈ {0.0, 0.02, 0.05, 0.1}
- ω ∈ {0.0, 0.05, 0.1, 0.2}

Plot a heatmap of WER and average predicted `d`. You'll see σ=0 with ω=0 trains an unstable model (durations collapse to 0 or 1); σ=0.1 may over-skip. The TDT paper's defaults (σ ≈ 0.02–0.05, ω ≈ 0.1) tend to be near-optimal.

### 2. Sparse durations (medium)
Try `durations: [0, 1, 2, 4, 8]` and `durations: [0, 1, 2, 3]`. Discuss the tradeoff: larger durations mean fewer joint calls (faster) but coarser timestamps and potential over-skipping. NeMo's `[0,1,2,3,4]` default is empirically a good compromise.

### 3. CUDA-Graphs label-looping decode (hard)
Switch the decoding strategy to `label_looping_cuda_graphs` (NeMo ≥ 1.22). Re-run `benchmark_tdt.py`. You should see an additional 2–3× speedup over `greedy_batch`, reproducing the chapter 26 "speed of light" result. Read `nemo/collections/asr/parts/submodules/tdt_loop_labels_computer.py` to see exactly what was captured.

### 4. Profile the joint network (medium)
Run inference with `torch.cuda.profiler` (or `torch.profiler`). Confirm that:
- The token logit head is one matmul of size `[B, hidden, V+1]`.
- The duration logit head is one matmul of size `[B, hidden, |D|]` — i.e. ~|D|/(V+1) ≈ 0.5% of the cost.
- The autoregressive prednet step dominates over the joint when `V` is small.

This is why TDT's speedup comes entirely from reducing the *number* of joint calls (via frame-skipping), not from a cheaper-per-call joint.

### 5. Error pattern analysis (medium)
Tabulate confusion errors for RNN-T and TDT on the same test set. TDT tends to make slightly more *deletion* errors (over-skipping a quiet syllable) and slightly more *length-mismatch* errors. Hypothesise why each happens.

### 6. Capstone: full small Parakeet (hard)
Scale your model to 24 layers × 1024 d_model × 8 heads (the real Parakeet 0.6B config). You'll need an 80 GB A100 (or multi-GPU with FSDP). Train on `train-clean-360` (360 hours) for a week. Compare to the published `nvidia/parakeet-tdt-0.6b-v2` checkpoint on `test-clean`. If you can get within 1.5 points of WER you've reproduced the Parakeet architecture at small data scale.
