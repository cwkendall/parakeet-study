# Exercise 4 — extension problems

### 1. SpecAugment ablation (easy)
The base config has `spec_augment` enabled. Re-run training with `spec_augment` removed. Plot loss curves and final WER. Expect ~2× faster initial drop but much worse final WER without SpecAugment — it's one of the highest-ROI training tricks in ASR.

### 2. InterCTC (medium)
Add two auxiliary CTC heads at layers 4 and 8 of the encoder, each with its own loss weighted by `0.3 * (layer_idx / total_layers)`. Quantify the convergence speedup (steps-to-X-WER) vs the baseline. Lee & Watanabe 2021 reports 1.5–2× speedup on similar configs.

### 3. Subsampling stem ablation (medium)
The default uses `subsampling: dw_striding` (FastConformer's depthwise-separable stem). Try `subsampling: striding` (plain stride-2 convs, the original Conformer stem). Measure:
- training step time
- parameter count of the stem
- final WER

Expect ~10% slower training and a tiny WER change. This is the FastConformer paper's claim made concrete.

### 4. Shallow LM fusion (medium)
Train a 4-gram KenLM on the LibriSpeech training transcripts:
```bash
kenlm/bin/lmplz -o 4 < transcripts.txt > lm.arpa
kenlm/bin/build_binary lm.arpa lm.bin
```
Decode your checkpoint with NeMo's `pyctcdecode` integration using shallow fusion (`lm_weight=0.5`, `length_norm=0.3`). Quantify the WER drop. Typical: 7.8% → 6.5%.

### 5. Fine-tune a public model (hard, requires good GPU)
Download `nvidia/parakeet-ctc-1.1b` from HuggingFace. Fine-tune on `train-clean-100` for 5k steps at LR `1e-5`. Compare WER to the 1.1B model out-of-the-box and to your from-scratch 30M model. This is the workflow most production ASR deployments actually follow.

### 6. Profile training memory (easy)
Add `torch.cuda.memory._record_memory_history(enabled='all')` before `trainer.fit()` and dump the snapshot after one step. What dominates the memory? (Spoiler: encoder activations during backward.)

### 7. Streaming-style inference on a long file (medium)
Take a long audio file (e.g. a 10-min podcast clip). Run `model.transcribe([path])` once and observe the latency and memory. Then break the file into 30-second overlapping chunks and run `transcribe` on each. Compare WER and look at how the hypotheses merge at chunk boundaries — this is the failure mode you fix with cache-aware streaming models in production.
