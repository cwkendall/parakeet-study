# Exercise 6 — Convert your RNN-T model to TDT

**Time budget:** ~1 weekend on the same GPU as exercises 4 & 5.
**Prerequisites:** exercise 5 completed.
**Builds:** the TDT variant of your transducer — the exact decoder family Parakeet-TDT uses. By the end you can quote, from your own measurements, the speedup TDT gives over RNN-T at equal WER.

> **No local GPU?** See [`../CLOUD_GPU_SETUP.md`](../CLOUD_GPU_SETUP.md). For the full Ex 4 → 5 → 6 sequence, RunPod RTX 4090 (~$9 total) is the path of least friction since you need warm checkpoints between stages.

This is the capstone: it produces a working analogue of Parakeet-TDT at small scale, and forces you to confront the loss-engineering details that the deep-dive only sketches (logit under-normalization σ, sampled loss ω, duration prior, frame-skipping behaviour at decode).

## Step 0 — sanity check

```bash
ls ../05-convert-to-rnnt/lightning_logs/*/checkpoints/last.ckpt
```

You'll initialise the TDT model from this RNN-T checkpoint, the same way you initialised the RNN-T model from the CTC one in exercise 5. The encoder transfers cleanly; the decoder transfers nearly cleanly (the prednet is identical); only the joint network gets reshaped.

## Step 1 — read the new config (20 minutes)

Open [`conf/fastconformer_tdt_small.yaml`](./conf/fastconformer_tdt_small.yaml). Key differences vs RNN-T:

```yaml
joint:
  _target_: nemo.collections.asr.modules.RNNTJoint
  num_extra_outputs: 5                # |D| = 5 durations: {0,1,2,3,4}
  jointnet:
    joint_hidden: 320
    activation: tanh
  fused_batch_size: 16

loss:
  _target_: nemo.collections.asr.losses.RNNTLossNumba
  loss_name: tdt                       # the magic word
  durations: [0, 1, 2, 3, 4]           # |D| = 5
  sigma: 0.05                          # logit under-normalisation — chapter 23
  omega: 0.1                           # sampled-loss probability — chapter 23

decoding:
  strategy: greedy_batch
  greedy:
    max_symbols: 10
    durations: [0, 1, 2, 3, 4]
```

**The joint network has 5 extra output units** beyond the V+1 token-vocab outputs. These 5 are the duration logits, softmaxed separately from the token logits. NeMo's `TDTLossNumba` knows the convention: the final 5 outputs are durations, the others are tokens. You **must** keep `num_extra_outputs` in the joint and `durations` in the loss in sync.

**Questions to answer before running**:

1. The token + duration logits share the joint network's penultimate hidden layer (the `tanh(W_e f + W_p g + b)` projection). The split into the two output heads is at the final linear layer. Why is this factorisation valid even though `P(v, d) = P_T(v) * P_D(d)` is assumed independent? (Hint: the assumption is independence *conditional on (t, u)*; the shared upstream features capture that conditioning.)
2. `sigma: 0.05` subtracts 0.05 from each token log-probability in the TDT loss. What total bias does this place on a 30-token transcript? (Answer: 1.5 nats per training example, biasing toward fewer transitions.)
3. `omega: 0.1` means 10% of training steps use the RNN-T loss (durations collapsed to `[1]`). Why is this useful? (Hint: keeps the model robust when the duration head mispredicts.)

## Step 2 — initialise from your RNN-T checkpoint (10 minutes)

The transplant is slightly more elaborate than CTC→RNN-T because the joint's output layer changes shape. The script handles it:

```bash
python init_from_rnnt.py \
    --rnnt_checkpoint ../05-convert-to-rnnt/lightning_logs/version_0/checkpoints/last.ckpt \
    --tdt_config conf/fastconformer_tdt_small.yaml \
    --tokenizer_dir ../04-train-fastconformer-ctc/tokenizer \
    --output tdt_init.nemo
```

What it does:
1. Loads the RNN-T `.ckpt`.
2. Builds a fresh TDT model with 5 extra joint outputs.
3. Copies encoder + decoder (prednet) weights verbatim.
4. Copies the joint's `joint_pre` (the `tanh(...)` layer) verbatim.
5. Copies the joint's `joint_out` token logit weights into the first `V+1` rows of the new joint's output layer; the last 5 rows (durations) are left random.

You should see:
```
Encoder:  184 tensors copied (~30M params).
Prednet:  9   tensors copied (~0.5M params).
Joint pre: 2 tensors copied (~0.2M params).
Joint out: V+1=1025 token rows copied; 5 duration rows left random.
```

## Step 3 — fine-tune (overnight)

```bash
python train.py \
    --config-path=conf \
    --config-name=fastconformer_tdt_small \
    model.train_ds.manifest_filepath=../04-train-fastconformer-ctc/data/train-clean-100/train_clean_100.json \
    model.validation_ds.manifest_filepath=../04-train-fastconformer-ctc/data/dev-clean/dev_clean.json \
    model.tokenizer.dir=../04-train-fastconformer-ctc/tokenizer \
    +init_from_nemo_model=tdt_init.nemo \
    trainer.max_steps=15000
```

Watch the training. You should see:

- **Early steps (0-1k):** loss is *higher* than the RNN-T starting point because the duration head is random and producing chaotic frame-skips. This is normal.
- **Steps 1k-3k:** loss drops rapidly as the duration head learns; WER may briefly worsen.
- **Steps 5k onwards:** loss and WER converge to within ~0.3 of the RNN-T values.

Expected trajectory:
| Step | Loss | Val WER |
|------|------|---------|
| 0    | ~10  | ~50%    |
| 2k   | ~5   | ~12%    |
| 5k   | ~3   | ~7%     |
| 10k  | ~2.5 | ~6.5%   |
| 15k  | ~2.3 | ~6.2%   |

## Step 4 — evaluate (5 minutes)

```bash
python evaluate.py \
    --checkpoint lightning_logs/version_0/checkpoints/last.ckpt \
    --manifest ../04-train-fastconformer-ctc/data/test-clean/test_clean.json
```

WER should be within 0.3 of RNN-T — TDT exchanges some WER headroom for inference speed. The big payoff is in step 5.

## Step 5 — measure inference speedup (the punchline, 15 minutes)

```bash
python benchmark_tdt.py \
    --rnnt_checkpoint ../05-convert-to-rnnt/lightning_logs/version_0/checkpoints/last.ckpt \
    --tdt_checkpoint  lightning_logs/version_0/checkpoints/last.ckpt \
    --manifest ../04-train-fastconformer-ctc/data/test-clean/test_clean.json \
    --batch_size 16
```

Reports for both models:
- **RTFx** (real-time factor)
- **Joint calls per second of audio** (the metric that explains the speedup)
- **Average duration `d` predicted** (TDT only) — for typical speech this should be ~2.3
- **WER** (so the speedup isn't measured at the cost of accuracy)

Expected output:

```
RNN-T:   RTFx 180   joint calls/audio-sec: 320   WER: 6.1%
TDT:     RTFx 480   joint calls/audio-sec: 130   avg d: 2.4   WER: 6.3%
Speedup: 2.7×
```

That ~2.7× is the headline TDT speedup. Real Parakeet-TDT-0.6B-v2 reports 2.82× over Parakeet-RNNT-0.6B at NeMo's published benchmark — your small model reproduces the same architectural advantage.

## Step 6 — extract timestamps (15 minutes)

TDT's other selling point: word-level timestamps fall out of the durations. Run:

```bash
python extract_timestamps.py \
    --checkpoint lightning_logs/version_0/checkpoints/last.ckpt \
    --audio path/to/any/16khz/clip.wav
```

This calls `model.transcribe(..., return_hypotheses=True, timestamps=True)` and prints each word with its predicted `(start_sec, end_sec)`. Sanity check: the start times should be monotonically increasing, gaps should match silences, total duration should approximately match the audio length.

Limitation per chapter 23: resolution is bounded by the encoder frame rate (~80 ms), so word boundaries within an encoder frame aren't resolvable. Compare against a forced-alignment tool (e.g. `nemo_forced_aligner` or `aeneas`) on the same audio to see the typical mean absolute boundary error.

## Reflection

1. The TDT joint had 5 extra outputs. By what fraction did total parameters increase from RNN-T? (Approximately 5 / (V+1) ≈ 0.5%.) The inference speedup is ~3×. Where did that come from given the model is almost the same size?
2. You initialised the duration logits at random and the model still converged. What would happen if you initialised them with a strong prior, say P(d=1) = 0.7? Try it (it's a one-line change in `init_from_rnnt.py`) and see if it speeds up convergence.
3. TDT's biggest weakness is *over-skipping* in noisy or fast speech: predicting `d=4` when only `d=1` was warranted, missing tokens. Find an audio clip with rapid speech and confirm whether your small model suffers from this. How might you train against it? (Hint: more diverse training data, lower `sigma`, restrict `D` to `[0,1,2,3]`.)

## Extension problems

See [`notes.md`](./notes.md):

- Hyperparameter sweep on `sigma` and `omega`. Plot a heatmap of WER and avg `d`.
- Try `durations: [0,1,2,3,4,6,8]` (sparser large jumps) and measure.
- Replace `decoding.strategy: greedy_batch` with `decoding.strategy: label_looping_cuda_graphs` and quantify the additional speedup. This is the chapter-26 stack.
- Profile the joint network at decode time and confirm that the duration-head is a single matmul per frame — i.e. essentially free.
- Compare TDT and RNN-T error patterns: which classes of errors does TDT make that RNN-T doesn't, and vice versa?
