# Exercise 5 — Convert your CTC model to RNN-T

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwkendall/parakeet-study/blob/main/exercises/05-convert-to-rnnt/explore.ipynb)

> ▶️ **Run on a free GPU.** Open the `explore.ipynb` companion in Colab (then **Runtime → Change runtime type → GPU**); it clones the repo and runs these scripts step by step. See also [`../CLOUD_GPU_SETUP.md`](../CLOUD_GPU_SETUP.md).


**Time budget:** ~1 day on the same GPU as exercise 4.
**Prerequisites:** exercise 4 completed (you should have a checkpoint at `../04-train-fastconformer-ctc/lightning_logs/...`).
**Builds:** the same encoder + a prediction network + a joint network + RNN-T loss. Same data, same encoder, different head.

> **No local GPU?** See [`../CLOUD_GPU_SETUP.md`](../CLOUD_GPU_SETUP.md). This exercise is short enough for Colab free-tier, but the persistent-storage friction (needing the Ex 4 checkpoint mounted) makes RunPod (~$2) the smoother path.

The point of this exercise is to feel — concretely, on the same data — what changes when you swap the CTC head for an RNN-T decoder. You will see:

1. The model gets ~1–2 points of WER better (the language-model effect of the prediction network).
2. Training is ~2–3× slower per step (the joint network is large).
3. Greedy decoding is ~2× slower (autoregressive emission loop).
4. Memory is ~3–4× higher (the `[B, T, U, V]` joint tensor).

These are the same tradeoffs you'll see at Parakeet scale, just smaller.

## Step 0 — sanity check that exercise 4 finished

```bash
ls ../04-train-fastconformer-ctc/lightning_logs/*/checkpoints/last.ckpt
ls ../04-train-fastconformer-ctc/tokenizer/tokenizer.model
ls ../04-train-fastconformer-ctc/data/train-clean-100/train_clean_100.json
```

If any of these are missing, finish exercise 4 first.

## Step 1 — read the new config (15 minutes)

Open [`conf/fastconformer_rnnt_small.yaml`](./conf/fastconformer_rnnt_small.yaml). The encoder block is identical to exercise 4's. The decoder block is replaced with:

```yaml
decoder:
  _target_: nemo.collections.asr.modules.RNNTDecoder
  prednet:
    pred_hidden: 320         # LSTM hidden size — chapter 22
    pred_rnn_layers: 1
    t_max: null
    dropout: 0.1

joint:
  _target_: nemo.collections.asr.modules.RNNTJoint
  jointnet:
    joint_hidden: 320        # the tanh-projection dim
    activation: tanh
    dropout: 0.1
  fused_batch_size: 16        # subdivide joint computation to bound memory
                              # chapter 22: "this is why fused_batch_size exists"

loss:
  _target_: nemo.collections.asr.losses.RNNTLossNumba    # CUDA forward-backward
  loss_name: warp_rnnt
```

**Questions to answer before running**:

1. The joint tensor at each step is `[B, T, U+1, V+1]`. For your small model with `B=16`, `T≈100` (after 8× subsampling of 8 sec audio), `U≈40`, `V=1024+1`, what's the size in bf16? Why does `fused_batch_size: 16` help?
2. Why does `pred_hidden: 320` not need to equal `d_model: 256`? Where do they meet?
3. RNN-T training has the *exposure bias* discussed in chapter 22. Is it being addressed anywhere in this config? (Answer: no, not for this small model. NeMo has alignment-restricted RNN-T for it, but we keep things simple here.)

## Step 2 — initialise from your CTC checkpoint (10 minutes)

Encoder weights should transfer from the CTC model — same architecture, same input distribution. The script `init_from_ctc.py` does this:

```bash
python init_from_ctc.py \
    --ctc_checkpoint ../04-train-fastconformer-ctc/lightning_logs/version_0/checkpoints/last.ckpt \
    --rnnt_config conf/fastconformer_rnnt_small.yaml \
    --output rnnt_init.nemo
```

What it does:
1. Loads the CTC `.ckpt`.
2. Builds the RNN-T model fresh from the YAML config.
3. Copies all `encoder.*` weights from the CTC model into the RNN-T model.
4. Leaves `decoder.*` and `joint.*` randomly initialised.
5. Saves the result as a `.nemo` checkpoint, ready for fine-tuning.

You should see a log line like:
```
Copied 184 encoder tensors (~30M params). 56 new tensors in decoder+joint (~2M params) left at random init.
```

## Step 3 — fine-tune (overnight)

```bash
python train.py \
    --config-path=conf \
    --config-name=fastconformer_rnnt_small \
    model.train_ds.manifest_filepath=../04-train-fastconformer-ctc/data/train-clean-100/train_clean_100.json \
    model.validation_ds.manifest_filepath=../04-train-fastconformer-ctc/data/dev-clean/dev_clean.json \
    model.tokenizer.dir=../04-train-fastconformer-ctc/tokenizer \
    +init_from_nemo_model=rnnt_init.nemo \
    trainer.max_steps=15000     # half as many steps as ex 4: encoder is pre-trained
```

Watch the loss. You should see it drop rapidly from the CTC starting point, because the encoder is already useful — only the new decoder + joint need to learn.

| Step | Train loss | Val WER |
|------|-----------|---------|
| 0    | random    | ~95% (decoder is random) |
| 2k   | ~12       | ~30% |
| 5k   | ~6        | ~10% |
| 10k  | ~4        | ~7%  |
| 15k  | ~3        | ~6%  |

If the loss explodes early, check `fused_batch_size` (try lowering to 4) and `accumulate_grad_batches` (try higher).

## Step 4 — evaluate (5 minutes)

```bash
python evaluate.py \
    --checkpoint lightning_logs/version_0/checkpoints/last.ckpt \
    --manifest ../04-train-fastconformer-ctc/data/test-clean/test_clean.json
```

Expect ~6% WER vs the ~8% you got with CTC on the same encoder. That ~2-point delta is the prediction network earning its keep — those are mistakes CTC made because it couldn't condition on the previous word.

## Step 5 — measure decoding speed (10 minutes)

```bash
python benchmark.py \
    --ctc_checkpoint ../04-train-fastconformer-ctc/lightning_logs/version_0/checkpoints/last.ckpt \
    --rnnt_checkpoint lightning_logs/version_0/checkpoints/last.ckpt \
    --manifest ../04-train-fastconformer-ctc/data/test-clean/test_clean.json
```

This loops over the test set, decoding both models, and reports:
- RTFx (real-time factor — audio seconds per wall-clock second)
- Tokens emitted per joint-network call

You should see CTC at ~50× RTFx faster than RNN-T greedy, because CTC argmax is one matmul per frame and RNN-T runs a prediction-net + joint-net per (potentially many) emissions per frame. Exercise 6 (TDT) closes this gap.

## Reflection

1. The encoder weights transferred without modification. Why did this work? Could you do the same trick between RNN-T and TDT (exercise 6)?
2. Look at the WER breakdown: which classes of errors did CTC make that RNN-T fixed? (Look at the example transcriptions in `evaluate.py`'s output.) Hint: homophones, repeated content words, function words.
3. Real Parakeet uses `pred_hidden: 640` and a 24-layer FastConformer XL encoder. What's the ratio of prediction-net params to encoder params? Why is the prednet small (think autoregressive decoder vs encoder)?

## Extension problems

See [`notes.md`](./notes.md):

- Run beam search instead of greedy and quantify the WER/speed tradeoff.
- Train with FastEmit (Yu et al. 2021) to reduce emission latency.
- Try `alignment_restricted_rnnt` from NeMo to dramatically reduce training memory.
- Add stateful RNN-T greedy decoding that batches across utterances.
