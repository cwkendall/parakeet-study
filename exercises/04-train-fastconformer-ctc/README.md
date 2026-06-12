# Exercise 4 — Train a small FastConformer-CTC on LibriSpeech-clean

**Time budget:** ~1 weekend on a single 24 GB GPU (RTX 3090 / 4090 / A4000) or ~1 day on an A100.
**Prerequisites:** exercises 2 done; chapters 17–18, 21, 25 of the deep-dive read.
**Builds:** a real (small) Parakeet-family model that you have trained end-to-end. Target: ≤10% WER on `dev-clean`.

> **No local GPU?** See [`../CLOUD_GPU_SETUP.md`](../CLOUD_GPU_SETUP.md) for step-by-step Colab (free T4), Kaggle (free P100), or RunPod (~$5 paid) instructions. This exercise fits comfortably in Colab's free tier.

This is where the abstractions become concrete. Previously you've written CTC loss in NumPy on toy data. Now you'll use NeMo's framework (the same code path Parakeet uses) on real LibriSpeech audio, with a 30-million-parameter FastConformer encoder. At the end you will have a checkpoint that transcribes English speech with ~5–10% WER and, more importantly, you'll have read every line of the training config and know what it does.

This exercise is **recipe-following with deliberate, narrated detours** — you stop at each step and look at what's happening, rather than treating the YAML as a black box.

## Hardware and data check

```bash
# Confirm you have CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Free GPU memory required: ~16 GB for a comfortable run.
nvidia-smi
```

LibriSpeech is ~60 GB unzipped if you grab everything. For this exercise we only need:
- `train-clean-100` (~6 GB) — training set
- `dev-clean` (~340 MB) — validation
- `test-clean` (~340 MB) — final evaluation

## Step 0 — install (15 minutes)

```bash
cd exercises/04-train-fastconformer-ctc
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

`nemo_toolkit[asr]` pulls in a large dependency tree (PyTorch Lightning, sentencepiece, librosa, kaldi-style stuff, hydra). Coffee break.

## Step 1 — download LibriSpeech (45 minutes, network-limited)

NeMo's `scripts/dataset_processing/get_librispeech_data.py` is the canonical downloader:

```bash
python download_librispeech.py --data_root ./data --data_sets train-clean-100,dev-clean,test-clean
```

The provided `download_librispeech.py` wraps the NeMo helper and adds proxy-aware retries (this machine uses `http://127.0.0.1:9000`). At the end you'll have:

```
./data/
  train-clean-100/  ~6 GB
  dev-clean/        ~340 MB
  test-clean/       ~340 MB
  train-clean-100/train_clean_100.json    ← manifest the trainer reads
  dev-clean/dev_clean.json
  test-clean/test_clean.json
```

Each manifest line is one utterance:

```json
{"audio_filepath": "/abs/path/to/0001.flac", "duration": 6.5, "text": "the quick brown fox"}
```

## Step 2 — train a BPE tokenizer (5 minutes)

Parakeet uses SentencePiece BPE, not characters. Train a 1024-token vocab on the training transcripts:

```bash
python train_tokenizer.py \
    --manifest data/train-clean-100/train_clean_100.json \
    --vocab_size 1024 \
    --output_dir tokenizer
```

This script wraps `nemo.collections.common.tokenizers.sentencepiece_tokenizer.SentencePieceTokenizer.build_from_text()`. Open it and read what it does — it's just a thin wrapper around SentencePiece. You'll get `tokenizer/tokenizer.model` and `tokenizer/tokenizer.vocab`.

Sanity-check by tokenising a sentence:

```bash
python -c "
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='tokenizer/tokenizer.model')
print(sp.encode('the quick brown fox', out_type=str))
"
# Expected: ['▁the', '▁quick', '▁brown', '▁fox']
```

## Step 3 — inspect the config (30 minutes — DO NOT SKIP)

Open `conf/fastconformer_ctc_small.yaml`. This is a stripped-down version of NeMo's [`fast-conformer_ctc_bpe.yaml`](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml) tuned for a single-GPU training run.

Read it block by block and verify you understand each setting:

```yaml
model:
  preprocessor:
    _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
    sample_rate: 16000
    window_size: 0.025        # 25 ms — chapter 6
    window_stride: 0.01       # 10 ms hop — 100 frames/sec
    features: 80              # n_mels — chapter 7
    n_fft: 512

  encoder:
    _target_: nemo.collections.asr.modules.ConformerEncoder
    feat_in: 80
    n_layers: 12              # SMALL: real Parakeet has 24
    d_model: 256              # SMALL: real Parakeet has 1024
    subsampling: dw_striding  # the 3x stride-2 depthwise-separable stem
    subsampling_factor: 8     # the 8x in "8x subsampling"
    subsampling_conv_channels: 256
    self_attention_model: rel_pos      # Transformer-XL — chapter 15
    n_heads: 4                          # SMALL: real Parakeet has 8
    ff_expansion_factor: 4
    conv_kernel_size: 9                 # FastConformer change — chapter 18
    pos_emb_max_len: 5000

  decoder:
    _target_: nemo.collections.asr.modules.ConvASRDecoder
    feat_in: 256              # must match encoder d_model
    num_classes: -1           # set by the tokenizer (1024 + 1 for blank)
    vocabulary: ???

  loss:
    _target_: nemo.collections.asr.losses.CTCLoss

  optim:
    name: adamw
    lr: 2e-3
    weight_decay: 1e-3
    sched:
      name: NoamAnnealing     # warmup-then-1/sqrt(t) — chapter 25 style
      warmup_steps: 5000
      min_lr: 1e-5

trainer:
  devices: 1
  accelerator: gpu
  max_steps: 30000            # roughly 1 day on a single 3090; 30k is enough for <10% WER
  precision: bf16-mixed
  gradient_clip_val: 1.0
  accumulate_grad_batches: 4
```

**Questions to answer for yourself before running training** (each is a checkpoint that you actually understand the architecture):

1. The encoder is 12 layers × 256-d. What's the total parameter count? (Open `train.py` after instantiation and `print(sum(p.numel() for p in asr_model.encoder.parameters()))` to check.)
2. With `subsampling_factor: 8`, what is the encoder frame rate? (Answer: 12.5 Hz, i.e. 80 ms per frame.)
3. `conv_kernel_size: 9` — what's the receptive field of a single encoder block in *time* (in ms)? Of the full 12-layer stack? (Hint: `9 frames * 80 ms = 720 ms` per block, additive across layers.)
4. Why `precision: bf16-mixed` and not `fp16-mixed`? (Hint: bfloat16 has the same dynamic range as fp32, so it never underflows the way fp16 sometimes does in attention.)
5. Why `accumulate_grad_batches: 4`? (Hint: effective batch size = micro batch × accumulation, even though only `micro batch` fits on the GPU at one time.)

## Step 4 — kick off training (overnight)

```bash
python train.py \
    --config-path=conf \
    --config-name=fastconformer_ctc_small \
    model.train_ds.manifest_filepath=./data/train-clean-100/train_clean_100.json \
    model.validation_ds.manifest_filepath=./data/dev-clean/dev_clean.json \
    model.tokenizer.dir=./tokenizer \
    trainer.max_steps=30000
```

Open a second terminal and watch the loss curve in TensorBoard:

```bash
tensorboard --logdir lightning_logs/
```

What "good" looks like:

| Step | Train loss | Val WER (dev-clean) |
|------|-----------|---------------------|
| 1k   | ~80       | -                   |
| 5k   | ~40       | ~80%                |
| 10k  | ~20       | ~30%                |
| 20k  | ~12       | ~12%                |
| 30k  | ~9        | ~8%                 |

If your loss is stuck above 50 at step 10k, the most common culprits are:
- LR too low (try `2e-3`)
- LR schedule warmup too short (try `5000` steps)
- Effective batch size too small (try `accumulate_grad_batches: 8`)
- Tokenizer vocab too big for the small encoder (try 512 instead of 1024)
- Dataset path wrong (look at the first batch — does it contain real speech?)

## Step 5 — evaluate (5 minutes)

```bash
python evaluate.py \
    --checkpoint lightning_logs/version_0/checkpoints/last.ckpt \
    --manifest data/test-clean/test_clean.json
```

This loads your checkpoint, runs greedy CTC decoding, computes WER and a few example transcriptions. Expected output:

```
Test set: 2620 utterances, 5.4 hours
Greedy WER: 7.8%
First 5 examples:
  REF: he hoped there would be stew for dinner turnips and carrots and bruised potatoes
  HYP: he hoped there would be stew for dinner turnips and carrots and bruised potatoes
  ...
```

Compare to Parakeet's official numbers: a full FastConformer Large (~110M params) trained on much more data hits ~3.5% WER on test-clean. Your 30M-param model on 100 hours getting to ~8% is **excellent** by 2018 standards.

## Step 6 — read the source you just used (1 hour)

Now that you have an end-to-end working model, follow the NeMo source path you actually exercised:

| File | What it does | What to look at |
|------|--------------|------|
| `nemo/collections/asr/models/ctc_bpe_models.py` | `EncDecCTCModelBPE` — the model class your YAML instantiates | `training_step`, `transcribe` |
| `nemo/collections/asr/modules/conformer_encoder.py` | The encoder you trained | `ConformerEncoder.__init__` and `forward` |
| `nemo/collections/asr/parts/submodules/conformer_modules.py` | The Conformer block itself | `ConformerLayer.forward` |
| `nemo/collections/asr/parts/submodules/multi_head_attention.py` | Relpos MHSA | `RelPositionMultiHeadAttention.rel_shift` |
| `nemo/collections/asr/parts/submodules/subsampling.py` | The 3x stride-2 dw-sep stem | `ConvSubsampling.forward` |
| `nemo/collections/asr/losses/ctc.py` | The CTC loss wrapper | thin — wraps `torch.nn.CTCLoss` |

If you can step through `EncDecCTCModelBPE.training_step` in pdb and predict the shapes at every line, you have engineering depth.

## Reflection

1. You trained on 100 hours. Parakeet was trained on 120,000. How much better do you expect a fully-trained model to be on test-clean? (Look at the leaderboard.) How much better do you expect on out-of-domain audio? (The answer to the second is *much more* — small models overfit narrow distributions.)
2. The encoder has ~30M params. The CTC head is just a single linear layer of (256 × 1025) ≈ 263K params. Why is the rest of NeMo so complicated, given the model is mostly encoder?
3. If you swap CTC for RNN-T (exercise 5), what changes in the model count? What changes in training memory?

## Extension problems

See [`notes.md`](./notes.md):

- Add SpecAugment and quantify the WER improvement.
- Switch to InterCTC (auxiliary heads at layers 4 and 8) and measure convergence speedup.
- Try the `subsampling: striding` (non-depthwise-separable) stem and measure the speed/WER tradeoff.
- Add an external KenLM 4-gram trained on the LibriSpeech text and quantify shallow-fusion WER gain.
- Fine-tune a public `parakeet-ctc-1.1b` (or smaller) checkpoint on `train-clean-100` and compare to your from-scratch model.
