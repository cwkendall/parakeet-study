# Parakeet study pack — exercises

Six guided exercises that build, by hand, every component you read about in [`../index.html`](../index.html). Each is a self-contained sub-project with starter code, step-by-step instructions, and a reference solution. They are deliberately staged so that exercises 1–3 give you the algorithmic understanding (DSP, CTC, RNN-T) and exercises 4–6 take you into real model training and the engineering tradeoffs that make Parakeet what it is.

| # | Folder | Time budget | What you build | Depends on chapters |
|---|--------|-------------|----------------|---------------------|
| 1 | [`01-stft-from-scratch/`](./01-stft-from-scratch/) | ~1–2 h | STFT and mel filterbank in pure NumPy, compared against SciPy / librosa | 4–8 |
| 2 | [`02-ctc-from-scratch/`](./02-ctc-from-scratch/) | ~1 day | CTC forward, backward, loss, gradient, greedy + prefix-beam decoder, plus a tiny CNN classifier trained with the loss | 10–11, 21 |
| 3 | [`03-rnnt-from-scratch/`](./03-rnnt-from-scratch/) | ~1 day | RNN-T forward-backward in NumPy, autograd cross-check against PyTorch's `RNNTLoss` | 22 |
| 4 | [`04-train-fastconformer-ctc/`](./04-train-fastconformer-ctc/) | ~1 weekend | A small FastConformer-CTC trained end-to-end on LibriSpeech-clean with NeMo, aiming for ≤10% WER | 17–18, 21, 25 |
| 5 | [`05-convert-to-rnnt/`](./05-convert-to-rnnt/) | ~1 day | Swap the CTC head for an RNN-T decoder, fine-tune, compare WER | 22 |
| 6 | [`06-convert-to-tdt/`](./06-convert-to-tdt/) | ~1 weekend | Swap the RNN-T head for a TDT decoder, measure greedy decode RTFx, profile the joint network | 23, 26 |

> **Exercises 4–6 need an NVIDIA GPU.** If you don't have one locally, see [`CLOUD_GPU_SETUP.md`](./CLOUD_GPU_SETUP.md) for step-by-step setup of Colab (free T4 — Ex 4 fits), Kaggle (free P100), or RunPod (~$10 total for all three, with persistent storage). Exercises 1–3 are pure NumPy and run anywhere, including Apple Silicon.

## How the exercises are structured

Each folder follows the same convention so you don't waste time on logistics:

```
NN-name/
  README.md           ← step-by-step tutorial sheet (read this first)
  requirements.txt    ← Python deps (pip install -r requirements.txt)
  starter.py          ← skeleton with TODO blocks for you to fill in
  reference.py        ← worked solution (peek only when stuck)
  test_<name>.py      ← pytest sanity checks for each TODO
  data/               ← any tiny fixtures the exercise needs
  notes.md            ← optional: deeper-dive extension problems
```

## Workflow recipe (works for every exercise)

```bash
cd exercises/01-stft-from-scratch

# 1. set up an isolated environment for this exercise
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. read the README end-to-end before touching code
$EDITOR README.md

# 3. work through starter.py, one TODO at a time
$EDITOR starter.py

# 4. run the test for the section you just completed
pytest test_stft.py::test_dft_real -v

# 5. when all tests pass, diff your solution against the reference
diff starter.py reference.py | less
```

`uv` is the recommended Python runner (already installed on this machine). If you prefer plain `python -m venv` that works too.

## Prerequisites

- Python 3.10+
- The packages listed in each exercise's `requirements.txt` (mostly `numpy`, `scipy`, `matplotlib`, `pytest` for ex 1–3; `nemo_toolkit[asr]`, `torch`, `lightning` for ex 4–6).
- For exercises 4–6: a CUDA GPU. An RTX 3090 / 4090 / A4000 (24 GB) is enough for the small configs in the tutorials; an A100 is overkill but pleasant.

## A note on difficulty curve

Exercises 1–3 are designed so you *cannot* finish them by reading the reference solution and copying — the tests check intermediate quantities (e.g. forward variables, alpha values, gradient shape and magnitude), not just final outputs. Exercises 4–6 are mostly recipe-following, but the *extension problems* in `notes.md` are where the real learning happens; treat the base exercise as a tutorial and the extensions as a homework assignment.

If at any point an exercise feels too easy, jump to the "Extension problems" section of that exercise's `notes.md`. If it feels too hard, the relevant chapter in `../index.html` has the prerequisites; you may be missing one.
