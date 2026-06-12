# Parakeet Deep Dive

An interactive, self-contained engineering companion to NVIDIA's **Parakeet**
family of automatic speech recognition (ASR) models — from raw waveform to
emitted text, with all the maths and none of the hand-waving.

It is built for an engineer who knows their CS and neural-network basics but
not the DSP and sequence-transduction internals that the Parakeet papers and
NeMo docs assume. The goal is to take you from "I can call `model.transcribe()`"
to "I could design, debug, or critique this architecture."

👉 **Live site:** _enable GitHub Pages (see [Publishing](#publishing-to-github-pages)) and your URL appears here._

---

## Why this exists

The Parakeet papers (FastConformer, TDT) and the NeMo documentation are written
for people who already know what an STFT is, what a CTC lattice looks like, and
what "relative positional encoding" means without further elaboration. That
assumed background *is* most of the value — and it's exactly what's missing from
a normal read-through.

This pack was prepared as first-class training material: a single, dependency-
ordered narrative (problem → audio/DSP → neural building blocks → encoder →
decoders → tokens/training/inference → engineering) where every concept is
introduced before it's used, every chapter that cites a paper points at the
exact section, and the hardest ideas (CTC / RNN-T / TDT alignment lattices) come
with **interactive widgets** you can drive yourself rather than static figures.

The six hands-on labs then take you from pure-NumPy implementations of the core
algorithms all the way to training a real (small) FastConformer model and
converting it CTC → RNN-T → TDT, mirroring the actual Parakeet recipe.

## What's inside

| Path | What it is |
|---|---|
| **`index.html`** | The training site — 33 chapters across 8 parts, KaTeX maths, 17 interactive demos inline. Open in a browser. |
| **`styles.css`** | Visual system: light/dark themes, sticky table of contents, and a print stylesheet for offline PDF export. |
| **`widgets.js`** | Every interactive demo (vanilla JS + SVG + a little Plotly). One self-contained IIFE per widget. |
| **`diagrams/`** | Original title-card illustrations — one cover + one per Part. |
| **`exercises/`** | Six step-by-step lab sub-projects. See [Doing the labs](#doing-the-labs). |
| **`papers/`** | Citations + download links for the canonical papers (the PDFs themselves are not redistributed). See [`papers/README.md`](./papers/README.md). |

### The interactive widgets

The 17 widgets are the fastest path to internalising the maths — they are not
decoration. In page order:

`align` · `sampling` · `fourier` · `stft` · `mel` · `specaug` · `conv` ·
`param-calc` · `attention` · `pe` · `conformer` · `longform` · `ctc` · `rnnt` ·
`tdt` · `bpe` · `full-pipeline`

## Running the site locally

The page loads KaTeX and Plotly from CDNs and references `widgets.js` /
`styles.css` over relative paths. Browsers block some of that under `file://`,
so serve it over HTTP:

```bash
git clone <your-fork-url> parakeet-study
cd parakeet-study
python3 -m http.server 8000
# open http://localhost:8000/
```

A sticky table of contents on the left lets you jump between parts; the maths
renders via KaTeX on load.

### Suggested reading order

1. **Skim the TOC** to see the shape of the journey.
2. **Read top-to-bottom** — every chapter is in dependency order. Don't skip the
   audio/DSP part even if you "know" Fourier; the framing matters for the
   encoder later.
3. **Play with every widget.** The CTC, RNN-T, and TDT lattice walkers are the
   payoff.
4. **Open each cited paper** from [`papers/README.md`](./papers/README.md) when a
   chapter references it.
5. **Finish on the *Reading and video list* chapter** — it lays out the path
   from "understands Parakeet" to "could ship a competing model".

## Doing the labs

Six labs under [`exercises/`](./exercises/), in two tiers. Full workflow in
[`exercises/README.md`](./exercises/README.md).

### Tier 1 — from scratch (no GPU, pure NumPy / PyTorch on CPU)

Build the core algorithms yourself. Each ships a `README.md`, a `starter.py`
with explicit `TODO` blocks, a `reference.py` worked solution, and a `pytest`
suite that turns green when your implementation is correct.

| Lab | You build |
|---|---|
| `01-stft-from-scratch` | DFT, Hann windowing, the STFT, and the mel scale — in NumPy |
| `02-ctc-from-scratch` | The CTC forward-backward algorithm and loss |
| `03-rnnt-from-scratch` | The RNN-T lattice and transducer loss |

```bash
cd exercises/01-stft-from-scratch
uv venv .venv && source .venv/bin/activate     # or: python3 -m venv .venv
uv pip install -r requirements.txt              # or: pip install -r requirements.txt
pytest -v                                        # red until you fill the TODOs
# edit starter.py, re-run pytest until green; peek at reference.py only if stuck
```

> Uses [`uv`](https://github.com/astral-sh/uv) if you have it (fast); plain
> `python3 -m venv` + `pip` works identically.

### Tier 2 — train a real model (GPU recommended)

Use NeMo (the same code path Parakeet uses) on real LibriSpeech audio, then
convert the decoder through the Parakeet lineage.

| Lab | You do |
|---|---|
| `04-train-fastconformer-ctc` | Train a ~30 M-param FastConformer-CTC to ≤10% WER on `dev-clean` |
| `05-convert-to-rnnt` | Re-use the encoder, attach a transducer decoder, retrain |
| `06-convert-to-tdt` | Convert RNN-T → TDT, get duration prediction + word timestamps |

```bash
cd exercises/04-train-fastconformer-ctc
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt              # pulls nemo_toolkit[asr] — large
python download_librispeech.py                   # train-clean-100 + dev/test-clean
python train_tokenizer.py
python train.py --config-path=conf --config-name=fastconformer_ctc_small
python evaluate.py
```

**No local GPU?** [`exercises/CLOUD_GPU_SETUP.md`](./exercises/CLOUD_GPU_SETUP.md)
has step-by-step Colab (free T4), Kaggle (free P100), and RunPod (~$5) recipes.
Lab 04 fits comfortably in Colab's free tier.

## Offline PDF

The website is the primary product (the widgets and labs don't translate to
paper). For a readable offline copy of the *prose*, a print stylesheet drops the
TOC, collapses to one column, and substitutes interactive widgets with a note.
Generate it headlessly:

```bash
python3 -m http.server 8000 &
npx -y playwright-chromium screenshot --help >/dev/null 2>&1  # ensures browser
node -e '
const { chromium } = require("playwright-chromium");
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto("http://localhost:8000/", { waitUntil: "networkidle" });
  await p.pdf({ path: "parakeet-deep-dive.pdf", format: "A4", printBackground: true });
  await b.close();
})();'
```

## Publishing to GitHub Pages

The repo is a static site at its root, ready to publish:

1. Push to GitHub (`main` branch).
2. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
3. The included workflow ([`.github/workflows/pages.yml`](./.github/workflows/pages.yml))
   deploys on every push to `main`. A `.nojekyll` file disables Jekyll so all
   assets are served verbatim.

Your site goes live at `https://<user>.github.io/<repo>/`.

## Acknowledgements

- **NVIDIA NeMo team** — for Parakeet, FastConformer, and the open NeMo
  toolkit, released under permissive licences. In particular the work of
  Somshubra Majumdar, Nithin Rao Koluguri, Vladimir Bataev, Piotr Żelasko,
  Daniel Galvez, and colleagues documented in the NVIDIA technical blogs.
- **The paper authors** cited in [`papers/README.md`](./papers/README.md) —
  Graves; Vaswani et al.; Gulati et al.; Dai et al.; Park et al.; Xu et al.;
  Huang et al.; and the rest — whose work this guide exists to make legible.
- **Suno.ai**, NVIDIA's collaborator on the original Parakeet models.

This is an independent educational companion. It is not affiliated with or
endorsed by NVIDIA. All architecture diagrams in `diagrams/` are original
illustrations, not reproductions of NVIDIA figures.

## Licence

Dual-licensed:

- **Code** (`widgets.js`, `exercises/`, configs, scripts) — MIT, see [`LICENSE`](./LICENSE).
- **Prose & diagrams** (`index.html` text, `diagrams/`, lab instructions) —
  CC-BY-4.0, see [`LICENSE-CONTENT`](./LICENSE-CONTENT).

Third-party papers are cited, not redistributed, and remain under their own
copyrights.

---

*Prepared by Chris Kendall as an engineering training resource.*
