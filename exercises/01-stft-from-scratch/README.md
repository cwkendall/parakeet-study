# Exercise 1 — STFT and mel filterbank from scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwkendall/parakeet-study/blob/main/exercises/01-stft-from-scratch/explore.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cwkendall/parakeet-study/main?labpath=exercises%2F01-stft-from-scratch%2Fexplore.ipynb)

> ▶️ **Prefer a notebook?** Open the runnable `explore.ipynb` companion (the full reference solution, broken into steps with plots) in Colab or Binder above, or locally with `jupyter lab explore.ipynb`.


**Time budget:** 1–2 hours.
**Prerequisites:** Chapters 4–8 of the deep-dive (`../../index.html#ch-digital` through `#ch-preproc`).
**Builds:** the exact preprocessing block that sits in front of every Parakeet model.

By the end of this exercise you will have written, in pure NumPy:

1. A discrete Fourier transform (slow but obviously correct).
2. A short-time Fourier transform with a Hann window.
3. A mel filterbank.
4. The full `waveform → log-mel-spectrogram` pipeline.

…and verified each piece against SciPy / librosa to within numerical tolerance. This is the same code path NeMo's `AudioToMelSpectrogramPreprocessor` runs in C++/CUDA — yours will be 100× slower, but it will give you the same numbers, which is the point.

## Why this matters

Every neural ASR system stands on top of these three or four operations. If you understand them numerically — what an FFT bin index *means* in Hz, what changing the window length costs you in frequency resolution, why mel binning is a constant matrix multiply rather than a "model" — you can read any preprocessor config in any toolkit and know exactly what it does.

## The tutorial

Open `starter.py`. It is a skeleton with five **TODO blocks**, each marked clearly. Work them in order; each one has a paired pytest in `test_stft.py` you can run as you go.

### Step 1 — the DFT (15 minutes)

The discrete Fourier transform is, literally, a matrix multiplication:

```
X[k] = sum over n=0..N-1 of x[n] * exp(-2πi · k · n / N)
```

Write this as one line of NumPy. Don't use `np.fft.fft` — the whole point is to confirm you can compute the same numbers by hand.

```python
def dft(x):
    N = len(x)
    k = np.arange(N).reshape(-1, 1)   # column vector
    n = np.arange(N).reshape(1, -1)   # row vector
    W = np.exp(-2j * np.pi * k * n / N)   # the DFT matrix
    return W @ x
```

When the test passes:

```bash
pytest test_stft.py::test_dft_matches_numpy -v
```

…you have proven your DFT agrees with NumPy's FFT on random inputs to within `1e-10`. You also have first-hand evidence that the DFT really is just a fixed change-of-basis matrix.

**Discussion question:** What's the time complexity of your DFT? What's NumPy's? Profile it with `time.perf_counter()` for `N=1024` and `N=4096`. You should see roughly a 16× difference between your $O(N^2)$ DFT and `np.fft.fft`'s $O(N \log N)$.

### Step 2 — the Hann window (10 minutes)

The STFT chops the signal into overlapping windows and FFTs each one. The "windowing" part means we multiply each frame by a smooth taper before FFTing, so that the implicit periodic extension at the edges doesn't introduce spurious high-frequency artifacts (spectral leakage).

The Hann window is:

```
w[n] = 0.5 * (1 - cos(2π · n / (N-1)))    for n in 0..N-1
```

Implement it in `hann_window(n)`. The test checks that it's symmetric and that `w[0] == w[N-1] == 0`.

### Step 3 — the STFT (20 minutes)

Write `stft(x, n_fft, hop_length, win_length)`. This is the workhorse:

1. Compute the window: `w = hann_window(win_length)`.
2. Zero-pad to `n_fft` if `win_length < n_fft` (NeMo's case: window is 400, FFT is 512).
3. For each frame start position `i = 0, hop_length, 2*hop_length, ...`:
   - Extract `frame = x[i : i + win_length]` (zero-pad the last frame if it overruns).
   - Multiply by the window: `windowed = frame * w`.
   - FFT it: `spectrum = np.fft.rfft(windowed, n=n_fft)`.
   - Stack.
4. Return a 2D array of shape `[n_fft//2 + 1, n_frames]`.

Use `np.fft.rfft` here (real FFT — your signal is real, so the FFT output is conjugate-symmetric and you only need the first `n_fft//2 + 1` bins). The test compares your output against `scipy.signal.stft` on a chirp signal.

**Trap to avoid:** SciPy and NeMo have different defaults for `center=True/False`. Read the test for the exact convention this exercise uses.

### Step 4 — the mel filterbank (20 minutes)

A mel filterbank is a sparse matrix `M` of shape `[n_mels, n_fft//2 + 1]`. Each row is a triangular weighting that picks out energy in one perceptual band. Computing `M @ power_spectrum` collapses the linear-frequency spectrum into a perceptual-frequency spectrum.

The construction:

```python
def mel_filterbank(n_mels, n_fft, sample_rate, f_min=0, f_max=None):
    if f_max is None:
        f_max = sample_rate / 2

    # 1. Choose n_mels + 2 evenly-spaced points on the mel scale
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # 2. Convert back to Hz, then to FFT bin index
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # 3. Build n_mels triangular filters
    # Filter m has its three vertices at bin_points[m], bin_points[m+1], bin_points[m+2]
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_left, f_centre, f_right = bin_points[m-1], bin_points[m], bin_points[m+1]
        for k in range(f_left, f_centre):
            fb[m-1, k] = (k - f_left) / (f_centre - f_left)
        for k in range(f_centre, f_right):
            fb[m-1, k] = (f_right - k) / (f_right - f_centre)
    return fb
```

You also need `hz_to_mel` and `mel_to_hz`. Use the Slaney convention (the one librosa defaults to):

```
mel = 2595 * log10(1 + hz / 700)
hz  = 700 * (10^(mel/2595) - 1)
```

The test compares your filterbank against `librosa.filters.mel(..., htk=False)`.

### Step 5 — the full preprocessor (15 minutes)

Now glue it together:

```python
def log_mel_spectrogram(x, sample_rate=16000, n_fft=512, win_length=400,
                       hop_length=160, n_mels=80, log_offset=1e-6):
    spec = stft(x, n_fft, hop_length, win_length)       # [n_fft//2+1, T]
    power = np.abs(spec) ** 2                             # power spectrogram
    fb = mel_filterbank(n_mels, n_fft, sample_rate)       # [n_mels, n_fft//2+1]
    mel = fb @ power                                       # [n_mels, T]
    log_mel = np.log(mel + log_offset)                     # [n_mels, T]
    return log_mel
```

The test runs your pipeline on a 1-second sine wave at 440 Hz and checks:

1. Output shape is `[80, T]` for some reasonable `T`.
2. The peak energy lives in the mel bins corresponding to ~440 Hz.
3. Other bins are far below the peak.

Then it runs on a short LibriSpeech-style speech clip (synthetic — provided in `data/`) and compares against librosa within a relative tolerance.

## Running everything

```bash
pytest test_stft.py -v
```

All five tests should pass. If they don't, read the failure message — the tests are designed to point at specific TODOs.

## Reflection

Before moving to exercise 2, answer for yourself:

1. If you doubled `win_length` from 400 to 800 (keeping `hop_length=160`), what changes in the output? Both in shape and in what the spectrogram "looks like".
2. What's the temporal resolution of the output? The frequency resolution? Why is there a tradeoff between them, and why is 25 ms / 10 ms the standard ASR sweet spot?
3. Why log? What does the per-feature normalisation step (which NeMo adds after this) do for the encoder?

If you can't answer these, re-read chapters 6–8 of the deep-dive before moving on.

## Extension problems (optional)

See [`notes.md`](./notes.md) for harder follow-ups:

- Implement the inverse STFT (overlap-add reconstruction).
- Reproduce the SpecAugment time/frequency masking.
- Write a vectorised STFT that doesn't use a Python `for` loop over frames.
- Time your full pipeline against `torchaudio.transforms.MelSpectrogram` on the same input — what's the slowdown factor?
