# Exercise 1 — extension problems

For when the base exercise wasn't hard enough, or you want to internalise the preprocessor more deeply.

### 1. Inverse STFT (medium)
Implement `istft(S, n_fft, hop_length, win_length)` using overlap-add. Verify that `istft(stft(x))` reconstructs `x` to within `1e-6` for a random signal. This is the moment when you appreciate that the Hann window's overlap-add property is what makes perfect reconstruction possible.

### 2. Vectorised STFT (medium)
Re-implement `stft` without a Python `for` loop over frames. Use `np.lib.stride_tricks.as_strided` or `np.lib.stride_tricks.sliding_window_view` to construct the framed signal in one call, then a single `np.fft.rfft` along the right axis. Benchmark: my reference vectorised version is ~50× faster than the loopy one for 30 s of 16 kHz audio.

### 3. SpecAugment (easy)
Add functions `spec_augment_freq_mask(spec, F, n_masks)` and `spec_augment_time_mask(spec, T, n_masks)` that zero out random rectangular regions in the spectrogram. Plot the before/after on the speech clip from `data/`. Compare to `nemo.collections.asr.modules.SpectrogramAugmentation`.

### 4. Match NeMo's exact preprocessor output (hard)
Load any Parakeet NeMo model, extract its `AudioToMelSpectrogramPreprocessor`, run it on the same input as your `log_mel_spectrogram`, and find the maximum absolute difference. They should agree to within `1e-3` if your conventions match. Reading the NeMo source to figure out *why* they don't agree (per-feature normalisation, dither, pre-emphasis, padding, log offset...) is itself a great exercise.

### 5. Pre-emphasis filter (easy)
Many classical ASR pipelines apply `y[n] = x[n] - 0.97 * x[n-1]` before the STFT — a one-tap high-pass filter that boosts high frequencies (where speech energy is naturally lower). Add this as an optional preprocessing step and observe what changes in the log-mel.

### 6. Profile against torchaudio (easy)
```python
import torch, torchaudio, time, numpy as np
from starter import log_mel_spectrogram

x = np.random.randn(16000 * 30).astype(np.float32)
xt = torch.from_numpy(x).unsqueeze(0)

melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80,
)

t0 = time.perf_counter(); _ = log_mel_spectrogram(x); print("ours:    ", time.perf_counter() - t0)
t0 = time.perf_counter(); _ = melspec(xt);             print("torchaudio:", time.perf_counter() - t0)
```
Expect torchaudio to be 50–200× faster on CPU. With `.cuda()` and a real GPU, the gap is more like 1000×.

### 7. Mel cepstrum (capstone)
If you want to go all the way back to classical ASR features: take the log-mel, apply a discrete cosine transform along the mel axis, keep the first ~13 coefficients. You've just computed MFCCs, the feature used by every GMM-HMM ASR system before 2012.
