"""
Exercise 1 — STFT and mel filterbank from scratch
=================================================
Fill in each TODO block in order. Run the matching pytest after each one:

    pytest test_stft.py::test_dft_matches_numpy -v
    pytest test_stft.py::test_hann_window         -v
    pytest test_stft.py::test_stft_matches_scipy  -v
    pytest test_stft.py::test_mel_filterbank      -v
    pytest test_stft.py::test_full_pipeline       -v

Read README.md *before* starting. Don't peek at reference.py until you're stuck.
"""
from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
# TODO 1: discrete Fourier transform from the definition
# -----------------------------------------------------------------------------
# Implement the DFT as a matrix-vector product, *not* by calling np.fft.fft.
# Shape: x is a 1D real array of length N; return a 1D complex array of length N.
#
# Hint: the DFT matrix W has entries W[k, n] = exp(-2j * pi * k * n / N).
#       Build it with broadcasting from np.arange(N).reshape(-1, 1) and
#       np.arange(N).reshape(1, -1).
def dft(x: np.ndarray) -> np.ndarray:
    N = len(x)
    raise NotImplementedError("TODO 1: write the DFT as a matrix-vector product")


# -----------------------------------------------------------------------------
# TODO 2: Hann window
# -----------------------------------------------------------------------------
# Return a 1D array of length n, where w[i] = 0.5 * (1 - cos(2*pi*i / (n-1))).
# Must satisfy w[0] == w[n-1] == 0 and the array is symmetric.
def hann_window(n: int) -> np.ndarray:
    raise NotImplementedError("TODO 2: implement Hann window")


# -----------------------------------------------------------------------------
# TODO 3: short-time Fourier transform
# -----------------------------------------------------------------------------
# Return a 2D complex array of shape [n_fft // 2 + 1, n_frames].
#
# Convention used by this exercise (matches NeMo/torchaudio's center=False mode):
#   * No reflection padding around the signal.
#   * Frame i starts at sample index i * hop_length.
#   * If a frame runs off the end of x, zero-pad it.
#   * n_frames = 1 + (len(x) - win_length) // hop_length, when len(x) >= win_length;
#     otherwise n_frames = 1 (just the zero-padded first frame).
#   * Window the frame, zero-pad to n_fft if win_length < n_fft, then np.fft.rfft.
def stft(
    x: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    raise NotImplementedError("TODO 3: implement the STFT")


# -----------------------------------------------------------------------------
# TODO 4: Hz <-> mel conversions and the mel filterbank
# -----------------------------------------------------------------------------
# Use the Slaney convention (what librosa defaults to with htk=False).
# mel = 2595 * log10(1 + hz / 700)
# hz  = 700 * (10 ** (mel / 2595) - 1)
def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    raise NotImplementedError("TODO 4a: Hz -> mel")


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    raise NotImplementedError("TODO 4b: mel -> Hz")


# Build a [n_mels, n_fft // 2 + 1] matrix of triangular filters spaced
# uniformly on the mel scale between f_min and f_max.
#
# Algorithm:
#   1. mel_points = n_mels + 2 evenly spaced mel values from f_min to f_max.
#   2. hz_points = mel_to_hz(mel_points).
#   3. bin_points = floor((n_fft + 1) * hz_points / sample_rate).
#   4. For m in 1..n_mels:
#        rising  edge: bins [bin_points[m-1], bin_points[m])   slope up
#        falling edge: bins [bin_points[m],   bin_points[m+1]) slope down
def mel_filterbank(
    n_mels: int = 80,
    n_fft: int = 512,
    sample_rate: int = 16000,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    raise NotImplementedError("TODO 4c: build the mel filterbank")


# -----------------------------------------------------------------------------
# TODO 5: full waveform -> log-mel-spectrogram pipeline
# -----------------------------------------------------------------------------
# Output shape: [n_mels, n_frames]. Use the helpers above.
def log_mel_spectrogram(
    x: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    log_offset: float = 1e-6,
) -> np.ndarray:
    raise NotImplementedError("TODO 5: glue STFT + |.|^2 + mel + log together")


# -----------------------------------------------------------------------------
# Tiny demo entry point — runnable so you can sanity-check visually
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sr = 16000
    t = np.arange(sr) / sr                 # 1 second
    x = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

    log_mel = log_mel_spectrogram(x, sample_rate=sr)
    print(f"Output shape: {log_mel.shape}")
    peak_bin = int(np.argmax(log_mel.sum(axis=1)))
    print(f"Peak mel-bin index: {peak_bin}  (should be near the 440 Hz bin)")
