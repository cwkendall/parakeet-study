"""
Exercise 1 — reference solution.

Only consult this when you're stuck on a TODO in starter.py.
"""
from __future__ import annotations

import numpy as np


def dft(x: np.ndarray) -> np.ndarray:
    N = len(x)
    k = np.arange(N).reshape(-1, 1)
    n = np.arange(N).reshape(1, -1)
    W = np.exp(-2j * np.pi * k * n / N)
    return W @ x


def hann_window(n: int) -> np.ndarray:
    if n == 1:
        return np.ones(1)
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))


def stft(
    x: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    assert win_length <= n_fft, "win_length must be <= n_fft (zero-pad otherwise)"
    w = hann_window(win_length)

    if len(x) < win_length:
        n_frames = 1
    else:
        n_frames = 1 + (len(x) - win_length) // hop_length

    n_bins = n_fft // 2 + 1
    out = np.zeros((n_bins, n_frames), dtype=np.complex128)

    pad_left = (n_fft - win_length) // 2
    pad_right = n_fft - win_length - pad_left

    for i in range(n_frames):
        start = i * hop_length
        frame = x[start:start + win_length]
        if len(frame) < win_length:
            frame = np.pad(frame, (0, win_length - len(frame)))
        windowed = frame * w
        padded = np.pad(windowed, (pad_left, pad_right))
        out[:, i] = np.fft.rfft(padded, n=n_fft)

    return out


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int = 80,
    n_fft: int = 512,
    sample_rate: int = 16000,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    if f_max is None:
        f_max = sample_rate / 2

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_centre = bin_points[m]
        f_right = bin_points[m + 1]
        if f_centre > f_left:
            for k in range(f_left, f_centre):
                fb[m - 1, k] = (k - f_left) / (f_centre - f_left)
        if f_right > f_centre:
            for k in range(f_centre, f_right):
                fb[m - 1, k] = (f_right - k) / (f_right - f_centre)
    return fb


def log_mel_spectrogram(
    x: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    log_offset: float = 1e-6,
) -> np.ndarray:
    spec = stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    power = np.abs(spec) ** 2
    fb = mel_filterbank(n_mels=n_mels, n_fft=n_fft, sample_rate=sample_rate)
    mel = fb @ power
    return np.log(mel + log_offset)


if __name__ == "__main__":
    sr = 16000
    t = np.arange(sr) / sr
    x = 0.5 * np.sin(2 * np.pi * 440 * t)
    log_mel = log_mel_spectrogram(x, sample_rate=sr)
    print(f"Output shape: {log_mel.shape}")
    peak_bin = int(np.argmax(log_mel.sum(axis=1)))
    print(f"Peak mel-bin index: {peak_bin}")
