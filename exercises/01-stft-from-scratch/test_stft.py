"""
Pytest checks for exercise 1.

Run a single test:
    pytest test_stft.py::test_dft_matches_numpy -v

Run them all:
    pytest test_stft.py -v
"""
import numpy as np
import pytest

# Importing starter (the student's solution). If you want to test the reference
# instead, change this line.
from starter import (
    dft,
    hann_window,
    stft,
    hz_to_mel,
    mel_to_hz,
    mel_filterbank,
    log_mel_spectrogram,
)

rng = np.random.default_rng(0)


def test_dft_matches_numpy():
    for N in [16, 64, 257]:
        x = rng.standard_normal(N)
        ours = dft(x)
        theirs = np.fft.fft(x)
        np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_hann_window():
    w = hann_window(400)
    assert w.shape == (400,)
    assert w[0] == pytest.approx(0.0, abs=1e-12)
    assert w[-1] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(w, w[::-1], atol=1e-12)
    # Energy is correct: sum of squared Hann window is approximately 3*(N-1)/8 ≈ N*3/8.
    # For N=400 that's ~149.6. Check in the right ballpark.
    assert 140 < (w ** 2).sum() < 160


def test_stft_shape():
    # 1-second mono at 16 kHz, 25 ms window, 10 ms hop.
    x = rng.standard_normal(16000)
    S = stft(x, n_fft=512, hop_length=160, win_length=400)
    expected_frames = 1 + (16000 - 400) // 160
    assert S.shape == (257, expected_frames)


def test_stft_matches_scipy():
    from scipy.signal import stft as scipy_stft
    # Use the same conventions: window=hann, nperseg=400, noverlap=240, nfft=512,
    # boundary=None (no padding), padded=False, return_onesided=True.
    x = rng.standard_normal(8000)
    S_ours = stft(x, n_fft=512, hop_length=160, win_length=400)
    _, _, S_scipy = scipy_stft(
        x,
        fs=16000,
        window="hann",
        nperseg=400,
        noverlap=400 - 160,
        nfft=512,
        boundary=None,
        padded=False,
        return_onesided=True,
        scaling="spectrum",
    )
    # Scipy normalises by sum(window) by default; compare magnitude ratios bin-wise.
    # Both should have the same shape.
    assert S_ours.shape == S_scipy.shape

    # Compare magnitude (phase can wrap), with a generous tolerance: scipy applies
    # its own window normalisation, so we only check that the *pattern* matches by
    # checking the correlation per frame is near 1.
    for t in range(S_ours.shape[1]):
        a = np.abs(S_ours[:, t])
        b = np.abs(S_scipy[:, t])
        if a.sum() < 1e-9 or b.sum() < 1e-9:
            continue
        corr = np.corrcoef(a, b)[0, 1]
        assert corr > 0.999, f"frame {t} correlation only {corr:.4f}"


def test_mel_filterbank():
    fb = mel_filterbank(n_mels=80, n_fft=512, sample_rate=16000)
    assert fb.shape == (80, 257)

    # All entries non-negative.
    assert fb.min() >= 0

    # With this small FFT and many mel bins, the first few low-frequency filters
    # collapse to width-zero (consecutive bin_points round to the same int).
    # That is expected: most of the filterbank should still be active.
    n_active = sum(1 for m in range(80) if fb[m].max() > 0)
    assert n_active >= 60, f"only {n_active}/80 mel filters are non-empty"

    # Among the active filters, peaks should be monotonically non-decreasing.
    active_peaks = [int(np.argmax(fb[m])) for m in range(80) if fb[m].max() > 0]
    assert active_peaks == sorted(active_peaks), "active filter peaks should be monotonic"

    # Cross-check against librosa (Slaney convention).
    try:
        import librosa
    except ImportError:
        pytest.skip("librosa not installed")
    fb_ref = librosa.filters.mel(
        sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000, htk=False, norm=None,
    )
    # Compare column-wise totals (sum of all filter weights at each FFT bin).
    # Per-row comparison is too sensitive to ±1 bin rounding differences
    # between this implementation's `floor(...)` and librosa's exact bin
    # frequencies; the column-sum integrates over those small offsets.
    col_ours = fb.sum(axis=0)
    col_ref = fb_ref.sum(axis=0)
    # Both should agree on which FFT bins are covered by the bank at all.
    overlap = (col_ours > 0) & (col_ref > 0)
    assert overlap.sum() > 200, "filterbanks cover wildly different FFT bins"
    # And the relative magnitude pattern should match.
    corr = np.corrcoef(col_ours[overlap], col_ref[overlap])[0, 1]
    assert corr > 0.9, f"filterbank column-sum correlation only {corr:.3f}"


def test_hz_mel_roundtrip():
    hz = np.array([100.0, 440.0, 1000.0, 4000.0, 8000.0])
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(hz)), hz, atol=1e-6)


def test_full_pipeline():
    # Pure tone at 440 Hz - its energy should pile up in the mel bin spanning 440 Hz.
    sr = 16000
    t = np.arange(sr) / sr
    x = 0.5 * np.sin(2 * np.pi * 440 * t)
    log_mel = log_mel_spectrogram(x, sample_rate=sr)
    assert log_mel.shape == (80, 1 + (16000 - 400) // 160)
    peak = int(np.argmax(log_mel.sum(axis=1)))
    # 440 Hz on the Slaney mel scale, in 80-bin filterbank with f_max=8000,
    # should fall around mel index 15-25. Generous bounds.
    assert 5 < peak < 35, f"peak at unexpected mel bin {peak}"

    # Mostly silent: silence should produce very negative log values everywhere.
    silence = np.zeros(16000)
    log_mel_silence = log_mel_spectrogram(silence, sample_rate=sr)
    assert log_mel_silence.max() < -10
