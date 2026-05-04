import numpy as np

from python.time_frequency import istft, stft

FS = 44100
FRAME = 512
HOP = 128


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_stft_shape():
    sig = _sine(n=4096)
    S = stft(sig, FRAME, HOP)
    expected_frames = 1 + (4096 - FRAME) // HOP
    assert S.shape == (FRAME // 2 + 1, expected_frames)


def test_stft_dtype_complex():
    S = stft(_sine(), FRAME, HOP)
    assert np.iscomplexobj(S)


def test_istft_shape():
    S = stft(_sine(n=4096), FRAME, HOP)
    out = istft(S, HOP)
    assert out.ndim == 1


def test_stft_istft_roundtrip():
    frame_size = 512
    hop_size = 128
    sig = _sine(n=4096)
    S = stft(sig, frame_size, hop_size)
    out = istft(S, hop_size)
    start = frame_size
    end = len(sig) - frame_size
    np.testing.assert_allclose(out[start:end], sig[start:end], atol=0.02)


def test_stft_sine_peak_at_correct_bin():
    freq = 1000.0
    frame_size = 4096
    sig = np.sin(2 * np.pi * freq * np.arange(frame_size) / FS)
    S = stft(sig, frame_size, frame_size)
    mag = np.abs(S[:, 0])
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / FS)
    peak_freq = freqs[np.argmax(mag)]
    assert abs(peak_freq - freq) < 20.0
