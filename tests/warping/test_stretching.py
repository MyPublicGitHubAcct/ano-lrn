import numpy as np

from python.warping import pitch_shift, time_stretch

FS = 44100
FRAME = 512
HOP = 128


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def _dominant_freq(signal, fs, n=None):
    n = n or len(signal)
    spectrum = np.abs(np.fft.rfft(signal, n=n))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return float(freqs[np.argmax(spectrum)])


def test_time_stretch_returns_1d():
    sig = _sine(n=4096)
    out = time_stretch(sig, rate=2.0, frame_size=FRAME, hop_size=HOP)
    assert out.ndim == 1


def test_pitch_shift_preserves_length():
    sig = _sine(n=4096)
    out = pitch_shift(sig, semitones=3.0, fs=FS, frame_size=FRAME, hop_size=HOP)
    assert len(out) == len(sig)


def test_time_stretch_rate_1_identity():
    sig = _sine(n=4096)
    out = time_stretch(sig, rate=1.0)
    np.testing.assert_array_equal(out, sig)


def test_time_stretch_slower_produces_longer_output():
    sig = _sine(n=4096)
    out = time_stretch(sig, rate=2.0, frame_size=FRAME, hop_size=HOP)
    assert len(out) > len(sig)


def test_time_stretch_faster_produces_shorter_output():
    sig = _sine(n=4096)
    out = time_stretch(sig, rate=0.5, frame_size=FRAME, hop_size=HOP)
    assert len(out) < len(sig)


def test_pitch_shift_zero_semitones_identity():
    sig = _sine(n=4096)
    out = pitch_shift(sig, semitones=0.0, fs=FS)
    np.testing.assert_array_equal(out, sig)


def test_pitch_shift_up_raises_dominant_frequency():
    freq = 500.0
    sig = np.sin(2 * np.pi * freq * np.arange(FS) / FS)
    out = pitch_shift(sig, semitones=12.0, fs=FS, frame_size=2048, hop_size=512)
    dom_in = _dominant_freq(sig, FS)
    dom_out = _dominant_freq(out, FS)
    assert dom_out > dom_in * 1.5
