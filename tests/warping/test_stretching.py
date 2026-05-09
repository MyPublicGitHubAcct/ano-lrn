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
    """time_stretch must return a 1-D array (single audio channel)."""
    sig = _sine(n=4096)
    out = time_stretch(sig, rate=2.0, frame_size=FRAME, hop_size=HOP)
    assert out.ndim == 1


def test_pitch_shift_preserves_length():
    """pitch_shift must return an array of the same length as the input."""
    sig = _sine(n=4096)
    out = pitch_shift(sig, semitones=3.0, fs=FS, frame_size=FRAME, hop_size=HOP)
    assert len(out) == len(sig)


def test_time_stretch_rate_1_identity():
    """rate=1.0 is a no-op; the output must equal the input exactly."""
    sig = _sine(n=4096)
    out = time_stretch(sig, rate=1.0)
    np.testing.assert_array_equal(out, sig)


def test_time_stretch_slower_produces_longer_output():
    """rate > 1 slows the signal down (more output samples per input sample); output must be longer than input."""
    sig = _sine(n=4096)
    out = time_stretch(sig, rate=2.0, frame_size=FRAME, hop_size=HOP)
    assert len(out) > len(sig)


def test_time_stretch_faster_produces_shorter_output():
    """rate < 1 speeds the signal up (fewer output samples per input sample); output must be shorter than input."""
    sig = _sine(n=4096)
    out = time_stretch(sig, rate=0.5, frame_size=FRAME, hop_size=HOP)
    assert len(out) < len(sig)


def test_pitch_shift_zero_semitones_identity():
    """0 semitones is a no-op; the output must equal the input exactly."""
    sig = _sine(n=4096)
    out = pitch_shift(sig, semitones=0.0, fs=FS)
    np.testing.assert_array_equal(out, sig)


def test_pitch_shift_up_raises_dominant_frequency():
    """Shifting up by 12 semitones (one octave) must at least raise the dominant frequency by more than 1.5×."""
    freq = 500.0
    sig = np.sin(2 * np.pi * freq * np.arange(FS) / FS)
    out = pitch_shift(sig, semitones=12.0, fs=FS, frame_size=2048, hop_size=512)
    dom_in = _dominant_freq(sig, FS)
    dom_out = _dominant_freq(out, FS)
    assert dom_out > dom_in * 1.5


def test_pitch_shift_12_semitones_doubles_frequency():
    """12 semitones = 1 octave; dominant output frequency must be within 10% of 2× the input frequency."""
    freq = 400.0
    sig = np.sin(2 * np.pi * freq * np.arange(FS) / FS)
    out = pitch_shift(sig, semitones=12.0, fs=FS, frame_size=2048, hop_size=512)
    dom_out = _dominant_freq(out, FS)
    assert abs(dom_out - 2 * freq) / (2 * freq) < 0.10


def test_time_stretch_preserves_dominant_frequency():
    """time_stretch must not change pitch; stretched output dominant frequency must match the original."""
    freq = 440.0
    sig = _sine(freq=freq, n=4096)
    out = time_stretch(sig, rate=1.5, frame_size=FRAME, hop_size=HOP)
    dom_in = _dominant_freq(sig, FS, n=len(sig) * 2)
    dom_out = _dominant_freq(out, FS, n=len(out) * 2)
    bin_width = FS / max(len(sig), len(out))
    assert abs(dom_out - dom_in) < bin_width * 3

