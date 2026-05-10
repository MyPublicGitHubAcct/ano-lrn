import numpy as np
import pytest

from python.delay import tape_delay
from python.delay.line import fractional_delay_line

FS = 44100


def _impulse(duration=0.5):
    N = int(duration * FS)
    sig = np.zeros(N)
    sig[0] = 1.0
    return sig


def _sine(freq=440.0, duration=2.0):
    N = int(duration * FS)
    t = np.arange(N) / FS
    return np.sin(2 * np.pi * freq * t)


# --- output shape ---

def test_shape():
    """tape_delay must return (t, output) arrays each matching the input length."""
    sig = _impulse()
    t, out = tape_delay(sig, FS)
    assert t.shape == sig.shape
    assert out.shape == sig.shape


def test_time_axis():
    """Time axis must start at 0 and have spacing 1/fs."""
    sig = _impulse(duration=0.3)
    t, _ = tape_delay(sig, FS)
    assert t[0] == pytest.approx(0.0)
    assert t[1] - t[0] == pytest.approx(1.0 / FS)


# --- feedback = 0 single echo ---

def test_no_feedback_single_echo_position():
    """With feedback=0, wow=0, flutter=0 the only echo peak lands at delay_time*fs samples."""
    delay_time = 0.1  # 4410 samples — integer at 44100 Hz
    sig = _impulse(duration=0.5)
    _, out = tape_delay(
        sig, FS,
        delay_time=delay_time,
        feedback=0.0,
        flutter_depth=0.0,
        wow_depth=0.0,
        saturation=0.0,
    )
    peak_idx = int(np.argmax(np.abs(out)))
    expected_idx = int(delay_time * FS)
    assert abs(peak_idx - expected_idx) <= 2


def test_no_feedback_single_echo_amplitude():
    """With feedback=0, wow=0, flutter=0 the echo amplitude should be close to 1."""
    delay_time = 0.1
    sig = _impulse(duration=0.5)
    _, out = tape_delay(
        sig, FS,
        delay_time=delay_time,
        feedback=0.0,
        flutter_depth=0.0,
        wow_depth=0.0,
        saturation=0.0,
    )
    assert np.max(np.abs(out)) > 0.9


def test_no_feedback_matches_fractional_delay():
    """With feedback=0 the output must equal fractional_delay_line of the input
    evaluated with the same delay curve (wow+flutter perturbations included)."""
    N = int(0.3 * FS)
    delay_time = 0.05
    flutter_rate = 10.0
    flutter_depth = 0.0003
    wow_rate = 0.5
    wow_depth = 0.001

    rng = np.random.default_rng(0)
    sig = rng.standard_normal(N)

    n_axis = np.arange(N)
    wow = wow_depth * FS * np.sin(2 * np.pi * wow_rate * n_axis / FS)
    flutter = flutter_depth * FS * np.sin(2 * np.pi * flutter_rate * n_axis / FS)
    delay_curve = np.maximum(1.0, delay_time * FS + wow + flutter)
    expected = fractional_delay_line(sig, delay_curve)

    _, out = tape_delay(
        sig, FS,
        delay_time=delay_time,
        feedback=0.0,
        flutter_rate=flutter_rate,
        flutter_depth=flutter_depth,
        wow_rate=wow_rate,
        wow_depth=wow_depth,
        saturation=0.0,
    )

    np.testing.assert_allclose(out, expected, atol=1e-10)


# --- flutter frequency deviation ---

def test_flutter_produces_frequency_deviation():
    """Flutter at a known rate must produce measurable pitch variation in the echo.

    The delay modulation d(n) = flutter_depth * sin(2π * flutter_rate * n / fs)
    creates an instantaneous frequency deviation Δf ≈ f * 2π * flutter_rate * flutter_depth.
    This manifests as a time-varying spectral centroid in the echo spectrogram.
    """
    freq = 440.0
    flutter_rate = 10.0
    flutter_depth = 0.001  # 1 ms — gives ±~28 Hz deviation at 440 Hz

    sig = _sine(freq=freq, duration=2.0)
    _, out = tape_delay(
        sig, FS,
        delay_time=0.1,
        feedback=0.0,
        flutter_rate=flutter_rate,
        flutter_depth=flutter_depth,
        wow_depth=0.0,
        saturation=0.0,
    )

    # Skip the pre-echo silence, then compute per-frame spectral centroid
    skip = int(0.15 * FS)
    echo = out[skip:]
    win_size = 2048
    hop = 512
    window = np.hanning(win_size)
    freqs = np.fft.rfftfreq(win_size, 1.0 / FS)
    nframes = (len(echo) - win_size) // hop

    centroids = np.empty(nframes)
    for i in range(nframes):
        frame = echo[i * hop: i * hop + win_size] * window
        spec = np.abs(np.fft.rfft(frame))
        centroids[i] = np.sum(freqs * spec) / (np.sum(spec) + 1e-10)

    # Flutter should produce at least a few Hz of centroid variation
    assert np.std(centroids) > 1.0


# --- finite output with high feedback ---

def test_finite_with_feedback_09():
    """Output must contain no NaN or Inf values when feedback=0.9."""
    sig = _impulse(duration=1.5)
    _, out = tape_delay(
        sig, FS,
        delay_time=0.1,
        feedback=0.9,
        flutter_depth=0.0001,
        wow_depth=0.001,
        saturation=0.0,
    )
    assert np.all(np.isfinite(out))


def test_finite_with_saturation():
    """Output must remain finite when saturation is active and feedback is high."""
    sig = _impulse(duration=1.5)
    _, out = tape_delay(
        sig, FS,
        delay_time=0.1,
        feedback=0.8,
        flutter_depth=0.0001,
        wow_depth=0.001,
        saturation=0.5,
    )
    assert np.all(np.isfinite(out))


# --- saturation effect ---

def test_saturation_changes_output():
    """Non-zero saturation must produce a different output than saturation=0."""
    sig = _impulse(duration=0.5)
    _, out_linear = tape_delay(
        sig, FS, delay_time=0.05, feedback=0.7, saturation=0.0,
        flutter_depth=0.0, wow_depth=0.0,
    )
    _, out_sat = tape_delay(
        sig, FS, delay_time=0.05, feedback=0.7, saturation=1.0,
        flutter_depth=0.0, wow_depth=0.0,
    )
    assert not np.allclose(out_linear, out_sat)


# --- wow / flutter modulation changes output ---

def test_flutter_changes_output():
    """Non-zero flutter_depth must produce a different output than flutter_depth=0."""
    sig = _sine(freq=440.0, duration=0.5)
    _, out_static = tape_delay(
        sig, FS, delay_time=0.1, feedback=0.0,
        flutter_depth=0.0, wow_depth=0.0,
    )
    _, out_flutter = tape_delay(
        sig, FS, delay_time=0.1, feedback=0.0,
        flutter_rate=10.0, flutter_depth=0.001, wow_depth=0.0,
    )
    assert not np.allclose(out_static, out_flutter)


def test_wow_changes_output():
    """Non-zero wow_depth must produce a different output than wow_depth=0."""
    sig = _sine(freq=440.0, duration=0.5)
    _, out_static = tape_delay(
        sig, FS, delay_time=0.1, feedback=0.0,
        flutter_depth=0.0, wow_depth=0.0,
    )
    _, out_wow = tape_delay(
        sig, FS, delay_time=0.1, feedback=0.0,
        flutter_depth=0.0, wow_rate=0.5, wow_depth=0.003,
    )
    assert not np.allclose(out_static, out_wow)


# --- silence for zero input ---

def test_zero_input_gives_zero_output():
    """A silent input must produce a silent echo trail."""
    sig = np.zeros(int(0.5 * FS))
    _, out = tape_delay(sig, FS)
    np.testing.assert_allclose(out, np.zeros_like(sig), atol=1e-12)
