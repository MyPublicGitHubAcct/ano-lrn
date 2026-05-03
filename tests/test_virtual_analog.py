import numpy as np
import pytest

from python.virtual_analog import analog_saturate, diode_clip, moog_ladder
from python.generators import generate_impulse, generate_sine

FS = 44100


def _impulse(n=512):
    _, imp = generate_impulse(fs=FS, duration=n / FS)
    return imp


def _sine(freq=440.0, n=2048):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


# ── Shape ─────────────────────────────────────────────────────────────────────

def test_moog_ladder_shape():
    sig = _sine()
    assert moog_ladder(sig, cutoff=1000.0, fs=FS).shape == sig.shape


def test_diode_clip_shape():
    sig = _sine()
    assert diode_clip(sig).shape == sig.shape


def test_analog_saturate_shape():
    sig = _sine()
    assert analog_saturate(sig).shape == sig.shape


# ── Moog ladder ───────────────────────────────────────────────────────────────

def test_moog_attenuates_high_frequencies():
    sig_low = _sine(freq=100.0)
    sig_high = _sine(freq=10000.0)
    out_low = moog_ladder(sig_low, cutoff=1000.0, fs=FS)
    out_high = moog_ladder(sig_high, cutoff=1000.0, fs=FS)
    # Signal far below cutoff should pass much better than signal far above.
    rms_low = _rms(out_low[len(out_low) // 2:])
    rms_high = _rms(out_high[len(out_high) // 2:])
    assert rms_low > rms_high * 2


def test_moog_zero_resonance_no_self_oscillation():
    sig = np.zeros(2048)
    sig[0] = 1.0  # impulse
    out = moog_ladder(sig, cutoff=1000.0, fs=FS, resonance=0.0)
    assert np.all(np.isfinite(out))


# ── Diode clip ────────────────────────────────────────────────────────────────

def test_diode_clip_positive_hard_clipped():
    sig = np.array([0.5, 0.7, 0.9, 1.5])
    out = diode_clip(sig, threshold=0.7)
    assert out[2] == pytest.approx(0.7, abs=1e-6)
    assert out[3] == pytest.approx(0.7, abs=1e-6)


def test_diode_clip_positive_unchanged_below_threshold():
    sig = np.array([0.3, 0.5])
    out = diode_clip(sig, threshold=0.7)
    np.testing.assert_allclose(out, sig, atol=1e-12)


def test_diode_clip_asymmetric():
    # Positive and negative clips should differ (diode asymmetry).
    t = np.linspace(-1.5, 1.5, 200)
    out = diode_clip(t, threshold=0.7)
    # Positive peak should equal threshold exactly; negative should be softer.
    pos_peak = np.max(out)
    neg_peak = np.min(out)
    assert pos_peak == pytest.approx(0.7, abs=1e-6)
    assert neg_peak > -0.7  # soft negative clip does not reach -threshold


# ── Analog saturate ───────────────────────────────────────────────────────────

def test_analog_saturate_zero_input_zero_output():
    np.testing.assert_allclose(analog_saturate(np.array([0.0])), [0.0], atol=1e-12)


def test_analog_saturate_bounded():
    sig = np.linspace(-10.0, 10.0, 500)
    out = analog_saturate(sig, drive=5.0)
    assert np.max(np.abs(out)) < 1.0


def test_analog_saturate_odd_symmetry():
    sig = np.linspace(-1.0, 1.0, 100)
    np.testing.assert_allclose(analog_saturate(-sig), -analog_saturate(sig), atol=1e-12)


def test_analog_saturate_monotone():
    sig = np.linspace(-1.0, 1.0, 100)
    out = analog_saturate(sig)
    assert np.all(np.diff(out) > 0)
