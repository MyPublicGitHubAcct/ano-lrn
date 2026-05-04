import numpy as np
import pytest

from python.virtual_analog import analog_saturate, diode_clip

FS = 44100


def _sine(freq=440.0, n=2048):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_diode_clip_shape():
    assert diode_clip(_sine()).shape == _sine().shape


def test_analog_saturate_shape():
    assert analog_saturate(_sine()).shape == _sine().shape


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
    t = np.linspace(-1.5, 1.5, 200)
    out = diode_clip(t, threshold=0.7)
    pos_peak = np.max(out)
    neg_peak = np.min(out)
    assert pos_peak == pytest.approx(0.7, abs=1e-6)
    assert neg_peak > -0.7


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
