import numpy as np
import pytest

from python.nonlinear import hard_clip, soft_clip


def _ramp(n=1024):
    return np.linspace(-2.0, 2.0, n)


def test_hard_clip_shape():
    assert hard_clip(_ramp()).shape == _ramp().shape


def test_soft_clip_shape():
    assert soft_clip(_ramp()).shape == _ramp().shape


def test_hard_clip_clamps_to_threshold():
    out = hard_clip(_ramp(), threshold=1.0)
    assert np.max(out) == pytest.approx(1.0)
    assert np.min(out) == pytest.approx(-1.0)


def test_hard_clip_identity_within_threshold():
    sig = np.linspace(-0.5, 0.5, 100)
    np.testing.assert_array_equal(hard_clip(sig, threshold=1.0), sig)


def test_hard_clip_symmetric():
    sig = _ramp()
    out = hard_clip(sig)
    np.testing.assert_allclose(hard_clip(-sig), -out, atol=1e-12)


def test_soft_clip_bounded():
    out = soft_clip(_ramp() * 10)
    assert np.max(np.abs(out)) <= 1.0


def test_soft_clip_monotone():
    sig = np.linspace(-5.0, 5.0, 500)
    out = soft_clip(sig)
    assert np.all(np.diff(out) > 0)


def test_soft_clip_symmetric():
    sig = _ramp()
    np.testing.assert_allclose(soft_clip(-sig), -soft_clip(sig), atol=1e-12)


def test_soft_clip_drive_increases_saturation():
    sig = np.array([0.5])
    out_low = soft_clip(sig, drive=1.0)
    out_high = soft_clip(sig, drive=5.0)
    assert out_high > out_low
