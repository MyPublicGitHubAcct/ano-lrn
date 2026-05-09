import numpy as np
import pytest

from python.nonlinear import hard_clip, soft_clip


def _ramp(n=1024):
    return np.linspace(-2.0, 2.0, n)


def test_hard_clip_shape():
    """hard_clip must return an array of the same length as the input."""
    assert hard_clip(_ramp()).shape == _ramp().shape


def test_soft_clip_shape():
    """soft_clip must return an array of the same length as the input."""
    assert soft_clip(_ramp()).shape == _ramp().shape


def test_hard_clip_clamps_to_threshold():
    """The output peak must equal exactly ±threshold; any overshoot means the clamp is not applied."""
    out = hard_clip(_ramp(), threshold=1.0)
    assert np.max(out) == pytest.approx(1.0)
    assert np.min(out) == pytest.approx(-1.0)


def test_hard_clip_identity_within_threshold():
    """Samples within ±threshold must pass through unmodified — hard clip is linear in its passband."""
    sig = np.linspace(-0.5, 0.5, 100)
    np.testing.assert_array_equal(hard_clip(sig, threshold=1.0), sig)


def test_hard_clip_symmetric():
    """Hard clipping must have odd symmetry: f(−x) = −f(x) for all x."""
    sig = _ramp()
    out = hard_clip(sig)
    np.testing.assert_allclose(hard_clip(-sig), -out, atol=1e-12)


def test_soft_clip_bounded():
    """With any drive value the soft-clip output must remain within [−1, +1]."""
    out = soft_clip(_ramp() * 10)
    assert np.max(np.abs(out)) <= 1.0


def test_soft_clip_monotone():
    """A soft clipper must be strictly monotone; non-monotonicity creates foldback distortion artefacts."""
    sig = np.linspace(-5.0, 5.0, 500)
    out = soft_clip(sig)
    assert np.all(np.diff(out) > 0)


def test_soft_clip_symmetric():
    """Soft clipping must have odd symmetry: f(−x) = −f(x) for all x."""
    sig = _ramp()
    np.testing.assert_allclose(soft_clip(-sig), -soft_clip(sig), atol=1e-12)


def test_soft_clip_drive_increases_saturation():
    """Higher drive must push the output closer to the saturation limit for the same input level."""
    sig = np.array([0.5])
    out_low = soft_clip(sig, drive=1.0)
    out_high = soft_clip(sig, drive=5.0)
    assert out_high > out_low


# --- parameter range tests ---

def test_hard_clip_zero_threshold_produces_zeros():
    """threshold=0 clips every sample to zero, confirming the edge case is handled without division error."""
    out = hard_clip(_ramp(), threshold=0.0)
    np.testing.assert_array_equal(out, np.zeros(len(_ramp())))


def test_soft_clip_zero_drive_produces_zeros():
    """drive=0 scales the input to zero before the nonlinearity, so the output must be all zeros."""
    out = soft_clip(_ramp(), drive=0.0)
    np.testing.assert_array_equal(out, np.zeros(len(_ramp())))
