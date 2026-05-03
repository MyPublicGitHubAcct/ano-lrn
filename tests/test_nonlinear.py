import numpy as np
import pytest

from python.nonlinear import bitcrush, hard_clip, soft_clip, waveshape

FS = 44100


def _ramp(n=1024):
    return np.linspace(-2.0, 2.0, n)


# ── Shape ─────────────────────────────────────────────────────────────────────

def test_hard_clip_shape():
    sig = _ramp()
    assert hard_clip(sig).shape == sig.shape


def test_soft_clip_shape():
    sig = _ramp()
    assert soft_clip(sig).shape == sig.shape


def test_waveshape_shape():
    sig = _ramp()
    assert waveshape(sig, [0.0, 1.0]).shape == sig.shape


def test_bitcrush_shape():
    sig = _ramp()
    assert bitcrush(sig).shape == sig.shape


# ── Hard clip ─────────────────────────────────────────────────────────────────

def test_hard_clip_clamps_to_threshold():
    sig = _ramp()
    out = hard_clip(sig, threshold=1.0)
    assert np.max(out) == pytest.approx(1.0)
    assert np.min(out) == pytest.approx(-1.0)


def test_hard_clip_identity_within_threshold():
    sig = np.linspace(-0.5, 0.5, 100)
    np.testing.assert_array_equal(hard_clip(sig, threshold=1.0), sig)


def test_hard_clip_symmetric():
    sig = _ramp()
    out = hard_clip(sig)
    np.testing.assert_allclose(hard_clip(-sig), -out, atol=1e-12)


# ── Soft clip ─────────────────────────────────────────────────────────────────

def test_soft_clip_bounded():
    sig = _ramp() * 10
    out = soft_clip(sig)
    assert np.max(np.abs(out)) <= 1.0


def test_soft_clip_monotone():
    sig = np.linspace(-5.0, 5.0, 500)
    out = soft_clip(sig)
    assert np.all(np.diff(out) > 0)


def test_soft_clip_symmetric():
    sig = _ramp()
    np.testing.assert_allclose(soft_clip(-sig), -soft_clip(sig), atol=1e-12)


def test_soft_clip_drive_increases_saturation():
    # Higher drive means more saturation (output closer to ±1) for the same input.
    sig = np.array([0.5])
    out_low = soft_clip(sig, drive=1.0)
    out_high = soft_clip(sig, drive=5.0)
    assert out_high > out_low


# ── Waveshape ─────────────────────────────────────────────────────────────────

def test_waveshape_identity_polynomial():
    # coeffs = [0, 1] → y = x
    sig = _ramp()
    np.testing.assert_allclose(waveshape(sig, [0.0, 1.0]), sig, atol=1e-12)


def test_waveshape_dc_offset():
    # coeffs = [2, 0] → y = 2
    sig = _ramp()
    np.testing.assert_allclose(waveshape(sig, [2.0, 0.0]), np.full_like(sig, 2.0), atol=1e-12)


def test_waveshape_square():
    # coeffs = [0, 0, 1] → y = x^2 (all positive)
    sig = np.linspace(-1.0, 1.0, 100)
    out = waveshape(sig, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(out, sig ** 2, atol=1e-12)


# ── Bitcrush ──────────────────────────────────────────────────────────────────

def test_bitcrush_reduces_unique_values():
    sig = np.linspace(-1.0, 1.0, 10000)
    out_8 = bitcrush(sig, bits=8)
    out_4 = bitcrush(sig, bits=4)
    assert len(np.unique(out_4)) < len(np.unique(out_8))


def test_bitcrush_1bit_binary():
    sig = np.linspace(-1.0, 1.0, 256)
    out = bitcrush(sig, bits=1)
    assert set(np.unique(out)).issubset({-1.0, 0.0, 1.0})


def test_bitcrush_preserves_zero():
    np.testing.assert_allclose(bitcrush(np.array([0.0]), bits=8), [0.0])
