import numpy as np

from python.nonlinear import bitcrush, waveshape


def _ramp(n=1024):
    return np.linspace(-2.0, 2.0, n)


def test_waveshape_shape():
    assert waveshape(_ramp(), [0.0, 1.0]).shape == _ramp().shape


def test_bitcrush_shape():
    assert bitcrush(_ramp()).shape == _ramp().shape


def test_waveshape_identity_polynomial():
    sig = _ramp()
    np.testing.assert_allclose(waveshape(sig, [0.0, 1.0]), sig, atol=1e-12)


def test_waveshape_dc_offset():
    sig = _ramp()
    np.testing.assert_allclose(waveshape(sig, [2.0, 0.0]), np.full_like(sig, 2.0), atol=1e-12)


def test_waveshape_square():
    sig = np.linspace(-1.0, 1.0, 100)
    out = waveshape(sig, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(out, sig ** 2, atol=1e-12)


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
