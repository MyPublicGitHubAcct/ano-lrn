import numpy as np

from python.nonlinear import bitcrush, waveshape


def _ramp(n=1024):
    return np.linspace(-2.0, 2.0, n)


def test_waveshape_shape():
    """waveshape must return an array of the same length as the input."""
    assert waveshape(_ramp(), [0.0, 1.0]).shape == _ramp().shape


def test_bitcrush_shape():
    """bitcrush must return an array of the same length as the input."""
    assert bitcrush(_ramp()).shape == _ramp().shape


def test_waveshape_identity_polynomial():
    """Polynomial [0, 1] is f(x) = x, the identity; waveshape must return the input unchanged."""
    sig = _ramp()
    np.testing.assert_allclose(waveshape(sig, [0.0, 1.0]), sig, atol=1e-12)


def test_waveshape_dc_offset():
    """Polynomial [c, 0] = c is a constant; waveshape must return a constant array equal to c."""
    sig = _ramp()
    np.testing.assert_allclose(waveshape(sig, [2.0, 0.0]), np.full_like(sig, 2.0), atol=1e-12)


def test_waveshape_square():
    """Polynomial [0, 0, 1] = x²; waveshape must return the element-wise square of the input."""
    sig = np.linspace(-1.0, 1.0, 100)
    out = waveshape(sig, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(out, sig ** 2, atol=1e-12)


def test_bitcrush_reduces_unique_values():
    """Lower bit depth must produce fewer unique quantisation levels than higher bit depth."""
    sig = np.linspace(-1.0, 1.0, 10000)
    out_8 = bitcrush(sig, bits=8)
    out_4 = bitcrush(sig, bits=4)
    assert len(np.unique(out_4)) < len(np.unique(out_8))


def test_bitcrush_1bit_binary():
    """1-bit quantisation must produce only values in {−1, 0, +1} (mid-tread or mid-riser quantiser)."""
    sig = np.linspace(-1.0, 1.0, 256)
    out = bitcrush(sig, bits=1)
    assert set(np.unique(out)).issubset({-1.0, 0.0, 1.0})


def test_bitcrush_preserves_zero():
    """Zero must map to zero at any bit depth; any quantisation error at exactly zero indicates an offset bug."""
    np.testing.assert_allclose(bitcrush(np.array([0.0]), bits=8), [0.0])


def test_bitcrush_output_within_unit_range():
    """bitcrush output must stay within [−1, +1] for any input within [−1, +1]."""
    sig = np.linspace(-1.0, 1.0, 1024)
    for bits in [2, 4, 8, 16]:
        out = bitcrush(sig, bits=bits)
        assert np.max(out) <= 1.0 + 1e-9
        assert np.min(out) >= -1.0 - 1e-9


def test_waveshape_chebyshev_order2_produces_double_frequency():
    """Chebyshev T2(x) = 2x^2 − 1; applied to a sine at f it produces energy primarily at 2f."""
    n = 44100
    freq = 200.0
    fs = 44100
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * freq * t)
    # T2(x) = 2x^2 - 1 → coefficients: [-1, 0, 2]
    out = waveshape(sig, [-1.0, 0.0, 2.0])
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    idx_f = np.argmin(np.abs(freqs - freq))
    idx_2f = np.argmin(np.abs(freqs - 2 * freq))
    assert spectrum[idx_2f] > spectrum[idx_f] * 2

