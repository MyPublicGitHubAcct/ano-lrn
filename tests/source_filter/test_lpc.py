import numpy as np

from python.source_filter import lpc_coeffs, lpc_synthesize
from python.generators import generate_impulse

FS = 44100


def _impulse(n=1024):
    _, imp = generate_impulse(fs=FS, duration=n / FS)
    return imp


def _sine(freq=440.0, n=2048):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_lpc_coeffs_length():
    order = 12
    assert lpc_coeffs(_sine(), order).shape == (order,)


def test_lpc_synthesize_shape():
    sig = _sine()
    coeffs = lpc_coeffs(sig, order=8)
    excitation = _impulse(len(sig))
    out = lpc_synthesize(excitation, coeffs)
    assert out.shape == excitation.shape


def test_lpc_coeffs_sine_single_pole():
    freq = 1000.0
    n = 4096
    sig = np.sin(2 * np.pi * freq * np.arange(n) / FS)
    coeffs = lpc_coeffs(sig, order=2)
    assert len(coeffs) == 2
    for i in range(10, 20):
        predicted = coeffs[0] * sig[i - 1] + coeffs[1] * sig[i - 2]
        assert abs(predicted - sig[i]) < 0.05


def test_lpc_coeffs_different_orders_different_length():
    sig = _sine()
    for order in [4, 8, 16]:
        assert len(lpc_coeffs(sig, order)) == order


def test_lpc_synthesize_output_dtype_float():
    sig = _sine()
    coeffs = lpc_coeffs(sig, order=8)
    out = lpc_synthesize(_impulse(len(sig)), coeffs)
    assert out.dtype == float
