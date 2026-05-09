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
    """lpc_coeffs must return exactly `order` coefficients regardless of signal length."""
    order = 12
    assert lpc_coeffs(_sine(), order).shape == (order,)


def test_lpc_synthesize_shape():
    """lpc_synthesize must return an array with the same length as the excitation signal."""
    sig = _sine()
    coeffs = lpc_coeffs(sig, order=8)
    excitation = _impulse(len(sig))
    out = lpc_synthesize(excitation, coeffs)
    assert out.shape == excitation.shape


def test_lpc_coeffs_sine_single_pole():
    """A pure sine has a 2-pole all-pole model; order-2 LPC coefficients must predict it to within 0.05 at mid-signal."""
    freq = 1000.0
    n = 4096
    sig = np.sin(2 * np.pi * freq * np.arange(n) / FS)
    coeffs = lpc_coeffs(sig, order=2)
    assert len(coeffs) == 2
    for i in range(10, 20):
        predicted = coeffs[0] * sig[i - 1] + coeffs[1] * sig[i - 2]
        assert abs(predicted - sig[i]) < 0.05


def test_lpc_coeffs_different_orders_different_length():
    """Output length must always equal the requested order, even when the order varies."""
    sig = _sine()
    for order in [4, 8, 16]:
        assert len(lpc_coeffs(sig, order)) == order


def test_lpc_synthesize_output_dtype_float():
    """lpc_synthesize output must be floating-point so it can be used directly in DSP chains."""
    sig = _sine()
    coeffs = lpc_coeffs(sig, order=8)
    out = lpc_synthesize(_impulse(len(sig)), coeffs)
    assert out.dtype == float


def test_lpc_round_trip_preserves_dominant_frequency():
    """lpc_synthesize driven by an impulse must produce a signal whose dominant frequency matches the original."""
    freq = 440.0
    n = 4096
    sig = _sine(freq=freq, n=n)
    coeffs = lpc_coeffs(sig, order=12)
    out = lpc_synthesize(_impulse(n), coeffs)
    spectrum = np.abs(np.fft.rfft(out))
    freqs_axis = np.fft.rfftfreq(n, d=1.0 / FS)
    dominant = float(freqs_axis[np.argmax(spectrum)])
    assert abs(dominant - freq) < 20.0


def test_lpc_order_one_is_single_sample_prediction():
    """order=1 produces a first-order AR model; the single coefficient predicts each sample from the previous."""
    freq = 500.0
    n = 2048
    sig = np.sin(2 * np.pi * freq * np.arange(n) / FS)
    coeffs = lpc_coeffs(sig, order=1)
    assert len(coeffs) == 1
    # The first-order predictor: predicted[i] = coeffs[0] * sig[i-1]
    predicted = coeffs[0] * sig[5]
    error = abs(predicted - sig[6])
    assert error < 0.2

