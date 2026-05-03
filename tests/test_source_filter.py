import numpy as np
import pytest

from python.source_filter import formant_filter, lpc_coeffs, lpc_synthesize
from python.generators import generate_impulse

FS = 44100


def _impulse(n=1024):
    _, imp = generate_impulse(fs=FS, duration=n / FS)
    return imp


def _sine(freq=440.0, n=2048):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


# ── Shape ─────────────────────────────────────────────────────────────────────

def test_lpc_coeffs_length():
    order = 12
    sig = _sine()
    assert lpc_coeffs(sig, order).shape == (order,)


def test_lpc_synthesize_shape():
    sig = _sine()
    coeffs = lpc_coeffs(sig, order=8)
    excitation = _impulse(len(sig))
    out = lpc_synthesize(excitation, coeffs)
    assert out.shape == excitation.shape


def test_formant_filter_shape():
    sig = _impulse()
    out = formant_filter(sig, formant_freqs=[500.0, 1500.0], bandwidths=[80.0, 120.0], fs=FS)
    assert out.shape == sig.shape


# ── LPC coefficients ──────────────────────────────────────────────────────────

def test_lpc_coeffs_sine_single_pole():
    # A pure sine wave is modelled exactly by a 2nd-order all-pole filter.
    # With order=2, the predictor coefficients should reconstruct each sample
    # from the two preceding samples with near-zero error.
    freq = 1000.0
    n = 4096
    sig = np.sin(2 * np.pi * freq * np.arange(n) / FS)
    coeffs = lpc_coeffs(sig, order=2)
    assert len(coeffs) == 2
    # sig[i] ≈ a[0]*sig[i-1] + a[1]*sig[i-2]
    for i in range(10, 20):
        predicted = coeffs[0] * sig[i - 1] + coeffs[1] * sig[i - 2]
        assert abs(predicted - sig[i]) < 0.05


def test_lpc_coeffs_different_orders_different_length():
    sig = _sine()
    for order in [4, 8, 16]:
        assert len(lpc_coeffs(sig, order)) == order


# ── LPC synthesize ────────────────────────────────────────────────────────────

def test_lpc_synthesize_output_dtype_float():
    sig = _sine()
    coeffs = lpc_coeffs(sig, order=8)
    out = lpc_synthesize(_impulse(len(sig)), coeffs)
    assert out.dtype == float


# ── Formant filter ────────────────────────────────────────────────────────────

def test_formant_filter_boosts_formant_frequency():
    # A resonator at 1000 Hz should pass a 1000 Hz sine more than DC.
    sig = _impulse(2048)
    out = formant_filter(sig, formant_freqs=[1000.0], bandwidths=[100.0], fs=FS)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), d=1.0 / FS)
    # Energy around 1000 Hz bin
    mask_1k = (freqs >= 900) & (freqs <= 1100)
    mask_dc = freqs < 100
    energy_1k = np.sum(spectrum[mask_1k] ** 2)
    energy_dc = np.sum(spectrum[mask_dc] ** 2)
    assert energy_1k > energy_dc
