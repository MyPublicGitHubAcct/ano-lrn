import numpy as np
import pytest

from python.source_separation import hpss, wiener_filter

FS = 44100
FRAME = 512
HOP = 128


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


# ── Shape ─────────────────────────────────────────────────────────────────────

def test_hpss_output_shapes():
    sig = _sine(n=4096)
    h, p = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=7)
    assert h.shape == sig.shape
    assert p.shape == sig.shape


def test_wiener_filter_output_shape():
    mixture = _sine(440.0, n=4096)
    estimate = _sine(440.0, n=4096) * 0.9
    out = wiener_filter(mixture, estimate, frame_size=FRAME, hop_size=HOP)
    assert out.shape == mixture.shape


# ── HPSS ──────────────────────────────────────────────────────────────────────

def test_hpss_harmonic_plus_percussive_approx_original():
    sig = _sine(n=4096)
    h, p = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=7)
    # Soft mask: H_mask + P_mask = 1, so h + p ≈ original in time domain.
    rms_diff = _rms(h + p - sig)
    rms_sig = _rms(sig)
    assert rms_diff < rms_sig * 0.1


def test_hpss_harmonic_component_non_trivial():
    sig = _sine(freq=440.0, n=4096)
    h, _ = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=7)
    assert _rms(h) > 0.01


def test_hpss_masks_sum_to_unity():
    # Both outputs are non-negative fractions of the original signal power.
    sig = _sine(n=4096)
    h, p = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=7)
    combined_rms = _rms(h + p)
    original_rms = _rms(sig)
    # Combined should be within 10% of original (masks sum to 1 in STFT domain).
    assert abs(combined_rms - original_rms) < 0.1 * original_rms


# ── Wiener filter ─────────────────────────────────────────────────────────────

def test_wiener_filter_identical_estimate_passes_signal():
    sig = _sine(n=4096)
    out = wiener_filter(sig, sig, frame_size=FRAME, hop_size=HOP)
    # Mask = |sig|^2 / |sig|^2 = 1 → output ≈ input.
    assert _rms(out) == pytest.approx(_rms(sig), rel=0.1)


def test_wiener_filter_zero_estimate_suppresses_signal():
    sig = _sine(n=4096)
    estimate = np.zeros_like(sig)
    out = wiener_filter(sig, estimate, frame_size=FRAME, hop_size=HOP)
    assert _rms(out) < _rms(sig) * 0.01
