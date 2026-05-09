import numpy as np

from python.source_separation import hpss

FS = 44100
FRAME = 512
HOP = 128


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def test_hpss_output_shapes():
    """hpss must return two arrays (harmonic, percussive) each matching the input length."""
    sig = _sine(n=4096)
    h, p = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=7)
    assert h.shape == sig.shape
    assert p.shape == sig.shape


def test_hpss_harmonic_plus_percussive_approx_original():
    """The sum h + p must reconstruct the original to within 10% RMS (soft-mask property)."""
    sig = _sine(n=4096)
    h, p = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=7)
    rms_diff = _rms(h + p - sig)
    rms_sig = _rms(sig)
    assert rms_diff < rms_sig * 0.1


def test_hpss_harmonic_component_non_trivial():
    """A pure sine is maximally harmonic; the harmonic component must contain most of the signal energy."""
    sig = _sine(freq=440.0, n=4096)
    h, _ = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=7)
    assert _rms(h) > 0.01


def test_hpss_masks_sum_to_unity():
    """The combined RMS of h + p must be within 10% of the original RMS (energy conservation)."""
    sig = _sine(n=4096)
    h, p = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=7)
    combined_rms = _rms(h + p)
    original_rms = _rms(sig)
    assert abs(combined_rms - original_rms) < 0.1 * original_rms


def test_hpss_pure_sine_classified_as_harmonic():
    """A pure sine is maximally harmonic; the harmonic component must contain far more energy than percussive."""
    sig = _sine(freq=440.0, n=4096)
    h, p = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=17)
    assert _rms(h) > _rms(p) * 3


def test_hpss_larger_kernel_increases_harmonic_contrast():
    """A larger kernel_size sharpens separation; harmonic/percussive contrast must be greater with a larger kernel."""
    sig = _sine(freq=440.0, n=4096)
    h_small, p_small = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=5)
    h_large, p_large = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=31)
    ratio_small = _rms(h_small) / (_rms(p_small) + 1e-12)
    ratio_large = _rms(h_large) / (_rms(p_large) + 1e-12)
    assert ratio_large > ratio_small

