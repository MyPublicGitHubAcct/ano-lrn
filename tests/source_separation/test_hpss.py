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
    sig = _sine(n=4096)
    h, p = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=7)
    assert h.shape == sig.shape
    assert p.shape == sig.shape


def test_hpss_harmonic_plus_percussive_approx_original():
    sig = _sine(n=4096)
    h, p = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=7)
    rms_diff = _rms(h + p - sig)
    rms_sig = _rms(sig)
    assert rms_diff < rms_sig * 0.1


def test_hpss_harmonic_component_non_trivial():
    sig = _sine(freq=440.0, n=4096)
    h, _ = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=7)
    assert _rms(h) > 0.01


def test_hpss_masks_sum_to_unity():
    sig = _sine(n=4096)
    h, p = hpss(sig, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=7)
    combined_rms = _rms(h + p)
    original_rms = _rms(sig)
    assert abs(combined_rms - original_rms) < 0.1 * original_rms
