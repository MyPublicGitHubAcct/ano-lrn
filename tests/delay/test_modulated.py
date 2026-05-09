import numpy as np
import pytest

from python.delay import chorus, flanger
from python.generators import generate_impulse

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def _impulse(n=22050):
    _, imp = generate_impulse(fs=FS, duration=n / FS)
    return imp


# --- shape ---

def test_chorus_shape():
    """chorus must return an array of the same length as the input."""
    sig = _sine()
    assert chorus(sig, fs=FS).shape == sig.shape


def test_flanger_shape():
    """flanger must return an array of the same length as the input."""
    sig = _sine()
    assert flanger(sig, fs=FS).shape == sig.shape


# --- dry pass-through ---

def test_chorus_dry_is_identity():
    """mix=0 means no wet signal is added; output must equal the input exactly."""
    sig = _sine()
    np.testing.assert_allclose(chorus(sig, mix=0.0, fs=FS), sig, atol=1e-12)


def test_flanger_dry_is_identity():
    """mix=0 means no wet signal is added; output must equal the input exactly."""
    sig = _sine()
    np.testing.assert_allclose(flanger(sig, mix=0.0, fs=FS), sig, atol=1e-12)


# --- chorus wet path ---

def test_chorus_wet_equals_fixed_delay():
    """mix=1 with depth_ms=0 (no LFO swing) must equal a simple fractional delay of the input."""
    from python.delay.line import fractional_delay_line

    sig = _sine()
    d_ms = 20.0
    out = chorus(sig, rate=1.0, depth_ms=0.0, delay_ms=d_ms, mix=1.0, fs=FS)
    d_samples = d_ms * FS / 1000.0
    expected = fractional_delay_line(sig, d_samples)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_chorus_rms_within_3db():
    """Chorus output RMS must stay within 3 dB of input RMS for any mix value."""
    sig = _sine(n=FS)
    out = chorus(sig, rate=1.0, depth_ms=2.5, delay_ms=25.0, mix=0.5, fs=FS)
    rms_in = np.sqrt(np.mean(sig ** 2))
    rms_out = np.sqrt(np.mean(out ** 2))
    ratio_db = 20.0 * np.log10(rms_out / rms_in)
    assert abs(ratio_db) <= 3.0


def test_chorus_non_silent():
    """A non-silent input must produce a non-silent chorus output."""
    sig = _sine()
    out = chorus(sig, fs=FS)
    assert np.max(np.abs(out)) > 0.1


def test_chorus_modulation_changes_output():
    """A non-zero LFO depth must produce a different output than depth_ms=0."""
    sig = _sine()
    out_static = chorus(sig, rate=1.0, depth_ms=0.0, delay_ms=25.0, mix=0.5, fs=FS)
    out_modulated = chorus(sig, rate=1.0, depth_ms=3.0, delay_ms=25.0, mix=0.5, fs=FS)
    assert not np.allclose(out_static, out_modulated)


# --- flanger comb structure ---

def test_flanger_static_comb_notch():
    """With depth_ms=0 (static delay), mix=0.5, feedback=0, the feedforward comb must
    produce a notch near fs/(2*D) Hz and a peak near fs/D Hz."""
    imp = _impulse()
    d_ms = 5.0
    d_samples = d_ms * FS / 1000.0  # 220.5 samples → notch near 100 Hz, peak near 200 Hz
    out = flanger(imp, rate=0.0, depth_ms=0.0, delay_ms=d_ms, feedback=0.0, mix=0.5, fs=FS)

    freqs = np.fft.rfftfreq(len(out), 1.0 / FS)
    spectrum = np.abs(np.fft.rfft(out))

    # Notch: average energy in ±5 Hz band around 100 Hz
    mask_notch = np.abs(freqs - FS / (2 * d_samples)) < 5.0
    # Peak: average energy in ±5 Hz band around 200 Hz
    mask_peak = np.abs(freqs - FS / d_samples) < 5.0

    assert spectrum[mask_notch].mean() < spectrum[mask_peak].mean()


def test_flanger_feedback_deepens_notches():
    """Positive feedback increases peak amplitude relative to the zero-feedback case."""
    imp = _impulse()
    d_ms = 5.0
    d_samples = d_ms * FS / 1000.0

    out_no_fb = flanger(imp, rate=0.0, depth_ms=0.0, delay_ms=d_ms, feedback=0.0, mix=0.5, fs=FS)
    out_with_fb = flanger(imp, rate=0.0, depth_ms=0.0, delay_ms=d_ms, feedback=0.7, mix=0.5, fs=FS)

    spectrum_no = np.abs(np.fft.rfft(out_no_fb))
    spectrum_fb = np.abs(np.fft.rfft(out_with_fb))
    freqs = np.fft.rfftfreq(len(out_no_fb), 1.0 / FS)

    # Peak near fs/D should be taller with feedback
    mask_peak = np.abs(freqs - FS / d_samples) < 5.0
    assert spectrum_fb[mask_peak].mean() > spectrum_no[mask_peak].mean()


def test_flanger_mix_one_feedback_zero_is_delayed_signal():
    """mix=1, feedback=0, static delay: output must equal the linearly interpolated
    delayed input (the same as the flanger wet path with no recirculation)."""
    sig = _sine(n=2048)
    d_ms = 2.5
    d_samples = d_ms * FS / 1000.0
    d_int = int(d_samples)
    frac = d_samples - d_int

    out = flanger(sig, rate=0.0, depth_ms=0.0, delay_ms=d_ms, feedback=0.0, mix=1.0, fs=FS)

    # Build expected: linear interpolation of sig delayed by d_samples
    # out[i] = sig[i-d_int]*(1-frac) + sig[i-d_int-1]*frac   (zeros before start)
    pad = np.zeros(d_int + 2)
    padded = np.concatenate([pad, sig])
    n = len(sig)
    expected = np.array([
        padded[i + len(pad) - d_int] * (1.0 - frac) + padded[i + len(pad) - d_int - 1] * frac
        for i in range(n)
    ])
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_flanger_output_finite_with_stable_feedback():
    """|feedback| < 1 must produce a finite output for any input."""
    sig = _sine(n=FS)
    out = flanger(sig, rate=0.5, depth_ms=1.5, delay_ms=2.5, feedback=0.8, mix=0.5, fs=FS)
    assert np.all(np.isfinite(out))


def test_flanger_non_silent():
    """A non-silent input must produce a non-silent flanger output."""
    sig = _sine()
    out = flanger(sig, fs=FS)
    assert np.max(np.abs(out)) > 0.1


def test_flanger_modulation_changes_output():
    """A non-zero LFO depth must produce a different output than depth_ms=0."""
    sig = _sine()
    out_static = flanger(sig, rate=0.5, depth_ms=0.0, delay_ms=2.5, feedback=0.5, mix=0.5, fs=FS)
    out_modulated = flanger(sig, rate=0.5, depth_ms=2.0, delay_ms=2.5, feedback=0.5, mix=0.5, fs=FS)
    assert not np.allclose(out_static, out_modulated)
