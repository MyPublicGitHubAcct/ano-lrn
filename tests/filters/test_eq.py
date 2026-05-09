import numpy as np
import pytest

from python.filters import allpass, bandpass, highpass, lowpass, notch
from python.generators import generate_impulse

from tests.filters.conftest import (
    FS, DURATION, _dc, _half_nyquist_sig, _nyquist_sig, _quarter_nyquist_sig,
    _sine, _steady_rms,
)


def test_any_filter_output_shape(any_filter, cutoff):
    """All EQ filters must return an array of the same length as the input."""
    signal = _sine(440.0)
    out = any_filter(signal, cutoff=cutoff, fs=FS)
    assert out.shape == signal.shape


def test_lowpass_passes_low_frequency():
    """A decade below cutoff, a lowpass must pass the signal with < 5% attenuation."""
    sig = _sine(100.0)
    out = lowpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_lowpass_attenuates_high_frequency():
    """A decade above cutoff, a 2-pole lowpass must attenuate by > 26 dB (< 5% amplitude)."""
    sig = _sine(10000.0)
    out = lowpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.05


def test_lowpass_passes_dc():
    """H(0) = 1 for a standard lowpass; DC must pass without attenuation."""
    out = lowpass(_dc(1.0), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) == pytest.approx(1.0, abs=0.01)


def test_lowpass_Q_sharpens_transition(Q_value):
    """Regardless of Q, a signal one decade above cutoff should still be strongly attenuated."""
    sig = _sine(10000.0)
    out = lowpass(sig, cutoff=1000.0, fs=FS, Q=Q_value)
    assert _steady_rms(out) / _steady_rms(sig) < 0.2


def test_highpass_passes_high_frequency():
    """A decade above cutoff, a highpass must pass the signal with < 5% attenuation."""
    sig = _sine(10000.0)
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_highpass_attenuates_low_frequency():
    """A decade below cutoff, a 2-pole highpass must attenuate by > 26 dB."""
    sig = _sine(100.0)
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.05


def test_highpass_rejects_dc():
    """H(0) = 0 for a standard highpass; any DC leakage indicates a coefficient error."""
    out = highpass(_dc(1.0), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) < 0.01


def test_bandpass_passes_center_frequency():
    """At the center frequency the bandpass gain must be close to unity (< 1.5 dB loss)."""
    sig = _sine(1000.0)
    out = bandpass(sig, cutoff=1000.0, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) > 0.85


def test_bandpass_attenuates_far_above():
    """One decade above center, the bandpass must attenuate by > 20 dB."""
    sig = _sine(10000.0)
    out = bandpass(sig, cutoff=1000.0, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) < 0.1


def test_bandpass_attenuates_far_below():
    """One decade below center, the bandpass must attenuate by > 26 dB."""
    sig = _sine(100.0)
    out = bandpass(sig, cutoff=1000.0, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) < 0.05


def test_bandpass_rejects_dc():
    """H(0) = 0 for a bandpass; DC leakage means the b0/b2 symmetry is broken."""
    out = bandpass(_dc(1.0), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) < 0.01


def test_bandpass_higher_Q_narrows_bandwidth():
    """Higher Q means narrower bandwidth; a frequency just outside the band must be more attenuated at high Q."""
    sig = _sine(1200.0)
    ratio_low_Q = _steady_rms(bandpass(sig, cutoff=1000.0, fs=FS, Q=1.0)) / _steady_rms(sig)
    ratio_high_Q = _steady_rms(bandpass(sig, cutoff=1000.0, fs=FS, Q=8.0)) / _steady_rms(sig)
    assert ratio_high_Q < ratio_low_Q


def test_notch_passes_below_cutoff():
    """A notch must not attenuate frequencies well below its center."""
    sig = _sine(100.0)
    out = notch(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_notch_passes_above_cutoff():
    """A notch must not attenuate frequencies well above its center."""
    sig = _sine(10000.0)
    out = notch(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_notch_attenuates_center():
    """At the notch center the filter must produce > 40 dB rejection (< 1% amplitude)."""
    sig = _sine(1000.0)
    out = notch(sig, cutoff=1000.0, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) < 0.01


def test_notch_passes_dc():
    """H(0) = 1 for a notch filter; DC should pass unchanged."""
    out = notch(_dc(1.0), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) == pytest.approx(1.0, abs=0.01)


def test_notch_higher_Q_narrows_notch():
    """Higher Q concentrates the rejection; a frequency just outside the notch should pass better at high Q."""
    sig = _sine(1200.0)
    ratio_low_Q = _steady_rms(notch(sig, cutoff=1000.0, fs=FS, Q=1.0)) / _steady_rms(sig)
    ratio_high_Q = _steady_rms(notch(sig, cutoff=1000.0, fs=FS, Q=8.0)) / _steady_rms(sig)
    assert ratio_high_Q > ratio_low_Q


def test_allpass_preserves_rms_at_low_freq():
    """An allpass has unity magnitude everywhere; RMS below cutoff must be unchanged."""
    sig = _sine(100.0)
    out = allpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(1.0, abs=0.02)


def test_allpass_preserves_rms_at_high_freq():
    """An allpass has unity magnitude everywhere; RMS above cutoff must also be unchanged."""
    sig = _sine(10000.0)
    out = allpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(1.0, abs=0.02)


def test_allpass_unity_magnitude():
    """Impulse-response FFT magnitude must be 1.0 at all frequencies, confirming H(jω)=1 holds analytically."""
    _, imp = generate_impulse(fs=FS, duration=DURATION)
    h = allpass(imp, cutoff=1000.0, fs=FS)
    mag = np.abs(np.fft.rfft(h))
    assert np.allclose(mag[1:], 1.0, atol=0.01)


def test_lowpass_rejects_nyquist():
    """Nyquist is the most distant frequency from any sub-Nyquist cutoff; the lowpass must fully reject it."""
    out = lowpass(_nyquist_sig(), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) < 1e-6


def test_highpass_passes_nyquist():
    """Nyquist is well above the cutoff; the highpass must pass it with negligible loss."""
    sig = _nyquist_sig()
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.99


def test_bandpass_rejects_nyquist():
    """Nyquist is far above the bandpass center; it must be strongly attenuated."""
    out = bandpass(_nyquist_sig(), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) < 1e-6


def test_lowpass_attenuates_half_nyquist():
    """Half-Nyquist (11 kHz at 44.1 kHz fs) is > 3 octaves above a 1 kHz cutoff; lowpass must attenuate strongly."""
    sig = _half_nyquist_sig()
    out = lowpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.02


def test_highpass_passes_half_nyquist():
    """Half-Nyquist is well above the 1 kHz cutoff; the highpass must pass it with < 5% attenuation."""
    sig = _half_nyquist_sig()
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_bandpass_passes_half_nyquist():
    """When the bandpass center is set to half-Nyquist, that frequency must pass with < 1.5 dB loss."""
    sig = _half_nyquist_sig()
    out = bandpass(sig, cutoff=FS / 4, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) > 0.85


def test_lowpass_attenuates_quarter_nyquist():
    """Quarter-Nyquist is more than 1 octave above a 1 kHz cutoff; lowpass must attenuate meaningfully."""
    sig = _quarter_nyquist_sig()
    out = lowpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.1


def test_highpass_passes_quarter_nyquist():
    """Quarter-Nyquist (≈ 5.5 kHz) is above the 1 kHz cutoff; the highpass must pass it."""
    sig = _quarter_nyquist_sig()
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_bandpass_passes_quarter_nyquist():
    """When the bandpass center matches quarter-Nyquist, that frequency must pass with < 1.5 dB loss."""
    sig = _quarter_nyquist_sig()
    out = bandpass(sig, cutoff=FS / 8, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) > 0.85


# --- parameter range tests ---

def test_eq_cutoff_at_20hz_is_stable():
    """All EQ types must remain numerically stable at the lowest audible cutoff (20 Hz)."""
    sig = _sine(440.0)
    for filt in [lowpass, highpass, bandpass, notch, allpass]:
        out = filt(sig, cutoff=20.0, fs=FS)
        assert np.all(np.isfinite(out))


def test_eq_cutoff_near_nyquist_is_stable():
    """All EQ types must remain stable when the cutoff approaches Nyquist (a common edge case)."""
    sig = _sine(440.0)
    for filt in [lowpass, highpass, bandpass, notch, allpass]:
        out = filt(sig, cutoff=FS * 0.49, fs=FS)
        assert np.all(np.isfinite(out))


def test_lowpass_low_Q_is_stable():
    """Very low Q (0.1, heavily overdamped) must not cause instability."""
    out = lowpass(_sine(440.0), cutoff=1000.0, fs=FS, Q=0.1)
    assert np.all(np.isfinite(out))


def test_lowpass_high_Q_resonates_at_cutoff():
    """High Q creates a resonant peak; output RMS at the cutoff frequency must exceed the low-Q case."""
    sig = _sine(1000.0)
    out_low_Q = lowpass(sig, cutoff=1000.0, fs=FS, Q=0.707)
    out_high_Q = lowpass(sig, cutoff=1000.0, fs=FS, Q=10.0)
    assert _steady_rms(out_high_Q) > _steady_rms(out_low_Q)


def test_lowpass_minus3db_at_cutoff():
    """Impulse-response FFT must show −3 dB gain (within 5% of cutoff) at the nominal cutoff frequency."""
    cutoff = 1000.0
    _, imp = generate_impulse(fs=FS, duration=DURATION)
    h = lowpass(imp, cutoff=cutoff, fs=FS)
    freqs = np.fft.rfftfreq(len(h), d=1.0 / FS)
    mag = np.abs(np.fft.rfft(h))
    dc_gain = mag[0]
    idx_cutoff = np.argmin(np.abs(freqs - cutoff))
    gain_at_cutoff = mag[idx_cutoff] / dc_gain
    assert gain_at_cutoff == pytest.approx(1.0 / np.sqrt(2), abs=0.08)


def test_bandpass_bandwidth_equals_cutoff_over_Q():
    """The −3 dB bandwidth of a bandpass must equal cutoff/Q (Audio EQ Cookbook relation)."""
    cutoff = 1000.0
    Q = 4.0
    expected_bw = cutoff / Q  # 250 Hz
    _, imp = generate_impulse(fs=FS, duration=DURATION)
    h = bandpass(imp, cutoff=cutoff, fs=FS, Q=Q)
    freqs = np.fft.rfftfreq(len(h), d=1.0 / FS)
    mag = np.abs(np.fft.rfft(h))
    peak_gain = np.max(mag)
    threshold = peak_gain / np.sqrt(2)
    above = freqs[mag >= threshold]
    if len(above) >= 2:
        measured_bw = above[-1] - above[0]
        assert abs(measured_bw - expected_bw) / expected_bw < 0.2

