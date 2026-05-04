import numpy as np
import pytest

from python.filters import allpass, bandpass, highpass, lowpass, notch
from python.generators import generate_impulse

from tests.filters.conftest import (
    FS, DURATION, _dc, _half_nyquist_sig, _nyquist_sig, _quarter_nyquist_sig,
    _sine, _steady_rms,
)


def test_any_filter_output_shape(any_filter, cutoff):
    signal = _sine(440.0)
    out = any_filter(signal, cutoff=cutoff, fs=FS)
    assert out.shape == signal.shape


def test_lowpass_passes_low_frequency():
    sig = _sine(100.0)
    out = lowpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_lowpass_attenuates_high_frequency():
    sig = _sine(10000.0)
    out = lowpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.05


def test_lowpass_passes_dc():
    out = lowpass(_dc(1.0), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) == pytest.approx(1.0, abs=0.01)


def test_lowpass_Q_sharpens_transition(Q_value):
    sig = _sine(10000.0)
    out = lowpass(sig, cutoff=1000.0, fs=FS, Q=Q_value)
    assert _steady_rms(out) / _steady_rms(sig) < 0.2


def test_highpass_passes_high_frequency():
    sig = _sine(10000.0)
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_highpass_attenuates_low_frequency():
    sig = _sine(100.0)
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.05


def test_highpass_rejects_dc():
    out = highpass(_dc(1.0), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) < 0.01


def test_bandpass_passes_center_frequency():
    sig = _sine(1000.0)
    out = bandpass(sig, cutoff=1000.0, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) > 0.85


def test_bandpass_attenuates_far_above():
    sig = _sine(10000.0)
    out = bandpass(sig, cutoff=1000.0, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) < 0.1


def test_bandpass_attenuates_far_below():
    sig = _sine(100.0)
    out = bandpass(sig, cutoff=1000.0, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) < 0.05


def test_bandpass_rejects_dc():
    out = bandpass(_dc(1.0), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) < 0.01


def test_bandpass_higher_Q_narrows_bandwidth():
    sig = _sine(1200.0)
    ratio_low_Q = _steady_rms(bandpass(sig, cutoff=1000.0, fs=FS, Q=1.0)) / _steady_rms(sig)
    ratio_high_Q = _steady_rms(bandpass(sig, cutoff=1000.0, fs=FS, Q=8.0)) / _steady_rms(sig)
    assert ratio_high_Q < ratio_low_Q


def test_notch_passes_below_cutoff():
    sig = _sine(100.0)
    out = notch(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_notch_passes_above_cutoff():
    sig = _sine(10000.0)
    out = notch(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_notch_attenuates_center():
    sig = _sine(1000.0)
    out = notch(sig, cutoff=1000.0, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) < 0.01


def test_notch_passes_dc():
    out = notch(_dc(1.0), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) == pytest.approx(1.0, abs=0.01)


def test_notch_higher_Q_narrows_notch():
    sig = _sine(1200.0)
    ratio_low_Q = _steady_rms(notch(sig, cutoff=1000.0, fs=FS, Q=1.0)) / _steady_rms(sig)
    ratio_high_Q = _steady_rms(notch(sig, cutoff=1000.0, fs=FS, Q=8.0)) / _steady_rms(sig)
    assert ratio_high_Q > ratio_low_Q


def test_allpass_preserves_rms_at_low_freq():
    sig = _sine(100.0)
    out = allpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(1.0, abs=0.02)


def test_allpass_preserves_rms_at_high_freq():
    sig = _sine(10000.0)
    out = allpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(1.0, abs=0.02)


def test_allpass_unity_magnitude():
    _, imp = generate_impulse(fs=FS, duration=DURATION)
    h = allpass(imp, cutoff=1000.0, fs=FS)
    mag = np.abs(np.fft.rfft(h))
    assert np.allclose(mag[1:], 1.0, atol=0.01)


def test_lowpass_rejects_nyquist():
    out = lowpass(_nyquist_sig(), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) < 1e-6


def test_highpass_passes_nyquist():
    sig = _nyquist_sig()
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.99


def test_bandpass_rejects_nyquist():
    out = bandpass(_nyquist_sig(), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) < 1e-6


def test_lowpass_attenuates_half_nyquist():
    sig = _half_nyquist_sig()
    out = lowpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.02


def test_highpass_passes_half_nyquist():
    sig = _half_nyquist_sig()
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_bandpass_passes_half_nyquist():
    sig = _half_nyquist_sig()
    out = bandpass(sig, cutoff=FS / 4, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) > 0.85


def test_lowpass_attenuates_quarter_nyquist():
    sig = _quarter_nyquist_sig()
    out = lowpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.1


def test_highpass_passes_quarter_nyquist():
    sig = _quarter_nyquist_sig()
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_bandpass_passes_quarter_nyquist():
    sig = _quarter_nyquist_sig()
    out = bandpass(sig, cutoff=FS / 8, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) > 0.85


# --- parameter range tests ---

def test_eq_cutoff_at_20hz_is_stable():
    sig = _sine(440.0)
    for filt in [lowpass, highpass, bandpass, notch, allpass]:
        out = filt(sig, cutoff=20.0, fs=FS)
        assert np.all(np.isfinite(out))


def test_eq_cutoff_near_nyquist_is_stable():
    sig = _sine(440.0)
    for filt in [lowpass, highpass, bandpass, notch, allpass]:
        out = filt(sig, cutoff=FS * 0.49, fs=FS)
        assert np.all(np.isfinite(out))


def test_lowpass_low_Q_is_stable():
    out = lowpass(_sine(440.0), cutoff=1000.0, fs=FS, Q=0.1)
    assert np.all(np.isfinite(out))


def test_lowpass_high_Q_resonates_at_cutoff():
    sig = _sine(1000.0)
    out_low_Q = lowpass(sig, cutoff=1000.0, fs=FS, Q=0.707)
    out_high_Q = lowpass(sig, cutoff=1000.0, fs=FS, Q=10.0)
    assert _steady_rms(out_high_Q) > _steady_rms(out_low_Q)
