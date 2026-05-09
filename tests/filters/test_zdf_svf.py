import numpy as np
import pytest

from python.filters import zdf_svf
from python.generators import generate_impulse, generate_sine

from tests.filters.conftest import FS, DURATION, SKIP, _sine, _steady_rms

BUTTERWORTH_Q = 1.0 / np.sqrt(2)
CUTOFF = 1000.0


def _impulse(duration: float = DURATION) -> np.ndarray:
    _, imp = generate_impulse(fs=FS, duration=duration)
    return imp


def _mag_db_at(h: np.ndarray, freq: float) -> float:
    freqs = np.fft.rfftfreq(len(h), d=1.0 / FS)
    mag_db = 20.0 * np.log10(np.abs(np.fft.rfft(h)) + 1e-12)
    return float(mag_db[np.argmin(np.abs(freqs - freq))])


# --- shape and modes ---

def test_zdf_svf_lp_output_shape():
    """Default (LP) mode must return a 1-D array with the same length as the input."""
    sig = _sine(440.0)
    out = zdf_svf(sig, cutoff=CUTOFF, fs=FS)
    assert out.shape == sig.shape


def test_zdf_svf_all_returns_three_arrays():
    """mode='all' must return a 3-tuple of (LP, BP, HP) arrays, each matching the input shape."""
    sig = _sine(440.0)
    result = zdf_svf(sig, cutoff=CUTOFF, fs=FS, mode="all")
    assert isinstance(result, tuple) and len(result) == 3
    for arr in result:
        assert arr.shape == sig.shape


def test_zdf_svf_invalid_mode_raises():
    """An unrecognised mode string must raise ValueError rather than silently returning garbage."""
    with pytest.raises(ValueError):
        zdf_svf(_sine(440.0), cutoff=CUTOFF, fs=FS, mode="invalid")


# --- LP behaviour ---

def test_zdf_svf_lp_passes_low_freq():
    """A decade below cutoff, the LP output must pass with < 10% attenuation."""
    sig = _sine(100.0)
    out = zdf_svf(sig, cutoff=CUTOFF, fs=FS, resonance=BUTTERWORTH_Q, mode="lp")
    assert _steady_rms(out) / _steady_rms(sig) > 0.9


def test_zdf_svf_lp_attenuates_high_freq():
    """A decade above cutoff, the 2-pole LP must attenuate by > 26 dB (< 5% amplitude)."""
    sig = _sine(10000.0)
    out = zdf_svf(sig, cutoff=CUTOFF, fs=FS, resonance=BUTTERWORTH_Q, mode="lp")
    assert _steady_rms(out) / _steady_rms(sig) < 0.05


def test_zdf_svf_lp_rolloff_slope_12db_per_octave():
    """Impulse-response slope between 2× and 4× cutoff must be between −8 and −16 dB/octave (2-pole LP)."""
    lp = zdf_svf(_impulse(4.0), cutoff=500.0, fs=FS, resonance=BUTTERWORTH_Q, mode="lp")
    slope = _mag_db_at(lp, 2000.0) - _mag_db_at(lp, 1000.0)
    assert -16.0 < slope < -8.0


# --- LP −3 dB point accuracy at high fc/fs ratios (key ZDF advantage) ---

@pytest.mark.parametrize("fc_ratio", [0.01, 0.1, 0.4])
def test_zdf_svf_lp_minus3db_at_cutoff(fc_ratio):
    """Bilinear pre-warping in ZDF guarantees exactly −3 dB at cutoff for Butterworth Q, even at fc/fs = 0.4."""
    cutoff = fc_ratio * FS
    lp = zdf_svf(_impulse(4.0), cutoff=cutoff, fs=FS, resonance=BUTTERWORTH_Q, mode="lp")
    db_at_fc = _mag_db_at(lp, cutoff)
    assert -3.5 < db_at_fc < -2.5


# --- BP behaviour ---

def test_zdf_svf_bp_peaks_at_cutoff():
    """Impulse-response FFT peak must fall within 5% of the cutoff frequency in BP mode."""
    imp = _impulse(4.0)
    bp = zdf_svf(imp, cutoff=CUTOFF, fs=FS, resonance=2.0, mode="bp")
    freqs = np.fft.rfftfreq(len(bp), d=1.0 / FS)
    peak_freq = freqs[np.argmax(np.abs(np.fft.rfft(bp)))]
    assert abs(peak_freq - CUTOFF) / CUTOFF < 0.05


def test_zdf_svf_bp_attenuates_far_below_cutoff():
    """BP output at a frequency a decade below center must be lower than at center frequency."""
    out_low = zdf_svf(_sine(100.0), cutoff=CUTOFF, fs=FS, resonance=2.0, mode="bp")
    out_cut = zdf_svf(_sine(CUTOFF), cutoff=CUTOFF, fs=FS, resonance=2.0, mode="bp")
    assert _steady_rms(out_cut) > _steady_rms(out_low)


def test_zdf_svf_bp_attenuates_far_above_cutoff():
    """BP output at a frequency a decade above center must be lower than at center frequency."""
    out_high = zdf_svf(_sine(10000.0), cutoff=CUTOFF, fs=FS, resonance=2.0, mode="bp")
    out_cut = zdf_svf(_sine(CUTOFF), cutoff=CUTOFF, fs=FS, resonance=2.0, mode="bp")
    assert _steady_rms(out_cut) > _steady_rms(out_high)


# --- HP behaviour ---

def test_zdf_svf_hp_passes_high_freq():
    """A decade above cutoff, the HP output must pass with < 10% attenuation."""
    sig = _sine(10000.0)
    out = zdf_svf(sig, cutoff=CUTOFF, fs=FS, resonance=BUTTERWORTH_Q, mode="hp")
    assert _steady_rms(out) / _steady_rms(sig) > 0.9


def test_zdf_svf_hp_rejects_low_freq():
    # Butterworth HP2 at fc/10 attenuates to ≈ −40 dB ((f/fc)² = 0.01)
    sig = _sine(100.0)
    out = zdf_svf(sig, cutoff=CUTOFF, fs=FS, resonance=BUTTERWORTH_Q, mode="hp")
    assert _steady_rms(out) / _steady_rms(sig) < 0.015


def test_zdf_svf_hp_above_4x_cutoff_within_1db():
    """Four octaves above cutoff the HP must be within 1 dB of unity gain (flat passband)."""
    imp = _impulse(4.0)
    hp = zdf_svf(imp, cutoff=CUTOFF, fs=FS, resonance=BUTTERWORTH_Q, mode="hp")
    assert _mag_db_at(hp, 4.0 * CUTOFF) > -1.0


# --- Butterworth (maximally flat) ---

def test_zdf_svf_butterworth_passband_is_flat():
    # At fc/2, Butterworth LP2 should be < 0.3 dB down (ω⁴/(ω⁴+1) ≈ 0.9412 → −0.26 dB)
    imp = _impulse(4.0)
    lp = zdf_svf(imp, cutoff=CUTOFF, fs=FS, resonance=BUTTERWORTH_Q, mode="lp")
    assert _mag_db_at(lp, CUTOFF / 2.0) > -1.0


def test_zdf_svf_butterworth_no_resonant_peak():
    # At resonance = Q, LP gain should not exceed 0 dB in passband
    imp = _impulse(4.0)
    lp = zdf_svf(imp, cutoff=CUTOFF, fs=FS, resonance=BUTTERWORTH_Q, mode="lp")
    freqs = np.fft.rfftfreq(len(lp), d=1.0 / FS)
    mag_db = 20.0 * np.log10(np.abs(np.fft.rfft(lp)) + 1e-12)
    passband_mask = freqs < CUTOFF
    assert np.max(mag_db[passband_mask]) < 0.5


# --- notch ---

def test_zdf_svf_notch_rejects_at_cutoff():
    """Notch output at the center frequency must be lower than at a frequency a decade away."""
    out_cut = zdf_svf(_sine(CUTOFF), cutoff=CUTOFF, fs=FS, resonance=4.0, mode="notch")
    out_low = zdf_svf(_sine(100.0), cutoff=CUTOFF, fs=FS, resonance=4.0, mode="notch")
    assert _steady_rms(out_cut) < _steady_rms(out_low)


# --- KVL identity: LP + k·BP + HP = x ---

def test_zdf_svf_kvl_identity():
    """The TPT topology satisfies LP + (1/Q)·BP + HP = x exactly; any deviation indicates a state-variable bug."""
    sig = _sine(440.0)
    k = 1.0 / BUTTERWORTH_Q
    lp, bp, hp = zdf_svf(sig, cutoff=CUTOFF, fs=FS, resonance=BUTTERWORTH_Q, mode="all")
    np.testing.assert_allclose(lp + k * bp + hp, sig, atol=1e-10)


# --- parameter range tests ---

def test_zdf_svf_high_resonance_is_finite():
    """Very high resonance (Q=20, narrow resonant peak) must not cause NaN or Inf."""
    out = zdf_svf(_sine(440.0), cutoff=CUTOFF, fs=FS, resonance=20.0, mode="lp")
    assert np.all(np.isfinite(out))


def test_zdf_svf_cutoff_at_20hz_is_stable():
    """Cutoff at the lowest audible frequency (20 Hz) must not cause numerical instability."""
    out = zdf_svf(_sine(440.0), cutoff=20.0, fs=FS, resonance=BUTTERWORTH_Q, mode="lp")
    assert np.all(np.isfinite(out))


def test_zdf_svf_cutoff_near_nyquist_is_stable():
    # ZDF is stable up to fs/2 unlike the Chamberlin SVF
    out = zdf_svf(_sine(440.0), cutoff=FS * 0.49, fs=FS, resonance=BUTTERWORTH_Q, mode="lp")
    assert np.all(np.isfinite(out))


def test_zdf_svf_cutoff_at_20hz_blocks_signal():
    """With cutoff far below the signal, LP output must be strongly attenuated."""
    sig = _sine(440.0)
    out = zdf_svf(sig, cutoff=20.0, fs=FS, resonance=BUTTERWORTH_Q, mode="lp")
    assert _steady_rms(out) / _steady_rms(sig) < 0.1
