import numpy as np
import pytest

from python.filters import svf
from python.generators import generate_impulse, generate_sine

from tests.filters.conftest import FS, DURATION, SKIP, _sine, _steady_rms

CUTOFF = 1000.0


def _impulse() -> np.ndarray:
    _, imp = generate_impulse(fs=FS, duration=DURATION)
    return imp


def _mag_db_at(h: np.ndarray, freq: float) -> float:
    spectrum = np.abs(np.fft.rfft(h))
    freqs = np.fft.rfftfreq(len(h), d=1.0 / FS)
    idx = np.argmin(np.abs(freqs - freq))
    return 20.0 * np.log10(spectrum[idx] + 1e-12)


# --- shape and modes ---

def test_svf_lp_output_shape():
    """Default (LP) mode must return a 1-D array with the same length as the input."""
    sig = _sine(440.0)
    out = svf(sig, cutoff=CUTOFF, fs=FS)
    assert out.shape == sig.shape


def test_svf_all_returns_three_arrays():
    """mode='all' must return a 3-tuple of (LP, BP, HP) arrays, each matching the input shape."""
    sig = _sine(440.0)
    result = svf(sig, cutoff=CUTOFF, fs=FS, mode="all")
    assert isinstance(result, tuple) and len(result) == 3
    for arr in result:
        assert arr.shape == sig.shape


def test_svf_invalid_mode_raises():
    """An unrecognised mode string must raise ValueError rather than silently returning garbage."""
    with pytest.raises(ValueError):
        svf(_sine(440.0), cutoff=CUTOFF, fs=FS, mode="invalid")


# --- LP behaviour ---

def test_svf_lp_passes_low_freq():
    """A decade below cutoff, the LP output must pass with < 10% attenuation."""
    sig = _sine(100.0)
    out = svf(sig, cutoff=CUTOFF, fs=FS, resonance=0.0, mode="lp")
    assert _steady_rms(out) / _steady_rms(sig) > 0.9


def test_svf_lp_attenuates_high_freq():
    """A decade above cutoff, the 2-pole LP must attenuate by > 26 dB (< 5% amplitude)."""
    sig = _sine(10000.0)
    out = svf(sig, cutoff=CUTOFF, fs=FS, resonance=0.0, mode="lp")
    assert _steady_rms(out) / _steady_rms(sig) < 0.05


def test_svf_lp_rolloff_slope_12db_per_octave():
    # Slope between 2×cutoff and 4×cutoff should be ≈ −12 dB/octave (2-pole LP)
    imp = _impulse()
    lp = svf(imp, cutoff=500.0, fs=FS, resonance=0.0, mode="lp")
    db_at_2x = _mag_db_at(lp, 1000.0)
    db_at_4x = _mag_db_at(lp, 2000.0)
    slope = db_at_4x - db_at_2x
    assert -16.0 < slope < -8.0


# --- BP behaviour ---

def test_svf_bp_peaks_at_cutoff():
    # BP impulse response FFT magnitude should be highest near cutoff
    imp = _impulse()
    bp = svf(imp, cutoff=CUTOFF, fs=FS, resonance=0.5, mode="bp")
    freqs = np.fft.rfftfreq(len(bp), d=1.0 / FS)
    mag = np.abs(np.fft.rfft(bp))
    peak_freq = freqs[np.argmax(mag)]
    assert abs(peak_freq - CUTOFF) / CUTOFF < 0.1


def test_svf_bp_attenuates_far_below_cutoff():
    """BP output at a frequency a decade below center must be lower than at center frequency."""
    sig_low = _sine(100.0)
    sig_cut = _sine(CUTOFF)
    out_low = svf(sig_low, cutoff=CUTOFF, fs=FS, resonance=0.5, mode="bp")
    out_cut = svf(sig_cut, cutoff=CUTOFF, fs=FS, resonance=0.5, mode="bp")
    assert _steady_rms(out_cut) > _steady_rms(out_low)


def test_svf_bp_attenuates_far_above_cutoff():
    """BP output at a frequency a decade above center must be lower than at center frequency."""
    sig_high = _sine(10000.0)
    sig_cut = _sine(CUTOFF)
    out_high = svf(sig_high, cutoff=CUTOFF, fs=FS, resonance=0.5, mode="bp")
    out_cut = svf(sig_cut, cutoff=CUTOFF, fs=FS, resonance=0.5, mode="bp")
    assert _steady_rms(out_cut) > _steady_rms(out_high)


# --- HP behaviour ---

def test_svf_hp_passes_high_freq():
    """A decade above cutoff, the HP output must pass with < 10% attenuation."""
    sig = _sine(10000.0)
    out = svf(sig, cutoff=CUTOFF, fs=FS, resonance=0.0, mode="hp")
    assert _steady_rms(out) / _steady_rms(sig) > 0.9


def test_svf_hp_rejects_low_freq():
    """A decade below cutoff, the 2-pole HP must attenuate by > 26 dB (< 5% amplitude)."""
    sig = _sine(100.0)
    out = svf(sig, cutoff=CUTOFF, fs=FS, resonance=0.0, mode="hp")
    assert _steady_rms(out) / _steady_rms(sig) < 0.05


# --- notch behaviour ---

def test_svf_notch_rejects_at_cutoff():
    """Notch output at the center frequency must be lower than at a frequency a decade away."""
    sig_cut = _sine(CUTOFF)
    sig_low = _sine(100.0)
    out_cut = svf(sig_cut, cutoff=CUTOFF, fs=FS, resonance=0.7, mode="notch")
    out_low = svf(sig_low, cutoff=CUTOFF, fs=FS, resonance=0.7, mode="notch")
    assert _steady_rms(out_cut) < _steady_rms(out_low)


# --- resonance behaviour ---

def test_svf_resonance_boosts_at_cutoff():
    """Higher resonance must produce a larger peak at the cutoff frequency in LP mode."""
    sig = _sine(CUTOFF)
    out_low_res = svf(sig, cutoff=CUTOFF, fs=FS, resonance=0.0, mode="lp")
    out_high_res = svf(sig, cutoff=CUTOFF, fs=FS, resonance=0.8, mode="lp")
    assert _steady_rms(out_high_res) > _steady_rms(out_low_res)


# --- all outputs consistent ---

def test_svf_all_modes_have_same_shape():
    """All three outputs from mode='all' must match each other and the input in shape."""
    sig = _sine(440.0)
    lp, bp, hp = svf(sig, cutoff=CUTOFF, fs=FS, mode="all")
    assert lp.shape == bp.shape == hp.shape == sig.shape


# --- parameter range tests ---

def test_svf_resonance_near_one_is_finite():
    """Resonance near 1.0 approaches self-oscillation; the output must stay finite."""
    out = svf(_sine(440.0), cutoff=CUTOFF, fs=FS, resonance=0.99, mode="lp")
    assert np.all(np.isfinite(out))


def test_svf_cutoff_at_20hz_is_stable():
    """Cutoff at the lowest audible frequency (20 Hz) must not cause numerical instability."""
    out = svf(_sine(440.0), cutoff=20.0, fs=FS, resonance=0.0, mode="lp")
    assert np.all(np.isfinite(out))


def test_svf_cutoff_at_fs_over_5_is_stable():
    # Chamberlin SVF is stable below fs/4; fs/5 is safely within that limit
    out = svf(_sine(440.0), cutoff=FS / 5.0, fs=FS, resonance=0.0, mode="lp")
    assert np.all(np.isfinite(out))


def test_svf_cutoff_at_20hz_blocks_signal():
    """With cutoff far below the signal, LP output must be strongly attenuated."""
    sig = _sine(440.0)
    out = svf(sig, cutoff=20.0, fs=FS, resonance=0.0, mode="lp")
    assert _steady_rms(out) / _steady_rms(sig) < 0.1


def test_svf_lp_minus3db_within_10pct_of_cutoff():
    """Chamberlin LP −3 dB point must be within 10% of the cutoff frequency."""
    imp = _impulse()
    lp = svf(imp, cutoff=CUTOFF, fs=FS, resonance=0.0, mode="lp")
    freqs = np.fft.rfftfreq(len(lp), d=1.0 / FS)
    mag = np.abs(np.fft.rfft(lp))
    dc_gain = mag[0]
    threshold = dc_gain / np.sqrt(2)
    # Find the first frequency where magnitude drops below -3 dB
    above = np.where(mag >= threshold)[0]
    if len(above) > 0:
        f_3db = float(freqs[above[-1]])
        assert abs(f_3db - CUTOFF) / CUTOFF < 0.10


def test_svf_lp_bp_hp_sum_equals_input():
    """Chamberlin identity: lp[i] + q*bp[i-1] + hp[i] == x[i] (bp uses the previous sample's value)."""
    resonance = 0.5
    q = np.sqrt(2.0) * (1.0 - resonance)
    sig = _sine(440.0)
    lp, bp, hp = svf(sig, cutoff=CUTOFF, fs=FS, resonance=resonance, mode="all")
    # The Chamberlin loop uses the OLD bp when computing hp; bp_buf[i-1] is the old value at step i
    bp_old = np.concatenate([[0.0], bp[:-1]])
    reconstructed = lp + q * bp_old + hp
    np.testing.assert_allclose(reconstructed, sig, atol=1e-10)

