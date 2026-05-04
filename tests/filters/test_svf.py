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
    sig = _sine(440.0)
    out = svf(sig, cutoff=CUTOFF, fs=FS)
    assert out.shape == sig.shape


def test_svf_all_returns_three_arrays():
    sig = _sine(440.0)
    result = svf(sig, cutoff=CUTOFF, fs=FS, mode="all")
    assert isinstance(result, tuple) and len(result) == 3
    for arr in result:
        assert arr.shape == sig.shape


def test_svf_invalid_mode_raises():
    with pytest.raises(ValueError):
        svf(_sine(440.0), cutoff=CUTOFF, fs=FS, mode="invalid")


# --- LP behaviour ---

def test_svf_lp_passes_low_freq():
    sig = _sine(100.0)
    out = svf(sig, cutoff=CUTOFF, fs=FS, resonance=0.0, mode="lp")
    assert _steady_rms(out) / _steady_rms(sig) > 0.9


def test_svf_lp_attenuates_high_freq():
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
    sig_low = _sine(100.0)
    sig_cut = _sine(CUTOFF)
    out_low = svf(sig_low, cutoff=CUTOFF, fs=FS, resonance=0.5, mode="bp")
    out_cut = svf(sig_cut, cutoff=CUTOFF, fs=FS, resonance=0.5, mode="bp")
    assert _steady_rms(out_cut) > _steady_rms(out_low)


def test_svf_bp_attenuates_far_above_cutoff():
    sig_high = _sine(10000.0)
    sig_cut = _sine(CUTOFF)
    out_high = svf(sig_high, cutoff=CUTOFF, fs=FS, resonance=0.5, mode="bp")
    out_cut = svf(sig_cut, cutoff=CUTOFF, fs=FS, resonance=0.5, mode="bp")
    assert _steady_rms(out_cut) > _steady_rms(out_high)


# --- HP behaviour ---

def test_svf_hp_passes_high_freq():
    sig = _sine(10000.0)
    out = svf(sig, cutoff=CUTOFF, fs=FS, resonance=0.0, mode="hp")
    assert _steady_rms(out) / _steady_rms(sig) > 0.9


def test_svf_hp_rejects_low_freq():
    sig = _sine(100.0)
    out = svf(sig, cutoff=CUTOFF, fs=FS, resonance=0.0, mode="hp")
    assert _steady_rms(out) / _steady_rms(sig) < 0.05


# --- notch behaviour ---

def test_svf_notch_rejects_at_cutoff():
    sig_cut = _sine(CUTOFF)
    sig_low = _sine(100.0)
    out_cut = svf(sig_cut, cutoff=CUTOFF, fs=FS, resonance=0.7, mode="notch")
    out_low = svf(sig_low, cutoff=CUTOFF, fs=FS, resonance=0.7, mode="notch")
    assert _steady_rms(out_cut) < _steady_rms(out_low)


# --- resonance behaviour ---

def test_svf_resonance_boosts_at_cutoff():
    sig = _sine(CUTOFF)
    out_low_res = svf(sig, cutoff=CUTOFF, fs=FS, resonance=0.0, mode="lp")
    out_high_res = svf(sig, cutoff=CUTOFF, fs=FS, resonance=0.8, mode="lp")
    assert _steady_rms(out_high_res) > _steady_rms(out_low_res)


# --- all outputs consistent ---

def test_svf_all_modes_have_same_shape():
    sig = _sine(440.0)
    lp, bp, hp = svf(sig, cutoff=CUTOFF, fs=FS, mode="all")
    assert lp.shape == bp.shape == hp.shape == sig.shape


# --- parameter range tests ---

def test_svf_resonance_near_one_is_finite():
    out = svf(_sine(440.0), cutoff=CUTOFF, fs=FS, resonance=0.99, mode="lp")
    assert np.all(np.isfinite(out))


def test_svf_cutoff_at_20hz_is_stable():
    out = svf(_sine(440.0), cutoff=20.0, fs=FS, resonance=0.0, mode="lp")
    assert np.all(np.isfinite(out))


def test_svf_cutoff_at_fs_over_5_is_stable():
    # Chamberlin SVF is stable below fs/4; fs/5 is safely within that limit
    out = svf(_sine(440.0), cutoff=FS / 5.0, fs=FS, resonance=0.0, mode="lp")
    assert np.all(np.isfinite(out))


def test_svf_cutoff_at_20hz_blocks_signal():
    sig = _sine(440.0)
    out = svf(sig, cutoff=20.0, fs=FS, resonance=0.0, mode="lp")
    assert _steady_rms(out) / _steady_rms(sig) < 0.1
