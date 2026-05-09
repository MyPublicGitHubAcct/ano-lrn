import numpy as np
import pytest

from python.filters import moog_ladder

from tests.filters.conftest import FS, _dc, _sine, _steady_rms


def test_moog_ladder_output_shape():
    """Filter must return an array of the same length as the input."""
    signal = _sine(440.0)
    out = moog_ladder(signal, cutoff=1000.0, fs=FS)
    assert out.shape == signal.shape


def test_moog_ladder_passes_low_freq():
    """A decade below cutoff, the ladder must pass the signal with < 10% attenuation."""
    sig = _sine(100.0)
    out = moog_ladder(sig, cutoff=1000.0, fs=FS, resonance=0.0)
    assert _steady_rms(out) / _steady_rms(sig) > 0.9


def test_moog_ladder_attenuates_high_freq():
    """A decade above cutoff, a 4-pole ladder must attenuate by > 40 dB (< 1% amplitude)."""
    sig = _sine(10000.0)
    out = moog_ladder(sig, cutoff=1000.0, fs=FS, resonance=0.0)
    assert _steady_rms(out) / _steady_rms(sig) < 0.01


def test_moog_ladder_passes_dc():
    """H(0) = 1 for the ladder at zero resonance; DC must pass without attenuation."""
    out = moog_ladder(_dc(1.0), cutoff=1000.0, fs=FS, resonance=0.0)
    assert _steady_rms(out) == pytest.approx(1.0, abs=0.01)


def test_moog_ladder_resonance_boosts_at_cutoff():
    """Adding resonance feedback creates a peak at the cutoff; RMS at that frequency must be higher than without."""
    sig = _sine(1000.0)
    ratio_low = _steady_rms(moog_ladder(sig, cutoff=1000.0, fs=FS, resonance=0.0)) / _steady_rms(sig)
    ratio_high = _steady_rms(moog_ladder(sig, cutoff=1000.0, fs=FS, resonance=0.8)) / _steady_rms(sig)
    assert ratio_high > ratio_low


# --- parameter range tests ---

def test_moog_ladder_resonance_near_max_is_finite():
    """Resonance just below self-oscillation threshold must not produce NaN or Inf."""
    out = moog_ladder(_sine(440.0), cutoff=1000.0, fs=FS, resonance=0.99)
    assert np.all(np.isfinite(out))


def test_moog_ladder_cutoff_at_20hz_is_stable():
    """Cutoff at the lowest audible frequency (20 Hz) must not cause numerical instability."""
    out = moog_ladder(_sine(440.0), cutoff=20.0, fs=FS, resonance=0.0)
    assert np.all(np.isfinite(out))


def test_moog_ladder_cutoff_at_20khz_is_stable():
    """Cutoff near Nyquist (20 kHz at 44.1 kHz fs) must not cause numerical instability."""
    out = moog_ladder(_sine(440.0), cutoff=20000.0, fs=FS, resonance=0.0)
    assert np.all(np.isfinite(out))


def test_moog_ladder_cutoff_at_20khz_passes_low_freq():
    """When cutoff is above the signal, the ladder should act as an all-pass and pass the signal."""
    sig = _sine(440.0)
    out = moog_ladder(sig, cutoff=20000.0, fs=FS, resonance=0.0)
    assert _steady_rms(out) / _steady_rms(sig) > 0.9


def test_moog_ladder_cutoff_at_20hz_blocks_signal():
    """When cutoff is far below the signal, the ladder must heavily attenuate it."""
    sig = _sine(440.0)
    out = moog_ladder(sig, cutoff=20.0, fs=FS, resonance=0.0)
    assert _steady_rms(out) / _steady_rms(sig) < 0.1

