import numpy as np
import pytest

from python.filters import moog_ladder

from tests.filters.conftest import FS, _dc, _sine, _steady_rms


def test_moog_ladder_output_shape():
    signal = _sine(440.0)
    out = moog_ladder(signal, cutoff=1000.0, fs=FS)
    assert out.shape == signal.shape


def test_moog_ladder_passes_low_freq():
    sig = _sine(100.0)
    out = moog_ladder(sig, cutoff=1000.0, fs=FS, resonance=0.0)
    assert _steady_rms(out) / _steady_rms(sig) > 0.9


def test_moog_ladder_attenuates_high_freq():
    sig = _sine(10000.0)
    out = moog_ladder(sig, cutoff=1000.0, fs=FS, resonance=0.0)
    assert _steady_rms(out) / _steady_rms(sig) < 0.01


def test_moog_ladder_passes_dc():
    out = moog_ladder(_dc(1.0), cutoff=1000.0, fs=FS, resonance=0.0)
    assert _steady_rms(out) == pytest.approx(1.0, abs=0.01)


def test_moog_ladder_resonance_boosts_at_cutoff():
    sig = _sine(1000.0)
    ratio_low = _steady_rms(moog_ladder(sig, cutoff=1000.0, fs=FS, resonance=0.0)) / _steady_rms(sig)
    ratio_high = _steady_rms(moog_ladder(sig, cutoff=1000.0, fs=FS, resonance=0.8)) / _steady_rms(sig)
    assert ratio_high > ratio_low
