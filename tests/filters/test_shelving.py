import numpy as np
import pytest

from python.filters import highshelf, lowshelf

from tests.filters.conftest import FS, _dc, _nyquist_sig, _sine, _steady_rms


def test_lowshelf_boosts_dc():
    expected = 10 ** (6.0 / 20)
    out = lowshelf(_dc(1.0), cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) == pytest.approx(expected, abs=0.05)


def test_lowshelf_cuts_dc():
    expected = 10 ** (-6.0 / 20)
    out = lowshelf(_dc(1.0), cutoff=1000.0, fs=FS, gain_db=-6.0)
    assert _steady_rms(out) == pytest.approx(expected, abs=0.05)


def test_lowshelf_unity_at_nyquist():
    sig = _nyquist_sig()
    out = lowshelf(sig, cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(1.0, abs=0.01)


def test_lowshelf_boosts_low_freq():
    sig = _sine(100.0)
    out = lowshelf(sig, cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) / _steady_rms(sig) > 1.5


def test_highshelf_unity_at_dc():
    out = highshelf(_dc(1.0), cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) == pytest.approx(1.0, abs=0.01)


def test_highshelf_boosts_nyquist():
    expected = 10 ** (6.0 / 20)
    sig = _nyquist_sig()
    out = highshelf(sig, cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(expected, abs=0.05)


def test_highshelf_cuts_nyquist():
    expected = 10 ** (-6.0 / 20)
    sig = _nyquist_sig()
    out = highshelf(sig, cutoff=1000.0, fs=FS, gain_db=-6.0)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(expected, abs=0.05)


def test_highshelf_boosts_high_freq():
    sig = _sine(10000.0)
    out = highshelf(sig, cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) / _steady_rms(sig) > 1.5


# --- parameter range tests ---

def test_lowshelf_zero_gain_db_is_identity():
    sig = _sine(100.0)
    out = lowshelf(sig, cutoff=1000.0, fs=FS, gain_db=0.0)
    np.testing.assert_allclose(out, sig, atol=1e-10)


def test_highshelf_zero_gain_db_is_identity():
    sig = _sine(10000.0)
    out = highshelf(sig, cutoff=1000.0, fs=FS, gain_db=0.0)
    np.testing.assert_allclose(out, sig, atol=1e-10)


def test_lowshelf_large_boost_is_stable():
    out = lowshelf(_sine(100.0), cutoff=1000.0, fs=FS, gain_db=18.0)
    assert np.all(np.isfinite(out))


def test_highshelf_large_cut_is_stable():
    out = highshelf(_sine(10000.0), cutoff=1000.0, fs=FS, gain_db=-18.0)
    assert np.all(np.isfinite(out))


def test_lowshelf_large_boost_raises_level():
    expected = 10 ** (18.0 / 20)
    out = lowshelf(_dc(1.0), cutoff=1000.0, fs=FS, gain_db=18.0)
    assert _steady_rms(out) == pytest.approx(expected, rel=0.01)


def test_highshelf_large_cut_lowers_level():
    expected = 10 ** (-18.0 / 20)
    sig = _nyquist_sig()
    out = highshelf(sig, cutoff=1000.0, fs=FS, gain_db=-18.0)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(expected, rel=0.01)
