import numpy as np
import pytest

from python.filters import highshelf, lowshelf

from tests.filters.conftest import FS, _dc, _nyquist_sig, _sine, _steady_rms


def test_lowshelf_boosts_dc():
    """H(0) for a boosting low shelf must equal 10^(gain_db/20); verified via DC input RMS."""
    expected = 10 ** (6.0 / 20)
    out = lowshelf(_dc(1.0), cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) == pytest.approx(expected, abs=0.05)


def test_lowshelf_cuts_dc():
    """H(0) for a cutting low shelf must equal 10^(gain_db/20) with gain_db negative."""
    expected = 10 ** (-6.0 / 20)
    out = lowshelf(_dc(1.0), cutoff=1000.0, fs=FS, gain_db=-6.0)
    assert _steady_rms(out) == pytest.approx(expected, abs=0.05)


def test_lowshelf_unity_at_nyquist():
    """H(π) = 1 for a low shelf; the high-frequency shelf region must pass Nyquist unchanged."""
    sig = _nyquist_sig()
    out = lowshelf(sig, cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(1.0, abs=0.01)


def test_lowshelf_boosts_low_freq():
    """A frequency well below cutoff must be boosted by substantially more than unity (> 1.5×)."""
    sig = _sine(100.0)
    out = lowshelf(sig, cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) / _steady_rms(sig) > 1.5


def test_highshelf_unity_at_dc():
    """H(0) = 1 for a high shelf; DC must pass unchanged regardless of gain_db."""
    out = highshelf(_dc(1.0), cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) == pytest.approx(1.0, abs=0.01)


def test_highshelf_boosts_nyquist():
    """H(π) for a boosting high shelf must equal 10^(gain_db/20); verified via Nyquist RMS."""
    expected = 10 ** (6.0 / 20)
    sig = _nyquist_sig()
    out = highshelf(sig, cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(expected, abs=0.05)


def test_highshelf_cuts_nyquist():
    """H(π) for a cutting high shelf must equal 10^(gain_db/20) with gain_db negative."""
    expected = 10 ** (-6.0 / 20)
    sig = _nyquist_sig()
    out = highshelf(sig, cutoff=1000.0, fs=FS, gain_db=-6.0)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(expected, abs=0.05)


def test_highshelf_boosts_high_freq():
    """A frequency well above cutoff must be boosted by substantially more than unity (> 1.5×)."""
    sig = _sine(10000.0)
    out = highshelf(sig, cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) / _steady_rms(sig) > 1.5


# --- parameter range tests ---

def test_lowshelf_zero_gain_db_is_identity():
    """gain_db=0 must produce an all-pass response; output must equal input to floating-point precision."""
    sig = _sine(100.0)
    out = lowshelf(sig, cutoff=1000.0, fs=FS, gain_db=0.0)
    np.testing.assert_allclose(out, sig, atol=1e-10)


def test_highshelf_zero_gain_db_is_identity():
    """gain_db=0 must produce an all-pass response; output must equal input to floating-point precision."""
    sig = _sine(10000.0)
    out = highshelf(sig, cutoff=1000.0, fs=FS, gain_db=0.0)
    np.testing.assert_allclose(out, sig, atol=1e-10)


def test_lowshelf_large_boost_is_stable():
    """An 18 dB boost (large gain) must not cause NaN or Inf in the output."""
    out = lowshelf(_sine(100.0), cutoff=1000.0, fs=FS, gain_db=18.0)
    assert np.all(np.isfinite(out))


def test_highshelf_large_cut_is_stable():
    """An 18 dB cut (large negative gain) must not cause NaN or Inf in the output."""
    out = highshelf(_sine(10000.0), cutoff=1000.0, fs=FS, gain_db=-18.0)
    assert np.all(np.isfinite(out))


def test_lowshelf_large_boost_raises_level():
    """DC gain at +18 dB must equal 10^(18/20) ≈ 7.94×; confirms large-gain coefficient accuracy."""
    expected = 10 ** (18.0 / 20)
    out = lowshelf(_dc(1.0), cutoff=1000.0, fs=FS, gain_db=18.0)
    assert _steady_rms(out) == pytest.approx(expected, rel=0.01)


def test_highshelf_large_cut_lowers_level():
    """Nyquist gain at −18 dB must equal 10^(−18/20) ≈ 0.126×; confirms large-cut coefficient accuracy."""
    expected = 10 ** (-18.0 / 20)
    sig = _nyquist_sig()
    out = highshelf(sig, cutoff=1000.0, fs=FS, gain_db=-18.0)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(expected, rel=0.01)

