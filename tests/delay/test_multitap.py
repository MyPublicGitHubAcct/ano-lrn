import numpy as np
import pytest

from python.delay import multitap_delay

FS = 44100


def _impulse(n=16384):
    sig = np.zeros(n)
    sig[0] = 1.0
    return sig


# --- output shape ---

def test_shape():
    """multitap_delay must return (t, L, R) arrays each with the same length as the input."""
    sig = _impulse(n=4096)
    t, L, R = multitap_delay(sig, FS, [(0.01, 1.0, 0.0)])
    assert t.shape == sig.shape
    assert L.shape == sig.shape
    assert R.shape == sig.shape


def test_time_axis():
    """Time axis must start at 0 and have spacing 1/fs."""
    sig = _impulse(n=1024)
    t, _, _ = multitap_delay(sig, FS, [(0.01, 1.0, 0.0)])
    assert t[0] == pytest.approx(0.0)
    assert t[1] - t[0] == pytest.approx(1.0 / FS)


# --- empty tap list ---

def test_empty_taps_silence():
    """No taps must produce silence in both channels."""
    sig = _impulse()
    _, L, R = multitap_delay(sig, FS, [])
    np.testing.assert_allclose(L, np.zeros(len(sig)), atol=1e-12)
    np.testing.assert_allclose(R, np.zeros(len(sig)), atol=1e-12)


# --- zero delay reproduces dry signal ---

def test_zero_delay_center_pan_both_channels_equal():
    """A tap at delay=0, pan=0 must produce identical L and R arrays."""
    sig = np.random.default_rng(0).standard_normal(4096)
    _, L, R = multitap_delay(sig, FS, [(0.0, 1.0, 0.0)])
    np.testing.assert_allclose(L, R, atol=1e-12)


def test_zero_delay_center_pan_amplitude():
    """At delay=0, gain=1, pan=0 each channel carries signal × cos(π/4) (constant-power centre)."""
    sig = np.random.default_rng(1).standard_normal(4096)
    _, L, _ = multitap_delay(sig, FS, [(0.0, 1.0, 0.0)])
    np.testing.assert_allclose(L, np.cos(np.pi / 4) * sig, atol=1e-12)


# --- delay accuracy ---

def test_tap_energy_at_correct_integer_sample():
    """An integer-sample delay must place impulse energy at the expected sample index."""
    sig = _impulse()
    delay_time = 0.01  # 441 samples exactly at FS=44100
    D = round(delay_time * FS)
    _, L, _ = multitap_delay(sig, FS, [(delay_time, 1.0, 0.0)])
    assert L[D - 1] + L[D] + L[D + 1] > 0.5


def test_tap_fractional_delay_centroid():
    """With a non-integer delay, the energy centroid must land within 1 sample of the target."""
    sig = _impulse()
    delay_samples = 441.5
    _, _, R = multitap_delay(sig, FS, [(delay_samples / FS, 1.0, 1.0)])
    energy = R ** 2
    centroid = np.sum(np.arange(len(R)) * energy) / np.sum(energy)
    assert abs(centroid - delay_samples) < 1.0


# --- constant-power panning law ---

def test_pan_full_left_R_is_zero():
    """At pan=-1, all power must be in L; R must be identically zero."""
    sig = _impulse()
    _, L, R = multitap_delay(sig, FS, [(0.01, 1.0, -1.0)])
    np.testing.assert_allclose(R, np.zeros(len(sig)), atol=1e-12)
    assert np.max(np.abs(L)) > 0.5


def test_pan_full_right_L_is_zero():
    """At pan=+1, all power must be in R; L must be identically zero."""
    sig = _impulse()
    _, L, R = multitap_delay(sig, FS, [(0.01, 1.0, 1.0)])
    np.testing.assert_allclose(L, np.zeros(len(sig)), atol=1e-12)
    assert np.max(np.abs(R)) > 0.5


def test_pan_full_left_L_equals_dry():
    """At pan=-1, delay=0, gain=1: L must equal the dry signal (cos(0)=1)."""
    sig = np.random.default_rng(2).standard_normal(4096)
    _, L, R = multitap_delay(sig, FS, [(0.0, 1.0, -1.0)])
    np.testing.assert_allclose(L, sig, atol=1e-12)
    np.testing.assert_allclose(R, np.zeros(len(sig)), atol=1e-12)


def test_pan_full_right_R_equals_dry():
    """At pan=+1, delay=0, gain=1: R must equal the dry signal (sin(π/2)=1)."""
    sig = np.random.default_rng(3).standard_normal(4096)
    _, L, R = multitap_delay(sig, FS, [(0.0, 1.0, 1.0)])
    np.testing.assert_allclose(R, sig, atol=1e-12)
    np.testing.assert_allclose(L, np.zeros(len(sig)), atol=1e-12)


def test_constant_power_sums_to_unity():
    """L² + R² at the tap peak must equal gain² regardless of pan (constant-power invariant)."""
    sig = _impulse()
    D = round(0.01 * FS)  # integer delay → impulse response is exactly 1 at D
    for pan in (-1.0, -0.5, 0.0, 0.5, 1.0):
        _, L, R = multitap_delay(sig, FS, [(0.01, 1.0, pan)])
        power = L[D] ** 2 + R[D] ** 2
        assert power == pytest.approx(1.0, abs=1e-10), f"power={power} at pan={pan}"


# --- multiple taps ---

def test_multiple_taps_superpose():
    """Two taps at different delays must produce energy at both expected positions."""
    sig = _impulse()
    D1 = round(0.01 * FS)
    D2 = round(0.02 * FS)
    _, L, R = multitap_delay(sig, FS, [
        (0.01, 1.0, -1.0),  # first echo: full left
        (0.02, 1.0,  1.0),  # second echo: full right
    ])
    assert L[D1] > 0.5
    assert R[D2] > 0.5
    assert abs(R[D1]) < 1e-10
    assert abs(L[D2]) < 1e-10


def test_gain_scales_output():
    """Doubling the gain must double the output amplitude."""
    sig = _impulse()
    _, L1, _ = multitap_delay(sig, FS, [(0.01, 1.0, 0.0)])
    _, L2, _ = multitap_delay(sig, FS, [(0.01, 2.0, 0.0)])
    np.testing.assert_allclose(L2, 2.0 * L1, atol=1e-12)


# --- finite output ---

def test_output_finite():
    """All output samples must be finite for any valid tap configuration."""
    sig = np.random.default_rng(4).standard_normal(4096)
    taps = [(0.01, 0.8, -0.5), (0.02, 0.6, 0.0), (0.05, 0.4, 0.7)]
    _, L, R = multitap_delay(sig, FS, taps)
    assert np.all(np.isfinite(L))
    assert np.all(np.isfinite(R))
