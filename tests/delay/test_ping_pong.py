import numpy as np
import pytest

from python.delay import ping_pong_delay

FS = 44100


def _impulse(n=16384):
    sig = np.zeros(n)
    sig[0] = 1.0
    return sig


# --- output shape ---

def test_shape():
    """ping_pong_delay must return (t, L, R) arrays each with the same length as the input."""
    sig = _impulse(n=4096)
    t, L, R = ping_pong_delay(sig, FS, delay_time=0.01)
    assert t.shape == sig.shape
    assert L.shape == sig.shape
    assert R.shape == sig.shape


def test_time_axis():
    """Time axis must start at 0 and have spacing 1/fs."""
    sig = _impulse(n=1024)
    t, L, R = ping_pong_delay(sig, FS, delay_time=0.005)
    assert t[0] == pytest.approx(0.0)
    assert t[1] - t[0] == pytest.approx(1.0 / FS)


# --- mix = 0 returns dry ---

def test_mix_zero_left_is_dry():
    """mix=0 means no wet signal is added; left output must equal the input exactly."""
    sig = np.random.default_rng(0).standard_normal(4096)
    _, L, _ = ping_pong_delay(sig, FS, delay_time=0.01, feedback=0.5, mix=0.0)
    np.testing.assert_allclose(L, sig, atol=1e-12)


def test_mix_zero_right_is_dry():
    """mix=0 means no wet signal is added; right output must equal the input exactly."""
    sig = np.random.default_rng(0).standard_normal(4096)
    _, _, R = ping_pong_delay(sig, FS, delay_time=0.01, feedback=0.5, mix=0.0)
    np.testing.assert_allclose(R, sig, atol=1e-12)


# --- feedback = 0: single echo in R, L is silent ---

def test_feedback_zero_right_has_one_echo():
    """At feedback=0, only the first bounce (in R) survives; R must have energy near delay_time."""
    sig = _impulse()
    delay_time = 0.01  # 441 samples
    _, _, R = ping_pong_delay(sig, FS, delay_time, feedback=0.0, mix=1.0)
    D = round(delay_time * FS)  # 441
    # Energy is concentrated around sample D (±2 for Lagrange interpolation spread)
    assert R[D - 1] + R[D] + R[D + 1] > 0.8


def test_feedback_zero_left_is_zero():
    """At feedback=0, no cross-feed reaches L; the L wet signal must be identically zero."""
    sig = _impulse()
    _, L, _ = ping_pong_delay(sig, FS, delay_time=0.01, feedback=0.0, mix=1.0)
    np.testing.assert_allclose(L, np.zeros(len(sig)), atol=1e-6)


def test_feedback_zero_fractional_sample_index():
    """The first echo must land at the correct sub-sample position, not just at floor(D)."""
    sig = _impulse()
    delay_samples = 441.5  # half-sample fractional delay
    _, _, R = ping_pong_delay(sig, FS, delay_samples / FS, feedback=0.0, mix=1.0)
    # Lagrange interpolation spreads energy across adjacent samples; both must carry weight
    assert R[441] > 0.05
    assert R[442] > 0.05
    # Energy centroid must land within 1 sample of the target
    energy = R ** 2
    centroid = np.sum(np.arange(len(R)) * energy) / np.sum(energy)
    assert abs(centroid - delay_samples) < 1.0


# --- feedback = 0.5: second echo appears in L at 2 * delay_time ---

def test_feedback_half_right_echo_at_D():
    """At feedback=0.5, the first echo (in R) must appear at delay_time with amplitude ≈ 1."""
    sig = _impulse()
    delay_samples = 500  # integer → exact placement
    _, _, R = ping_pong_delay(sig, FS, delay_samples / FS, feedback=0.5, mix=1.0)
    assert R[500] == pytest.approx(1.0, abs=1e-6)


def test_feedback_half_left_echo_at_2D():
    """At feedback=0.5, a second echo must appear in the opposite channel (L) at 2*delay_time."""
    sig = _impulse()
    delay_samples = 500
    _, L, _ = ping_pong_delay(sig, FS, delay_samples / FS, feedback=0.5, mix=1.0)
    assert L[1000] == pytest.approx(0.5, abs=1e-6)


def test_feedback_half_right_no_echo_at_2D():
    """At feedback=0.5, sample 2D belongs to L; R must have negligible energy there."""
    sig = _impulse()
    delay_samples = 500
    _, _, R = ping_pong_delay(sig, FS, delay_samples / FS, feedback=0.5, mix=1.0)
    assert abs(R[1000]) < 1e-6


def test_feedback_half_third_echo_in_right():
    """At feedback=0.5, R must also contain a third echo at 3*delay_time with amplitude feedback**2."""
    sig = _impulse()
    delay_samples = 500
    _, _, R = ping_pong_delay(sig, FS, delay_samples / FS, feedback=0.5, mix=1.0)
    assert R[1500] == pytest.approx(0.25, abs=1e-6)


# --- output is finite for typical feedback values ---

def test_output_finite():
    """All output samples must be finite for feedback < 1."""
    sig = np.random.default_rng(1).standard_normal(4096)
    _, L, R = ping_pong_delay(sig, FS, delay_time=0.01, feedback=0.8, mix=0.5)
    assert np.all(np.isfinite(L))
    assert np.all(np.isfinite(R))
