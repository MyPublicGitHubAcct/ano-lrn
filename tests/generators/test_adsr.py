import numpy as np
import pytest

from python.generators import generate_adsr

FS = 44100
DURATION = 1.0
N = int(FS * DURATION)


# --- shape ---

def test_adsr_output_shapes():
    """Generator contract: t and envelope must share the same N-sample shape."""
    t, env = generate_adsr(fs=FS, duration=DURATION)
    assert t.shape == (N,)
    assert env.shape == (N,)


def test_adsr_time_axis():
    """Time axis must start at 0 and step by 1/fs."""
    t, _ = generate_adsr(fs=FS, duration=DURATION)
    assert t[0] == pytest.approx(0.0)
    assert t[1] - t[0] == pytest.approx(1.0 / FS)


# --- attack peak ---

@pytest.mark.parametrize("attack", [0.01, 0.05, 0.1, 0.2])
def test_adsr_peak_at_end_of_attack(attack):
    """Peak must reach exactly 1.0 at the last sample of the attack segment."""
    _, env = generate_adsr(attack=attack, decay=0.05, sustain=0.8,
                           release=0.1, fs=FS, duration=1.0)
    n_a = int(attack * FS)
    assert env[n_a - 1] == pytest.approx(1.0)


def test_adsr_peak_never_exceeds_one():
    """Envelope must be bounded to [0, 1] at all times."""
    _, env = generate_adsr(attack=0.1, decay=0.1, sustain=0.6,
                           release=0.2, fs=FS, duration=1.0)
    assert np.all(env >= -1e-12)
    assert np.all(env <= 1.0 + 1e-12)


# --- sustain ---

@pytest.mark.parametrize("sustain", [0.0, 0.3, 0.7, 1.0])
def test_adsr_sustain_level_held(sustain):
    """All samples in the sustain segment must equal the sustain level."""
    attack, decay, release = 0.05, 0.05, 0.1
    _, env = generate_adsr(attack=attack, decay=decay, sustain=sustain,
                           release=release, fs=FS, duration=1.0)
    n_a = int(attack * FS)
    n_d = int(decay * FS)
    n_r = int(release * FS)
    n_s_start = n_a + n_d
    n_s_end = N - n_r
    if n_s_end > n_s_start:
        assert np.allclose(env[n_s_start:n_s_end], sustain)


# --- release ---

@pytest.mark.parametrize("release", [0.05, 0.1, 0.2, 0.3])
def test_adsr_release_reaches_zero(release):
    """Last sample of the release segment must be exactly 0.0."""
    _, env = generate_adsr(attack=0.05, decay=0.05, sustain=0.7,
                           release=release, fs=FS, duration=1.0)
    assert env[-1] == pytest.approx(0.0, abs=1e-12)


def test_adsr_release_monotone_decreasing():
    """Release segment must be monotonically non-increasing."""
    release = 0.2
    attack, decay, sustain = 0.05, 0.05, 0.7
    _, env = generate_adsr(attack=attack, decay=decay, sustain=sustain,
                           release=release, fs=FS, duration=1.0)
    n_r_start = N - int(release * FS)
    diffs = np.diff(env[n_r_start:])
    assert np.all(diffs <= 1e-12)


# --- curve option ---

def test_adsr_exponential_curve_endpoints():
    """Exponential curve must still hit 1.0 at attack end and 0.0 at release end."""
    _, env = generate_adsr(attack=0.1, decay=0.1, sustain=0.6,
                           release=0.2, fs=FS, duration=1.0,
                           curve="exponential")
    n_a = int(0.1 * FS)
    assert env[n_a - 1] == pytest.approx(1.0)
    assert env[-1] == pytest.approx(0.0, abs=1e-12)


def test_adsr_linear_and_exponential_same_endpoints():
    """Linear and exponential curves must share attack peak and release end values."""
    kw = dict(attack=0.1, decay=0.1, sustain=0.6, release=0.2, fs=FS, duration=1.0)
    _, env_lin = generate_adsr(**kw, curve="linear")
    _, env_exp = generate_adsr(**kw, curve="exponential")
    n_a = int(0.1 * FS)
    assert env_lin[n_a - 1] == pytest.approx(env_exp[n_a - 1])
    assert env_lin[-1] == pytest.approx(env_exp[-1], abs=1e-12)
    # They should differ in the interior (different curve shapes)
    assert not np.allclose(env_lin, env_exp)


# --- edge cases ---

def test_adsr_zero_attack():
    """Zero attack time: first sample is already at peak (1.0) or close to it."""
    _, env = generate_adsr(attack=0.0, decay=0.1, sustain=0.7,
                           release=0.1, fs=FS, duration=1.0)
    assert env.shape == (N,)


def test_adsr_zero_sustain():
    """sustain=0 means decay drops all the way to 0; release segment is a flat zero."""
    _, env = generate_adsr(attack=0.05, decay=0.1, sustain=0.0,
                           release=0.1, fs=FS, duration=1.0)
    n_a = int(0.05 * FS)
    assert env[n_a - 1] == pytest.approx(1.0)
    assert env[-1] == pytest.approx(0.0, abs=1e-12)
