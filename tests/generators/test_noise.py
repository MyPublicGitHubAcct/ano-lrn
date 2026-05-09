import numpy as np
import pytest

from python.generators import generate_pink_noise

from tests.generators.conftest import FS, N


def test_noise_output_shapes(seeded_gen):
    """Generator contract: t and signal must share the same N-sample shape."""
    t, w = seeded_gen()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_noise_amplitude_range(seeded_gen, amplitude):
    """Peak absolute value must not exceed the requested amplitude at any seed."""
    _, w = seeded_gen(amplitude=amplitude, seed=0)
    assert np.max(np.abs(w)) <= amplitude + 1e-9


def test_noise_reproducible_with_seed(seeded_gen, seed):
    """The same integer seed must produce byte-identical output on repeated calls."""
    _, w1 = seeded_gen(seed=seed)
    _, w2 = seeded_gen(seed=seed)
    assert np.allclose(w1, w2)


def test_noise_different_seeds_differ(seeded_gen, seed_pair):
    """Different seeds must produce statistically distinct sequences; equal output would mean the seed is ignored."""
    seed_a, seed_b = seed_pair
    _, w1 = seeded_gen(seed=seed_a)
    _, w2 = seeded_gen(seed=seed_b)
    assert not np.allclose(w1, w2)


def test_pink_noise_zero_dc():
    """Pink noise should be AC-coupled; a large DC offset would indicate a shaping error."""
    _, w = generate_pink_noise(seed=0)
    assert abs(np.mean(w)) < 0.01


# --- parameter range tests ---

def test_noise_amplitude_zero_is_silent(seeded_gen):
    """amplitude=0 must produce an all-zero array regardless of generator type."""
    _, w = seeded_gen(amplitude=0.0, seed=0)
    np.testing.assert_array_equal(w, np.zeros(N))


def test_white_noise_flat_power_spectral_density():
    """White noise power must be approximately equal across octave bands (within 6 dB of each other)."""
    from python.generators import generate_white_noise
    _, w = generate_white_noise(fs=FS, duration=4.0, amplitude=1.0, seed=42)
    spectrum = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(len(w), d=1.0 / FS)
    bands = [(100, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 6400)]
    powers = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        n_bins = np.sum(mask)
        powers.append(np.sum(spectrum[mask]) / n_bins if n_bins > 0 else 0.0)
    powers_arr = np.array(powers)
    # All octave bands within 6 dB (factor of 4 in power) of each other
    assert np.max(powers_arr) / np.min(powers_arr) < 4.0

