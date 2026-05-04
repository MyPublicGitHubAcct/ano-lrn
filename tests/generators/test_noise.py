import numpy as np
import pytest

from python.generators import generate_pink_noise

from tests.generators.conftest import FS, N


def test_noise_output_shapes(seeded_gen):
    t, w = seeded_gen()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_noise_amplitude_range(seeded_gen, amplitude):
    _, w = seeded_gen(amplitude=amplitude, seed=0)
    assert np.max(np.abs(w)) <= amplitude + 1e-9


def test_noise_reproducible_with_seed(seeded_gen, seed):
    _, w1 = seeded_gen(seed=seed)
    _, w2 = seeded_gen(seed=seed)
    assert np.allclose(w1, w2)


def test_noise_different_seeds_differ(seeded_gen, seed_pair):
    seed_a, seed_b = seed_pair
    _, w1 = seeded_gen(seed=seed_a)
    _, w2 = seeded_gen(seed=seed_b)
    assert not np.allclose(w1, w2)


def test_pink_noise_zero_dc():
    _, w = generate_pink_noise(seed=0)
    assert abs(np.mean(w)) < 0.01


# --- parameter range tests ---

def test_noise_amplitude_zero_is_silent(seeded_gen):
    _, w = seeded_gen(amplitude=0.0, seed=0)
    np.testing.assert_array_equal(w, np.zeros(N))
