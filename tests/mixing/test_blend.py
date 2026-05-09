import numpy as np

from python.mixing import crossfade, mix

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_mix_shape():
    """mix must return an array of the same length as its inputs."""
    a = _sine(440.0)
    b = _sine(880.0)
    assert mix([a, b]).shape == a.shape


def test_crossfade_shape():
    """crossfade must return an array of the same length as its inputs."""
    a = _sine(440.0)
    b = _sine(880.0)
    assert crossfade(a, b, 0.5).shape == a.shape


def test_mix_equal_weight_sums_to_mean():
    """With equal weights, mix must produce the arithmetic mean of the inputs."""
    a = np.ones(100)
    b = np.ones(100) * 3.0
    out = mix([a, b])
    np.testing.assert_allclose(out, np.full(100, 2.0), atol=1e-12)


def test_mix_custom_weights():
    """Explicit weights must produce a weighted sum of the inputs."""
    a = np.ones(100)
    b = np.zeros(100)
    out = mix([a, b], weights=[0.75, 0.25])
    np.testing.assert_allclose(out, np.full(100, 0.75), atol=1e-12)


def test_mix_single_signal_weight_1_identity():
    """A single-element list with weight=1.0 must return the input unchanged."""
    sig = _sine()
    np.testing.assert_allclose(mix([sig], weights=[1.0]), sig, atol=1e-12)


def test_crossfade_position_0_is_a():
    """position=0.0 must return signal a unchanged — the full dry case."""
    a = _sine(440.0)
    b = _sine(880.0)
    np.testing.assert_allclose(crossfade(a, b, 0.0), a, atol=1e-12)


def test_crossfade_position_1_is_b():
    """position=1.0 must return signal b unchanged — the full wet case."""
    a = _sine(440.0)
    b = _sine(880.0)
    np.testing.assert_allclose(crossfade(a, b, 1.0), b, atol=1e-12)


def test_crossfade_midpoint():
    """position=0.5 must produce equal parts a and b (linear blend of constant signals)."""
    a = np.ones(100)
    b = np.zeros(100)
    out = crossfade(a, b, 0.5)
    np.testing.assert_allclose(out, np.full(100, 0.5), atol=1e-12)


def test_mix_all_zero_weights_returns_zero():
    """Mixing with all-zero weights must return an all-zero array."""
    a = _sine(440.0)
    b = _sine(880.0)
    out = mix([a, b], weights=[0.0, 0.0])
    np.testing.assert_array_equal(out, np.zeros_like(a))


def test_crossfade_linearity_at_intermediate_position():
    """crossfade at any position p must equal (1−p)·a + p·b for non-constant signals."""
    a = _sine(440.0)
    b = _sine(880.0)
    for pos in [0.25, 0.5, 0.75]:
        out = crossfade(a, b, pos)
        expected = (1 - pos) * a + pos * b
        np.testing.assert_allclose(out, expected, atol=1e-12)

