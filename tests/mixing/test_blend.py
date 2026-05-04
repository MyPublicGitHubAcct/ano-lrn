import numpy as np

from python.mixing import crossfade, mix

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_mix_shape():
    a = _sine(440.0)
    b = _sine(880.0)
    assert mix([a, b]).shape == a.shape


def test_crossfade_shape():
    a = _sine(440.0)
    b = _sine(880.0)
    assert crossfade(a, b, 0.5).shape == a.shape


def test_mix_equal_weight_sums_to_mean():
    a = np.ones(100)
    b = np.ones(100) * 3.0
    out = mix([a, b])
    np.testing.assert_allclose(out, np.full(100, 2.0), atol=1e-12)


def test_mix_custom_weights():
    a = np.ones(100)
    b = np.zeros(100)
    out = mix([a, b], weights=[0.75, 0.25])
    np.testing.assert_allclose(out, np.full(100, 0.75), atol=1e-12)


def test_mix_single_signal_weight_1_identity():
    sig = _sine()
    np.testing.assert_allclose(mix([sig], weights=[1.0]), sig, atol=1e-12)


def test_crossfade_position_0_is_a():
    a = _sine(440.0)
    b = _sine(880.0)
    np.testing.assert_allclose(crossfade(a, b, 0.0), a, atol=1e-12)


def test_crossfade_position_1_is_b():
    a = _sine(440.0)
    b = _sine(880.0)
    np.testing.assert_allclose(crossfade(a, b, 1.0), b, atol=1e-12)


def test_crossfade_midpoint():
    a = np.ones(100)
    b = np.zeros(100)
    out = crossfade(a, b, 0.5)
    np.testing.assert_allclose(out, np.full(100, 0.5), atol=1e-12)
