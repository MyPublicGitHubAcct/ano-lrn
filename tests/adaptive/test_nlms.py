import numpy as np

from python.adaptive import lms, nlms

FS = 44100


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def test_nlms_output_shape():
    """nlms must return (output, error, weights) with output and error matching the desired length and weights matching filter_order."""
    t = np.arange(2048) / FS
    ref = np.sin(2 * np.pi * 440.0 * t)
    des = np.sin(2 * np.pi * 880.0 * t)
    out, err, w = nlms(des, ref, filter_order=16, mu=0.5)
    assert out.shape == des.shape
    assert err.shape == des.shape
    assert w.shape == (16,)


def test_nlms_converges_faster_than_lms():
    """NLMS normalises the step size by signal power, so it should converge faster than plain LMS on an identifiable system."""
    n = 2048
    ref = np.random.default_rng(2).standard_normal(n)
    delay = 3
    des = np.concatenate([np.zeros(delay), ref[:-delay]])
    _, err_lms, _ = lms(des, ref, filter_order=16, mu=0.001)
    _, err_nlms, _ = nlms(des, ref, filter_order=16, mu=0.5)
    assert _rms(err_nlms[n // 2:]) < _rms(err_lms[n // 2:])


def test_nlms_weights_not_all_zero():
    """After running on non-trivial data, at least one weight must be non-zero (basic sanity check that adaptation occurs)."""
    n = 2048
    ref = np.random.default_rng(3).standard_normal(n)
    des = np.random.default_rng(4).standard_normal(n)
    _, _, w = nlms(des, ref, filter_order=8, mu=0.5)
    assert not np.all(w == 0.0)


def test_nlms_mu_zero_weights_stay_zero():
    """With mu=0 the update term vanishes; all weights must remain zero throughout."""
    n = 2048
    ref = np.random.default_rng(10).standard_normal(n)
    des = np.random.default_rng(11).standard_normal(n)
    _, _, w = nlms(des, ref, filter_order=16, mu=0.0)
    np.testing.assert_array_equal(w, np.zeros(16))


def test_nlms_silent_reference_no_nan():
    """When the reference is all zeros, epsilon prevents division by zero; output must be finite."""
    n = 2048
    ref = np.zeros(n)
    des = np.random.default_rng(12).standard_normal(n)
    out, err, w = nlms(des, ref, filter_order=16, mu=0.5, eps=1e-8)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(err))
    assert np.all(np.isfinite(w))

