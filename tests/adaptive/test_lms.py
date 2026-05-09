import numpy as np

from python.adaptive import lms

FS = 44100


def _sine(freq=440.0, n=2048):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def test_lms_output_shape():
    """lms must return (output, error, weights) with output and error matching the desired length and weights matching filter_order."""
    ref = _sine()
    des = _sine(880.0)
    out, err, w = lms(des, ref, filter_order=16, mu=0.001)
    assert out.shape == des.shape
    assert err.shape == des.shape
    assert w.shape == (16,)


def test_lms_error_decreases_for_identifiable_system():
    """For a system where desired = delayed(ref), the filter is identifiable and error must decrease over time."""
    n = 4096
    ref = np.random.default_rng(0).standard_normal(n)
    delay = 5
    des = np.concatenate([np.zeros(delay), ref[:-delay]])
    out, err, w = lms(des, ref, filter_order=32, mu=0.01)
    rms_early = _rms(err[:n // 4])
    rms_late = _rms(err[3 * n // 4:])
    assert rms_late < rms_early


def test_lms_identity_system_converges():
    """When desired == ref, the filter has a trivial solution (impulse at tap 0); LMS must converge to near-zero error."""
    n = 4096
    ref = np.random.default_rng(1).standard_normal(n)
    out, err, w = lms(ref, ref, filter_order=16, mu=0.01)
    assert _rms(err[3 * n // 4:]) < 0.1

