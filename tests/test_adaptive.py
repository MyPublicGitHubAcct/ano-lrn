import numpy as np
import pytest

from python.adaptive import lms, nlms

FS = 44100


def _sine(freq=440.0, n=2048):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


# ── Shape ─────────────────────────────────────────────────────────────────────

def test_lms_output_shape():
    ref = _sine()
    des = _sine(880.0)
    out, err, w = lms(des, ref, filter_order=16, mu=0.001)
    assert out.shape == des.shape
    assert err.shape == des.shape
    assert w.shape == (16,)


def test_nlms_output_shape():
    ref = _sine()
    des = _sine(880.0)
    out, err, w = nlms(des, ref, filter_order=16, mu=0.5)
    assert out.shape == des.shape
    assert err.shape == des.shape
    assert w.shape == (16,)


# ── LMS convergence ───────────────────────────────────────────────────────────

def test_lms_error_decreases_for_identifiable_system():
    # Desired = delayed version of reference (pure delay is a linear system).
    n = 4096
    ref = np.random.default_rng(0).standard_normal(n)
    delay = 5
    des = np.concatenate([np.zeros(delay), ref[:-delay]])
    out, err, w = lms(des, ref, filter_order=32, mu=0.01)
    rms_early = _rms(err[:n // 4])
    rms_late = _rms(err[3 * n // 4 :])
    assert rms_late < rms_early


def test_lms_identity_system_converges():
    # Desired = reference → optimal filter is a unit impulse.
    n = 4096
    ref = np.random.default_rng(1).standard_normal(n)
    out, err, w = lms(ref, ref, filter_order=16, mu=0.01)
    # After convergence, error should be close to zero.
    assert _rms(err[3 * n // 4:]) < 0.1


# ── NLMS convergence ──────────────────────────────────────────────────────────

def test_nlms_converges_faster_than_lms():
    n = 2048
    ref = np.random.default_rng(2).standard_normal(n)
    delay = 3
    des = np.concatenate([np.zeros(delay), ref[:-delay]])
    _, err_lms, _ = lms(des, ref, filter_order=16, mu=0.001)
    _, err_nlms, _ = nlms(des, ref, filter_order=16, mu=0.5)
    # NLMS with a well-chosen mu converges to a lower steady-state error.
    assert _rms(err_nlms[n // 2:]) < _rms(err_lms[n // 2:])


def test_nlms_weights_not_all_zero():
    n = 2048
    ref = np.random.default_rng(3).standard_normal(n)
    des = np.random.default_rng(4).standard_normal(n)
    _, _, w = nlms(des, ref, filter_order=8, mu=0.5)
    assert not np.all(w == 0.0)
