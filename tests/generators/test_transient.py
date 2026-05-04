import numpy as np
import pytest

from python.generators import generate_impulse, generate_step

from tests.generators.conftest import FS, N


def test_impulse_output_shapes():
    t, w = generate_impulse()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_impulse_single_nonzero_sample():
    _, w = generate_impulse()
    assert np.sum(w != 0) == 1


def test_impulse_delay(delay):
    _, w = generate_impulse(delay=delay)
    idx = int(delay * FS)
    assert w[idx] == 1.0
    assert np.sum(w != 0) == 1


def test_impulse_amplitude(amplitude):
    _, w = generate_impulse(amplitude=amplitude)
    assert np.max(w) == pytest.approx(amplitude)
    assert np.sum(w != 0) == 1


def test_step_output_shapes():
    t, w = generate_step()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_step_onset(onset):
    _, w = generate_step(onset=onset, amplitude=1.0)
    idx = int(onset * FS)
    assert np.all(w[:idx] == 0.0)
    assert np.all(w[idx:] == pytest.approx(1.0))


def test_step_amplitude(amplitude):
    _, w = generate_step(onset=0.0, amplitude=amplitude)
    assert np.all(w == pytest.approx(amplitude))
