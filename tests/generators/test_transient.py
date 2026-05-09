import numpy as np
import pytest

from python.generators import generate_impulse, generate_step

from tests.generators.conftest import FS, N


def test_impulse_output_shapes():
    """Generator contract: t and signal must share the same N-sample shape."""
    t, w = generate_impulse()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_impulse_single_nonzero_sample():
    """An ideal unit impulse has energy in exactly one sample; any additional non-zero samples are a bug."""
    _, w = generate_impulse()
    assert np.sum(w != 0) == 1


def test_impulse_delay(delay):
    """The impulse must land at sample index round(delay × fs) and nowhere else."""
    _, w = generate_impulse(delay=delay)
    idx = int(delay * FS)
    assert w[idx] == 1.0
    assert np.sum(w != 0) == 1


def test_impulse_amplitude(amplitude):
    """Peak value must equal the requested amplitude; single-sample energy scales as amplitude²."""
    _, w = generate_impulse(amplitude=amplitude)
    assert np.max(w) == pytest.approx(amplitude)
    assert np.sum(w != 0) == 1


def test_step_output_shapes():
    """Generator contract: t and signal must share the same N-sample shape."""
    t, w = generate_step()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_step_onset(onset):
    """Samples before the onset index must be zero; samples from onset onward must equal amplitude."""
    _, w = generate_step(onset=onset, amplitude=1.0)
    idx = int(onset * FS)
    assert np.all(w[:idx] == 0.0)
    assert np.all(w[idx:] == pytest.approx(1.0))


def test_step_amplitude(amplitude):
    """With onset=0 the entire signal must equal amplitude; this tests amplitude scaling directly."""
    _, w = generate_step(onset=0.0, amplitude=amplitude)
    assert np.all(w == pytest.approx(amplitude))

