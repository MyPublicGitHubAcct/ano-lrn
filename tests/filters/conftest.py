import numpy as np
import pytest

from python.filters import allpass, bandpass, highpass, highshelf, lowpass, lowshelf, notch
from python.generators import (
    generate_dc,
    generate_half_nyquist,
    generate_nyquist,
    generate_quarter_nyquist,
    generate_sine,
)

FS = 44100
DURATION = 1.0
SKIP = int(0.05 * FS)
ALL_FILTERS = [lowpass, highpass, bandpass, notch, allpass, lowshelf, highshelf]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _steady_rms(signal: np.ndarray) -> float:
    return _rms(signal[SKIP:])


def _sine(freq: float) -> np.ndarray:
    _, w = generate_sine(freq=freq, fs=FS, duration=DURATION)
    return w


def _dc(amplitude: float = 1.0) -> np.ndarray:
    _, w = generate_dc(fs=FS, duration=DURATION, amplitude=amplitude)
    return w


def _nyquist_sig() -> np.ndarray:
    _, w = generate_nyquist(fs=FS, duration=DURATION)
    return w


def _half_nyquist_sig() -> np.ndarray:
    _, w = generate_half_nyquist(fs=FS, duration=DURATION)
    return w


def _quarter_nyquist_sig() -> np.ndarray:
    _, w = generate_quarter_nyquist(fs=FS, duration=DURATION)
    return w


@pytest.fixture(params=[500.0, 1000.0, 4000.0])
def cutoff(request):
    return request.param


@pytest.fixture(params=[0.5, 0.707, 2.0])
def Q_value(request):
    return request.param


@pytest.fixture(
    params=ALL_FILTERS,
    ids=["lowpass", "highpass", "bandpass", "notch", "allpass", "lowshelf", "highshelf"],
)
def any_filter(request):
    return request.param
