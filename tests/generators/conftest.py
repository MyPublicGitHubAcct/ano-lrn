import numpy as np
import pytest

from python.generators import (
    generate_half_nyquist,
    generate_nyquist,
    generate_pink_noise,
    generate_quarter_nyquist,
    generate_sawtooth,
    generate_sine,
    generate_square,
    generate_triangle,
    generate_white_noise,
)

FS = 44100
DURATION = 1.0
N = int(FS * DURATION)

PERIODIC_GENERATORS = [generate_sine, generate_square, generate_sawtooth, generate_triangle]
SEEDED_GENERATORS = [generate_white_noise, generate_pink_noise]
_NYQUIST_GENERATORS = [generate_nyquist, generate_half_nyquist, generate_quarter_nyquist]

NYQUIST_PATTERNS = {
    generate_nyquist: np.array([-1.0, 1.0]),
    generate_half_nyquist: np.array([-1.0, 0.0, 1.0, 0.0]),
    generate_quarter_nyquist: np.array([
        -1.0, -1 / np.sqrt(2), 0.0, 1 / np.sqrt(2),
        1.0, 1 / np.sqrt(2), 0.0, -1 / np.sqrt(2),
    ]),
}


@pytest.fixture(params=PERIODIC_GENERATORS, ids=lambda f: f.__name__)
def periodic_gen(request):
    return request.param


@pytest.fixture(params=SEEDED_GENERATORS, ids=lambda f: f.__name__)
def seeded_gen(request):
    return request.param


@pytest.fixture(params=[0.5, 1.0, 2.0])
def amplitude(request):
    return request.param


@pytest.fixture(params=[110.0, 440.0, 880.0])
def freq(request):
    return request.param


@pytest.fixture(params=[(22050, 0.5), (48000, 2.0)], ids=["22050Hz-0.5s", "48000Hz-2.0s"])
def fs_and_duration(request):
    return request.param


@pytest.fixture(params=[np.pi / 4, np.pi / 2, np.pi])
def phase(request):
    return request.param


@pytest.fixture(params=[0.25, 0.5, 0.75])
def duty(request):
    return request.param


@pytest.fixture(params=[0, 42, 123])
def seed(request):
    return request.param


@pytest.fixture(params=[(0, 1), (42, 43)])
def seed_pair(request):
    return request.param


@pytest.fixture(params=[0.0, 0.1, 0.5, 0.9])
def delay(request):
    return request.param


@pytest.fixture(params=[0.0, 0.25, 0.5, 0.75])
def onset(request):
    return request.param


@pytest.fixture(params=["linear", "logarithmic"])
def chirp_method(request):
    return request.param


@pytest.fixture(params=[-1.0, 0.0, 0.5, 1.0, 2.0])
def dc_amplitude(request):
    return request.param


@pytest.fixture(
    params=[[440.0], [220.0, 880.0], [110.0, 440.0, 1760.0]],
    ids=["1-tone", "2-tone", "3-tone"],
)
def multi_tone_freqs(request):
    return request.param


@pytest.fixture(params=_NYQUIST_GENERATORS, ids=lambda f: f.__name__)
def nyquist_gen(request):
    return request.param
