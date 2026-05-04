import numpy as np
import pytest

from python.spectral import spectral_centroid, spectral_flux

FS = 44100
FRAME = 512
HOP = 128


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_spectral_centroid_shape():
    c = spectral_centroid(_sine(n=4096), FS, FRAME, HOP)
    assert c.ndim == 1
    assert len(c) > 0


def test_spectral_flux_shape():
    f = spectral_flux(_sine(n=4096), FRAME, HOP)
    assert f.ndim == 1
    assert len(f) > 0


def test_spectral_centroid_high_freq_higher_than_low_freq():
    c_low = np.mean(spectral_centroid(_sine(freq=200.0, n=4096), FS, FRAME, HOP))
    c_high = np.mean(spectral_centroid(_sine(freq=4000.0, n=4096), FS, FRAME, HOP))
    assert c_high > c_low


def test_spectral_centroid_within_audio_band():
    c = spectral_centroid(_sine(freq=1000.0, n=4096), FS, FRAME, HOP)
    assert np.all(c >= 0.0)
    assert np.all(c <= FS / 2)


def test_spectral_flux_first_frame_is_zero():
    f = spectral_flux(_sine(n=4096), FRAME, HOP)
    assert f[0] == pytest.approx(0.0)


def test_spectral_flux_steady_sine_low():
    f = spectral_flux(_sine(freq=440.0, n=4096), FRAME, HOP)
    assert np.mean(f[2:]) < 100.0


def test_spectral_flux_noise_higher_than_sine():
    n = 4096
    rng = np.random.default_rng(0)
    f_sine = np.mean(spectral_flux(_sine(n=n), FRAME, HOP)[1:])
    f_noise = np.mean(spectral_flux(rng.standard_normal(n), FRAME, HOP)[1:])
    assert f_noise > f_sine
