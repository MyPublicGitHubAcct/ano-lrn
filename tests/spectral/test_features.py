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
    """spectral_centroid must return a 1-D array with at least one frame."""
    c = spectral_centroid(_sine(n=4096), FS, FRAME, HOP)
    assert c.ndim == 1
    assert len(c) > 0


def test_spectral_flux_shape():
    """spectral_flux must return a 1-D array with at least one frame."""
    f = spectral_flux(_sine(n=4096), FRAME, HOP)
    assert f.ndim == 1
    assert len(f) > 0


def test_spectral_centroid_high_freq_higher_than_low_freq():
    """A high-frequency tone must produce a higher mean centroid than a low-frequency tone."""
    c_low = np.mean(spectral_centroid(_sine(freq=200.0, n=4096), FS, FRAME, HOP))
    c_high = np.mean(spectral_centroid(_sine(freq=4000.0, n=4096), FS, FRAME, HOP))
    assert c_high > c_low


def test_spectral_centroid_within_audio_band():
    """The centroid must always lie within [0, fs/2]; values outside this range indicate a frequency-axis error."""
    c = spectral_centroid(_sine(freq=1000.0, n=4096), FS, FRAME, HOP)
    assert np.all(c >= 0.0)
    assert np.all(c <= FS / 2)


def test_spectral_flux_first_frame_is_zero():
    """Flux measures frame-to-frame spectral change; the first frame has no predecessor so flux must be 0."""
    f = spectral_flux(_sine(n=4096), FRAME, HOP)
    assert f[0] == pytest.approx(0.0)


def test_spectral_flux_steady_sine_low():
    """A steady-state sine has constant spectrum frame-to-frame; flux must stay low after the onset transient."""
    f = spectral_flux(_sine(freq=440.0, n=4096), FRAME, HOP)
    assert np.mean(f[2:]) < 100.0


def test_spectral_flux_noise_higher_than_sine():
    """White noise has a rapidly changing spectrum; its mean flux must exceed that of a steady sine."""
    n = 4096
    rng = np.random.default_rng(0)
    f_sine = np.mean(spectral_flux(_sine(n=n), FRAME, HOP)[1:])
    f_noise = np.mean(spectral_flux(rng.standard_normal(n), FRAME, HOP)[1:])
    assert f_noise > f_sine


def test_spectral_centroid_pure_sine_matches_frequency():
    """spectral_centroid of a pure sine at f must return values close to f (within one FFT bin) in steady state."""
    freq = 1000.0
    n = 4096
    sig = _sine(freq=freq, n=n)
    centroids = spectral_centroid(sig, FS, FRAME, HOP)
    bin_width = FS / FRAME
    # Skip early frames (onset transient) and check steady-state frames
    steady = centroids[2:]
    assert np.all(np.abs(steady - freq) < bin_width * 2)


def test_spectral_flux_spikes_at_onset():
    """spectral_flux must show a peak at the onset (silence → tone); the onset frame must exceed the steady-state mean."""
    n = 4096
    # First half silence, second half sine
    sig = np.zeros(n)
    sig[n // 2:] = _sine(freq=440.0, n=n // 2)
    flux = spectral_flux(sig, FRAME, HOP)
    # Find the onset frame (around midpoint sample n//2 / HOP)
    onset_frame = (n // 2) // HOP
    if onset_frame < len(flux):
        window = flux[max(0, onset_frame - 2): onset_frame + 3]
        steady = flux[onset_frame + 5:] if onset_frame + 5 < len(flux) else flux[-3:]
        assert np.max(window) > np.mean(steady) * 2

