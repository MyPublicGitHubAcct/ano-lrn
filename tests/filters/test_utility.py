import numpy as np
import pytest

from python.filters import dc_block
from python.generators import generate_dc, generate_sine

from tests.filters.conftest import FS, DURATION, _dc, _nyquist_sig, _sine, _steady_rms


def test_dc_block_output_shape():
    """Filter must return an array of the same length as the input."""
    _, sig = generate_dc(fs=FS, duration=DURATION)
    out = dc_block(sig, cutoff=20.0, fs=FS)
    assert out.shape == sig.shape


def test_dc_block_rejects_dc():
    """The filter's primary purpose: a constant signal must be reduced to near zero in steady state."""
    out = dc_block(_dc(1.0), cutoff=20.0, fs=FS)
    assert _steady_rms(out) < 0.01


def test_dc_block_passes_high_frequency():
    """Audio-band signals well above the cutoff must pass without meaningful attenuation."""
    sig = _sine(1000.0)
    out = dc_block(sig, cutoff=20.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.99


def test_dc_block_passes_nyquist():
    """Nyquist is the furthest possible frequency from the 20 Hz cutoff; it must pass unattenuated."""
    sig = _nyquist_sig()
    out = dc_block(sig, cutoff=20.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(1.0, abs=0.01)


def test_dc_block_attenuates_sub_sonic():
    """Sub-sonic frequencies (2 Hz) are below the cutoff and must be meaningfully attenuated."""
    _, sig = generate_sine(freq=2.0, fs=FS, duration=DURATION)
    out = dc_block(sig, cutoff=20.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.15


def test_dc_block_minus3db_at_cutoff():
    """The highpass pole is defined by the cutoff; gain at that frequency must be −3 dB (±2 dB tolerance)."""
    cutoff = 100.0
    _, sig = generate_sine(freq=cutoff, fs=FS, duration=DURATION)
    out = dc_block(sig, cutoff=cutoff, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(1.0 / np.sqrt(2), abs=0.02)


def test_dc_block_near_zero_cutoff_passes_audio_band():
    """A dc_block with a very low cutoff (1 Hz) must not attenuate audio-band content."""
    sig = _sine(1000.0)
    out = dc_block(sig, cutoff=1.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.99


def test_dc_block_two_cascades_stable():
    """Two cascaded dc_block calls must not cause instability; audio-band RMS must remain comparable to one pass."""
    sig = _sine(1000.0)
    out1 = dc_block(sig, cutoff=20.0, fs=FS)
    out2 = dc_block(out1, cutoff=20.0, fs=FS)
    assert np.all(np.isfinite(out2))
    # Second pass should not more than halve the RMS (< 6 dB extra attenuation at 1 kHz)
    assert _steady_rms(out2) / _steady_rms(out1) > 0.5

