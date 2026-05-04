import numpy as np
import pytest

from python.filters import dc_block
from python.generators import generate_dc, generate_sine

from tests.filters.conftest import FS, DURATION, _dc, _nyquist_sig, _sine, _steady_rms


def test_dc_block_output_shape():
    _, sig = generate_dc(fs=FS, duration=DURATION)
    out = dc_block(sig, cutoff=20.0, fs=FS)
    assert out.shape == sig.shape


def test_dc_block_rejects_dc():
    out = dc_block(_dc(1.0), cutoff=20.0, fs=FS)
    assert _steady_rms(out) < 0.01


def test_dc_block_passes_high_frequency():
    sig = _sine(1000.0)
    out = dc_block(sig, cutoff=20.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.99


def test_dc_block_passes_nyquist():
    sig = _nyquist_sig()
    out = dc_block(sig, cutoff=20.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(1.0, abs=0.01)


def test_dc_block_attenuates_sub_sonic():
    _, sig = generate_sine(freq=2.0, fs=FS, duration=DURATION)
    out = dc_block(sig, cutoff=20.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.15


def test_dc_block_minus3db_at_cutoff():
    cutoff = 100.0
    _, sig = generate_sine(freq=cutoff, fs=FS, duration=DURATION)
    out = dc_block(sig, cutoff=cutoff, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(1.0 / np.sqrt(2), abs=0.02)
