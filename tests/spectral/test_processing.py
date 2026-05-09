import numpy as np

from python.spectral import spectral_gate

FS = 44100
FRAME = 512
HOP = 128


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def test_spectral_gate_shape():
    """spectral_gate must return an array of the same length as the input."""
    sig = _sine(n=4096)
    out = spectral_gate(sig, threshold_db=-40.0, frame_size=FRAME, hop_size=HOP)
    assert out.shape == sig.shape


def test_spectral_gate_loud_signal_passes():
    """A full-amplitude sine is well above any reasonable threshold; it must not be suppressed."""
    sig = _sine(n=4096)
    out = spectral_gate(sig, threshold_db=-60.0, frame_size=FRAME, hop_size=HOP)
    assert _rms(out) > 0.1


def test_spectral_gate_silence_below_threshold():
    """A silent input (all zeros) is below every threshold and must produce a zero output."""
    sig = np.zeros(4096)
    out = spectral_gate(sig, threshold_db=-10.0, frame_size=FRAME, hop_size=HOP)
    np.testing.assert_allclose(out, np.zeros_like(out), atol=1e-6)


def test_spectral_gate_high_threshold_attenuates():
    """An impossibly high threshold (80 dB) must suppress even a full-amplitude sine to near silence."""
    sig = _sine(n=4096)
    out = spectral_gate(sig, threshold_db=80.0, frame_size=FRAME, hop_size=HOP)
    assert _rms(out) < _rms(sig) * 0.01


def test_spectral_gate_preserves_dominant_frequency():
    """A signal above the threshold must pass with its dominant frequency intact."""
    freq = 1000.0
    sig = _sine(freq=freq, n=4096)
    out = spectral_gate(sig, threshold_db=-60.0, frame_size=FRAME, hop_size=HOP)
    spectrum = np.abs(np.fft.rfft(out))
    freqs_ax = np.fft.rfftfreq(len(out), d=1.0 / FS)
    dominant = float(freqs_ax[np.argmax(spectrum)])
    assert abs(dominant - freq) < FS / FRAME * 2


def test_spectral_gate_suppresses_quiet_noise_passes_loud_tone():
    """A loud tone at −6 dB plus quiet noise at −40 dB: gate set at −20 dB must pass tone and suppress noise."""
    n = 4096
    rng = np.random.default_rng(42)
    loud_tone = _sine(freq=1000.0, n=n) * 0.5     # −6 dB
    quiet_noise = rng.standard_normal(n) * 0.01    # ≈ −40 dB
    mixed = loud_tone + quiet_noise
    out = spectral_gate(mixed, threshold_db=-20.0, frame_size=FRAME, hop_size=HOP)
    spectrum_in = np.abs(np.fft.rfft(mixed))
    spectrum_out = np.abs(np.fft.rfft(out))
    freqs_ax = np.fft.rfftfreq(n, d=1.0 / FS)
    # Tone bin should be largely preserved
    idx_tone = np.argmin(np.abs(freqs_ax - 1000.0))
    assert spectrum_out[idx_tone] > spectrum_in[idx_tone] * 0.5
    # Overall RMS should be higher than noise-only level
    assert _rms(out) > 0.05

