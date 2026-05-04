import numpy as np
import pytest

from python.virtual_analog import lowpass_gate

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def _gate(n, open_frac=0.5):
    ctrl = np.zeros(n)
    ctrl[: int(n * open_frac)] = 1.0
    return ctrl


def test_lowpass_gate_shape():
    sig = _sine()
    ctrl = _gate(len(sig))
    assert lowpass_gate(sig, ctrl, fs=FS).shape == sig.shape


def test_lowpass_gate_control_length_mismatch_raises():
    sig = _sine(n=1024)
    ctrl = np.ones(512)
    with pytest.raises(ValueError):
        lowpass_gate(sig, ctrl, fs=FS)


def test_lowpass_gate_closed_gate_near_silence():
    sig = _sine(n=8192)
    ctrl = np.zeros(len(sig))
    out = lowpass_gate(sig, ctrl, fs=FS)
    # Fully closed — output should be essentially silent
    assert np.max(np.abs(out)) < 0.01


def test_lowpass_gate_open_gate_passes_signal():
    sig = _sine(n=8192)
    ctrl = np.ones(len(sig))
    out = lowpass_gate(sig, ctrl, fs=FS, mode="both")
    rms_in = float(np.sqrt(np.mean(sig[4096:] ** 2)))
    rms_out = float(np.sqrt(np.mean(out[4096:] ** 2)))
    # At full open the output should not be more than 6 dB below input
    assert rms_out > rms_in * 0.5


def test_lowpass_gate_attenuates_on_release():
    """After gate closes, output should decay — not stay at open level."""
    n = FS  # 1 second
    sig = _sine(n=n)
    ctrl = np.zeros(n)
    ctrl[:1000] = 1.0  # brief pulse
    out = lowpass_gate(sig, ctrl, fs=FS, mode="amplitude")
    rms_early = float(np.sqrt(np.mean(out[500:1000] ** 2)))   # gate open
    rms_late = float(np.sqrt(np.mean(out[20000:21000] ** 2)))  # well after gate
    assert rms_early > rms_late * 5


def test_lowpass_gate_mode_amplitude_no_filter():
    """Amplitude-only mode: a high-frequency sine should pass at full level."""
    freq = 10000.0
    n = 8192
    sig = np.sin(2 * np.pi * freq * np.arange(n) / FS)
    ctrl = np.ones(n)
    out = lowpass_gate(sig, ctrl, fs=FS, mode="amplitude",
                       attack_time=0.001, release_time=0.001)
    rms_in = float(np.sqrt(np.mean(sig[4096:] ** 2)))
    rms_out = float(np.sqrt(np.mean(out[4096:] ** 2)))
    # Should be close to the input — no filtering applied
    assert rms_out > rms_in * 0.8


def test_lowpass_gate_mode_lowpass_attenuates_high_freq():
    """Lowpass-only mode: high frequency should be attenuated at closed gate."""
    freq = 5000.0
    n = 8192
    sig = np.sin(2 * np.pi * freq * np.arange(n) / FS)
    ctrl = np.zeros(n)  # gate closed → cutoff at min_cutoff
    out = lowpass_gate(sig, ctrl, fs=FS, mode="lowpass",
                       min_cutoff=20.0, max_cutoff=8000.0)
    rms_out = float(np.sqrt(np.mean(out[4096:] ** 2)))
    rms_in = float(np.sqrt(np.mean(sig[4096:] ** 2)))
    assert rms_out < rms_in * 0.1


def test_lowpass_gate_vactrol_release_is_slower_than_attack():
    """The vactrol release must outlast the attack for the same Δcontrol."""
    n = FS * 2
    ctrl = np.zeros(n)
    ctrl[FS // 2: FS // 2 + 100] = 1.0  # 100-sample pulse
    sig = np.ones(n)
    out = lowpass_gate(sig, ctrl, fs=FS, mode="amplitude",
                       attack_time=0.005, release_time=0.300)
    peak_idx = int(np.argmax(out))
    # Find sample where output drops below 10% of peak
    tail = out[peak_idx:]
    below = np.where(tail < np.max(out) * 0.1)[0]
    release_samples = below[0] if len(below) else n
    # Release should span at least 0.1 s (4410 samples)
    assert release_samples > 4410


def test_lowpass_gate_output_finite():
    sig = _sine(n=4096)
    ctrl = np.random.default_rng(0).uniform(0, 1, len(sig))
    out = lowpass_gate(sig, ctrl, fs=FS)
    assert np.all(np.isfinite(out))
