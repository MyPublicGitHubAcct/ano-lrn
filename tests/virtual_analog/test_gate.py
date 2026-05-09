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
    """Gate must return an array of the same length as the input signal."""
    sig = _sine()
    ctrl = _gate(len(sig))
    assert lowpass_gate(sig, ctrl, fs=FS).shape == sig.shape


def test_lowpass_gate_control_length_mismatch_raises():
    """Mismatched signal and control lengths must raise ValueError to prevent silent sample misalignment."""
    sig = _sine(n=1024)
    ctrl = np.ones(512)
    with pytest.raises(ValueError):
        lowpass_gate(sig, ctrl, fs=FS)


def test_lowpass_gate_closed_gate_near_silence():
    """A permanently closed control (all zeros) must suppress the output to near silence."""
    sig = _sine(n=8192)
    ctrl = np.zeros(len(sig))
    out = lowpass_gate(sig, ctrl, fs=FS)
    # Fully closed — output should be essentially silent
    assert np.max(np.abs(out)) < 0.01


def test_lowpass_gate_open_gate_passes_signal():
    """A permanently open gate must pass the signal with no more than −6 dB attenuation."""
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
    """Amplitude-only mode applies VCA envelope but no frequency filtering; high-frequency content must pass."""
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
    """Lowpass-only mode sweeps the filter cutoff without amplitude VCA; closed gate must heavily attenuate high freq."""
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
    """Random control voltages must not cause NaN or Inf in any operating mode."""
    sig = _sine(n=4096)
    ctrl = np.random.default_rng(0).uniform(0, 1, len(sig))
    out = lowpass_gate(sig, ctrl, fs=FS)
    assert np.all(np.isfinite(out))


def test_lowpass_gate_mode_both_more_attenuated_than_amplitude_at_high_freq():
    """mode='both' applies filter+VCA; high-frequency output must be more attenuated than amplitude-only mode."""
    freq = 5000.0
    n = FS
    sig = np.sin(2 * np.pi * freq * np.arange(n) / FS)
    ctrl = np.ones(n) * 0.5  # half-open gate, low cutoff
    out_both = lowpass_gate(sig, ctrl, fs=FS, mode="both", max_cutoff=4000.0)
    out_amp = lowpass_gate(sig, ctrl, fs=FS, mode="amplitude")
    rms_both = float(np.sqrt(np.mean(out_both[n // 2:] ** 2)))
    rms_amp = float(np.sqrt(np.mean(out_amp[n // 2:] ** 2)))
    assert rms_both < rms_amp


def test_lowpass_gate_attack_time_constant_respected():
    """After a step open, output must reach ~63% of steady-state within one attack_time (one RC time constant)."""
    attack_time = 0.050  # 50 ms
    n = int(FS * 1.0)
    sig = np.ones(n)
    ctrl = np.ones(n)  # gate opens immediately
    out = lowpass_gate(sig, ctrl, fs=FS, mode="amplitude",
                       attack_time=attack_time, release_time=1.0)
    t_one_tau = int(attack_time * FS)
    # After one time constant, the one-pole smoother reaches 1 - 1/e ≈ 0.632
    assert 0.45 < out[t_one_tau] < 0.85

