import numpy as np
import pytest

from python.virtual_analog import analog_saturate, diode_clip, wavefold

FS = 44100


def _sine(freq=440.0, n=2048):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_diode_clip_shape():
    """Filter must return an array of the same length as the input."""
    assert diode_clip(_sine()).shape == _sine().shape


def test_analog_saturate_shape():
    """Saturator must return an array of the same length as the input."""
    assert analog_saturate(_sine()).shape == _sine().shape


def test_diode_clip_positive_hard_clipped():
    """Samples above the threshold must be clamped to exactly the threshold value."""
    sig = np.array([0.5, 0.7, 0.9, 1.5])
    out = diode_clip(sig, threshold=0.7)
    assert out[2] == pytest.approx(0.7, abs=1e-6)
    assert out[3] == pytest.approx(0.7, abs=1e-6)


def test_diode_clip_positive_unchanged_below_threshold():
    """Samples below the positive threshold must pass through unaltered."""
    sig = np.array([0.3, 0.5])
    out = diode_clip(sig, threshold=0.7)
    np.testing.assert_allclose(out, sig, atol=1e-12)


def test_diode_clip_asymmetric():
    """Diode clipping is asymmetric: positive peak must equal threshold while negative peak is lower in magnitude."""
    t = np.linspace(-1.5, 1.5, 200)
    out = diode_clip(t, threshold=0.7)
    pos_peak = np.max(out)
    neg_peak = np.min(out)
    assert pos_peak == pytest.approx(0.7, abs=1e-6)
    assert neg_peak > -0.7


def test_analog_saturate_zero_input_zero_output():
    """tanh(0) = 0 and scaling preserves zero; a zero input must produce a zero output."""
    np.testing.assert_allclose(analog_saturate(np.array([0.0])), [0.0], atol=1e-12)


def test_analog_saturate_bounded():
    """With any drive value the saturator output must remain strictly within (−1, +1)."""
    sig = np.linspace(-10.0, 10.0, 500)
    out = analog_saturate(sig, drive=5.0)
    assert np.max(np.abs(out)) < 1.0


def test_analog_saturate_odd_symmetry():
    """tanh-based saturation has odd symmetry: f(−x) = −f(x) for all x."""
    sig = np.linspace(-1.0, 1.0, 100)
    np.testing.assert_allclose(analog_saturate(-sig), -analog_saturate(sig), atol=1e-12)


def test_analog_saturate_monotone():
    """A saturator must be strictly monotone; non-monotonicity would create phase-cancellation artefacts."""
    sig = np.linspace(-1.0, 1.0, 100)
    out = analog_saturate(sig)
    assert np.all(np.diff(out) > 0)


def test_analog_saturate_small_signal_gain_near_unity():
    """For small inputs (|x| ≪ 1) the small-signal gain must be within 1% of unity with drive=1."""
    sig = np.linspace(-0.1, 0.1, 101)
    out = analog_saturate(sig, drive=1.0)
    nonzero = np.abs(sig) > 0.005
    ratio = out[nonzero] / sig[nonzero]
    np.testing.assert_allclose(ratio, 1.0, rtol=0.01)


def test_diode_clip_threshold_zero_clips_positive_to_zero():
    """With threshold=0, all positive samples must be clipped to 0 and the output must be finite."""
    sig = np.array([0.5, 1.0, -0.5, -1.0])
    out = diode_clip(sig, threshold=0.0)
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    assert out[1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# wavefold
# ---------------------------------------------------------------------------


def test_wavefold_shape():
    """wavefold must return an array of the same length as the input."""
    assert wavefold(_sine()).shape == _sine().shape


def test_wavefold_bounded():
    """Output must stay in [-1, +1] for any gain value."""
    sig = _sine()
    for gain in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        out = wavefold(sig, gain=gain)
        assert np.all(out >= -1.0 - 1e-12) and np.all(out <= 1.0 + 1e-12), f"gain={gain} out of bounds"


def test_wavefold_gain_zero_silence():
    """gain=0 must produce an all-zero output (silence)."""
    out = wavefold(_sine(), gain=0.0)
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_wavefold_harmonic_energy_increases_with_gain():
    """Higher gain must add more harmonic energy above 2× the fundamental."""
    FS = 44100
    FREQ = 440.0
    N = 4096
    t = np.arange(N) / FS
    sig = np.sin(2 * np.pi * FREQ * t)
    freqs = np.fft.rfftfreq(N, d=1.0 / FS)
    above_2f = freqs > 2 * FREQ

    energies = []
    for gain in [0.5, 1.0, 2.0, 3.0]:
        spectrum = np.abs(np.fft.rfft(wavefold(sig, gain=gain)))
        energies.append(float(np.sum(spectrum[above_2f] ** 2)))

    for i in range(len(energies) - 1):
        assert energies[i] < energies[i + 1], (
            f"harmonic energy did not increase from gain index {i} to {i+1}: {energies}"
        )


def test_wavefold_known_values():
    """Verify the reflection formula at specific sample values."""
    # x=0.5  → no fold, output 0.5
    # x=1.5  → fold once: 2-1.5=0.5
    # x=2.5  → fold once: 2-2.5=-0.5
    # x=-1.5 → fold once (negative): -2-(-1.5)=-0.5
    sig = np.array([0.5, 1.5, 2.5, -1.5])
    out = wavefold(sig, gain=1.0)
    np.testing.assert_allclose(out, [0.5, 0.5, -0.5, -0.5], atol=1e-12)


def test_wavefold_odd_symmetry():
    """wavefold has odd symmetry: wavefold(-x, g) = -wavefold(x, g)."""
    sig = np.linspace(-3.0, 3.0, 200)
    np.testing.assert_allclose(wavefold(-sig, gain=2.0), -wavefold(sig, gain=2.0), atol=1e-12)

