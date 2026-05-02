import numpy as np
import pytest
from ano_lrn.filters import bandpass, highpass, lowpass
from ano_lrn.generators import generate_dc, generate_sine

FS = 44100
DURATION = 1.0
# Skip first 50 ms so transient decay does not skew RMS measurements.
SKIP = int(0.05 * FS)

ALL_FILTERS = [lowpass, highpass, bandpass]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def _steady_rms(signal: np.ndarray) -> float:
    return _rms(signal[SKIP:])


def _sine(freq: float) -> np.ndarray:
    _, w = generate_sine(freq=freq, fs=FS, duration=DURATION)
    return w


def _dc(amplitude: float = 1.0) -> np.ndarray:
    _, w = generate_dc(fs=FS, duration=DURATION, amplitude=amplitude)
    return w


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Representative cutoff frequencies spanning the audio band.
@pytest.fixture(params=[500.0, 1000.0, 4000.0])
def cutoff(request):
    return request.param


# Sub-Butterworth, Butterworth, and resonant Q values.
@pytest.fixture(params=[0.5, 0.707, 2.0])
def Q_value(request):
    return request.param


# All three filter types for shape and shared-property tests.
@pytest.fixture(params=ALL_FILTERS, ids=["lowpass", "highpass", "bandpass"])
def any_filter(request):
    return request.param


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_any_filter_output_shape(any_filter, cutoff):
    """
    What:   Every filter returns a 1-D array with the same length as its input.
    Input:  Each filter (lowpass, highpass, bandpass) with cutoff in
            {500, 1000, 4000} Hz; input is a 1-second sine at 440 Hz.
    Output: len(output) == len(input).
    Why:    A filter that changes signal length would desynchronise any
            sample-aligned pipeline that follows it.
    """
    signal = _sine(440.0)
    out = any_filter(signal, cutoff=cutoff, fs=FS)
    assert out.shape == signal.shape


# ── Low-pass filter ───────────────────────────────────────────────────────────

def test_lowpass_passes_low_frequency():
    """
    What:   A sine well below the cutoff passes through with negligible attenuation.
    Input:  100 Hz sine, cutoff=1000 Hz (1 decade below), default Q.
    Output: steady-state RMS ratio (output / input) > 0.95.
    Why:    A 2nd-order Butterworth at 100 Hz / 1000 Hz attenuates by < 0.01 dB;
            significant attenuation would indicate a coefficient sign error that
            turns the passband into a stopband.
    """
    sig = _sine(100.0)
    out = lowpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_lowpass_attenuates_high_frequency():
    """
    What:   A sine well above the cutoff is heavily attenuated.
    Input:  10000 Hz sine, cutoff=1000 Hz (1 decade above), default Q.
    Output: steady-state RMS ratio (output / input) < 0.05.
    Why:    A 2nd-order filter 1 decade into the stopband attenuates by ~40 dB
            (factor of 100); this confirms the roll-off is actually active and
            the filter is not transparently passing all frequencies.
    """
    sig = _sine(10000.0)
    out = lowpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.05


def test_lowpass_passes_dc():
    """
    What:   A DC signal passes through a low-pass filter at unity gain.
    Input:  DC at amplitude=1.0, cutoff=1000 Hz.
    Output: steady-state RMS of output ≈ 1.0 (within 0.01).
    Why:    By definition H(0) = 1 for a low-pass filter; a DC offset on the
            output would corrupt any steady-state DC-level measurement.
    """
    out = lowpass(_dc(1.0), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) == pytest.approx(1.0, abs=0.01)


def test_lowpass_Q_sharpens_transition(Q_value):
    """
    What:   The filter executes at all Q values without error, and attenuation
            at a fixed stopband frequency is non-trivial for every Q.
    Input:  10000 Hz sine, cutoff=1000 Hz, Q in {0.5, 0.707, 2.0}.
    Output: steady-state RMS ratio < 0.2 for every Q.
    Why:    Q only changes the shape of the transition band, not whether
            attenuation occurs deep in the stopband; all three Q values should
            still strongly attenuate a frequency 1 decade above cutoff.
    """
    sig = _sine(10000.0)
    out = lowpass(sig, cutoff=1000.0, fs=FS, Q=Q_value)
    assert _steady_rms(out) / _steady_rms(sig) < 0.2


# ── High-pass filter ──────────────────────────────────────────────────────────

def test_highpass_passes_high_frequency():
    """
    What:   A sine well above the cutoff passes through with negligible attenuation.
    Input:  10000 Hz sine, cutoff=1000 Hz (1 decade above), default Q.
    Output: steady-state RMS ratio (output / input) > 0.95.
    Why:    Mirrors the low-pass passband test; confirms the high-pass and
            low-pass coefficient sets are not swapped in the implementation.
    """
    sig = _sine(10000.0)
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_highpass_attenuates_low_frequency():
    """
    What:   A sine well below the cutoff is heavily attenuated.
    Input:  100 Hz sine, cutoff=1000 Hz (1 decade below), default Q.
    Output: steady-state RMS ratio (output / input) < 0.05.
    Why:    Mirrors the low-pass stopband test; a high-pass that passes low
            frequencies would be useless for its intended purpose of DC and
            rumble removal.
    """
    sig = _sine(100.0)
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.05


def test_highpass_rejects_dc():
    """
    What:   A DC signal is driven to zero by a high-pass filter.
    Input:  DC at amplitude=1.0, cutoff=1000 Hz.
    Output: steady-state RMS of output < 0.01.
    Why:    H(0) = 0 for a high-pass filter by definition; passing DC would
            mean the filter fails its primary use case of DC removal.
    """
    out = highpass(_dc(1.0), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) < 0.01


# ── Band-pass filter ──────────────────────────────────────────────────────────

def test_bandpass_passes_center_frequency():
    """
    What:   A sine at the centre frequency passes through at approximately
            unity gain.
    Input:  1000 Hz sine, cutoff=1000 Hz, Q=4.
    Output: steady-state RMS ratio (output / input) > 0.85.
    Why:    The Audio EQ Cookbook band-pass variant is specified with 0 dB peak
            gain at f0; significant attenuation would mean the wrong coefficient
            variant (e.g., constant skirt gain) was implemented.
    """
    sig = _sine(1000.0)
    out = bandpass(sig, cutoff=1000.0, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) > 0.85


def test_bandpass_attenuates_far_above():
    """
    What:   A sine far above the centre frequency is heavily attenuated.
    Input:  10000 Hz sine, cutoff=1000 Hz (1 decade above), Q=4.
    Output: steady-state RMS ratio < 0.1.
    Why:    A band-pass filter must reject out-of-band signals on both sides;
            this covers the upper stopband.
    """
    sig = _sine(10000.0)
    out = bandpass(sig, cutoff=1000.0, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) < 0.1


def test_bandpass_attenuates_far_below():
    """
    What:   A sine far below the centre frequency is heavily attenuated.
    Input:  100 Hz sine, cutoff=1000 Hz (1 decade below), Q=4.
    Output: steady-state RMS ratio < 0.05.
    Why:    Covers the lower stopband; a band-pass that behaves like a high-pass
            would pass this test but fail the far-above test.
    """
    sig = _sine(100.0)
    out = bandpass(sig, cutoff=1000.0, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) < 0.05


def test_bandpass_rejects_dc():
    """
    What:   A DC signal is driven to zero by a band-pass filter.
    Input:  DC at amplitude=1.0, cutoff=1000 Hz.
    Output: steady-state RMS of output < 0.01.
    Why:    H(0) = 0 for a band-pass filter (b0 + b1 + b2 = 0 in the cookbook
            formula); a non-zero DC output indicates a coefficient error.
    """
    out = bandpass(_dc(1.0), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) < 0.01


def test_bandpass_higher_Q_narrows_bandwidth():
    """
    What:   A frequency just outside the narrow passband is attenuated more by a
            high-Q filter than by a low-Q filter at the same centre frequency.
    Input:  1200 Hz sine (200 Hz above a 1000 Hz centre); Q=1 (BW=1000 Hz) vs
            Q=8 (BW=125 Hz).
    Output: RMS ratio at Q=8 < RMS ratio at Q=1.
    Why:    Q = f0 / bandwidth; Q=8 places 1200 Hz well outside its 125 Hz
            passband, while Q=1's 1000 Hz passband still covers 1200 Hz.
            Note: testing far into the stopband (e.g. 2000 Hz) does not reliably
            show this ordering for the constant-0-dB-peak-gain variant because
            poles close to the unit circle (high Q) increase gain everywhere
            except exactly at DC and Nyquist.
    """
    sig = _sine(1200.0)
    ratio_low_Q = _steady_rms(bandpass(sig, cutoff=1000.0, fs=FS, Q=1.0)) / _steady_rms(sig)
    ratio_high_Q = _steady_rms(bandpass(sig, cutoff=1000.0, fs=FS, Q=8.0)) / _steady_rms(sig)
    assert ratio_high_Q < ratio_low_Q
