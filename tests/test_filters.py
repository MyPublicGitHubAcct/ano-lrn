import numpy as np
import pytest
from python.filters import allpass, bandpass, highpass, highshelf, lowpass, lowshelf, notch
from python.generators import (
    generate_dc,
    generate_half_nyquist,
    generate_impulse,
    generate_nyquist,
    generate_quarter_nyquist,
    generate_sine,
)

FS = 44100
DURATION = 1.0
# Skip first 50 ms so transient decay does not skew RMS measurements.
SKIP = int(0.05 * FS)

ALL_FILTERS = [lowpass, highpass, bandpass, notch, allpass, lowshelf, highshelf]


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
@pytest.fixture(
    params=ALL_FILTERS,
    ids=["lowpass", "highpass", "bandpass", "notch", "allpass", "lowshelf", "highshelf"],
)
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


# ── Nyquist-frequency signal tests ────────────────────────────────────────────

def _nyquist_sig() -> np.ndarray:
    _, w = generate_nyquist(fs=FS, duration=DURATION)
    return w


def _half_nyquist_sig() -> np.ndarray:
    _, w = generate_half_nyquist(fs=FS, duration=DURATION)
    return w


def _quarter_nyquist_sig() -> np.ndarray:
    _, w = generate_quarter_nyquist(fs=FS, duration=DURATION)
    return w


# Nyquist (fs/2 = 22050 Hz)

def test_lowpass_rejects_nyquist():
    """
    What:   A low-pass filter produces near-zero output for the Nyquist signal.
    Input:  Nyquist signal (fs/2 = 22050 Hz), cutoff=1000 Hz, default Q.
    Output: steady-state RMS < 1e-6.
    Why:    The LP biquad has a structural zero at z = −1 (Nyquist): the three
            b-coefficients sum to zero at ω=π, so any input at exactly fs/2 is
            cancelled by the FIR part of the filter. This confirms the zero is
            present and the filter is not accidentally passing the highest
            representable frequency.
    """
    out = lowpass(_nyquist_sig(), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) < 1e-6


def test_highpass_passes_nyquist():
    """
    What:   A high-pass filter passes the Nyquist signal at unity gain.
    Input:  Nyquist signal (fs/2 = 22050 Hz), cutoff=1000 Hz, default Q.
    Output: steady-state RMS ratio (output / input) > 0.99.
    Why:    H(π) = 1 exactly for the HP biquad: the numerator evaluates to
            2(1 + cos ω₀) and the denominator to the same value at z = −1,
            giving unity gain regardless of cutoff or Q. A ratio below 1 would
            indicate a coefficient error that misplaces the gain at the band edge.
    """
    sig = _nyquist_sig()
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.99


def test_bandpass_rejects_nyquist():
    """
    What:   A band-pass filter produces near-zero output for the Nyquist signal.
    Input:  Nyquist signal (fs/2 = 22050 Hz), cutoff=1000 Hz, default Q.
    Output: steady-state RMS < 1e-6.
    Why:    The BP biquad also has a structural zero at Nyquist (b0 + b2 = 0 at
            ω=π). A band-pass must reject both DC and Nyquist; a non-zero output
            would mean the Nyquist component is leaking into the passband.
    """
    out = bandpass(_nyquist_sig(), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) < 1e-6


# Half-Nyquist (fs/4 = 11025 Hz)

def test_lowpass_attenuates_half_nyquist():
    """
    What:   A low-pass filter heavily attenuates a signal at fs/4.
    Input:  Half-Nyquist signal (11025 Hz), cutoff=1000 Hz, default Q.
    Output: steady-state RMS ratio (output / input) < 0.02.
    Why:    11025 Hz is 11× above the cutoff; a 2nd-order rolloff gives
            attenuation of (11)² ≈ 100× (≈40 dB). A ratio above 0.02 would
            indicate the stopband rolloff is not working.
    """
    sig = _half_nyquist_sig()
    out = lowpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.02


def test_highpass_passes_half_nyquist():
    """
    What:   A high-pass filter passes a signal at fs/4 with negligible attenuation.
    Input:  Half-Nyquist signal (11025 Hz), cutoff=1000 Hz, default Q.
    Output: steady-state RMS ratio (output / input) > 0.95.
    Why:    11025 Hz is well into the passband of a 1000 Hz high-pass; significant
            attenuation would indicate a passband gain error or coefficient swap.
    """
    sig = _half_nyquist_sig()
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_bandpass_passes_half_nyquist():
    """
    What:   A band-pass filter centred at fs/4 passes the half-Nyquist signal.
    Input:  Half-Nyquist signal (11025 Hz), cutoff=fs/4=11025 Hz, Q=4.
    Output: steady-state RMS ratio (output / input) > 0.85.
    Why:    The cookbook BP has 0 dB peak gain at f0; feeding a signal at exactly
            the centre frequency is the tightest possible check that the peak is
            placed correctly. A ratio below 0.85 would mean the centre frequency
            is shifted or the gain is wrong.
    """
    sig = _half_nyquist_sig()
    out = bandpass(sig, cutoff=FS / 4, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) > 0.85


# Quarter-Nyquist (fs/8 = 5512.5 Hz)

def test_lowpass_attenuates_quarter_nyquist():
    """
    What:   A low-pass filter attenuates a signal at fs/8.
    Input:  Quarter-Nyquist signal (5512.5 Hz), cutoff=1000 Hz, default Q.
    Output: steady-state RMS ratio (output / input) < 0.1.
    Why:    5512.5 Hz is 5.5× above the cutoff; a 2nd-order rolloff gives
            attenuation of (5.5)² ≈ 30× (≈30 dB), so the ratio should be well
            below 0.1. This tests the stopband a full octave below the
            half-Nyquist case, confirming rolloff across the upper spectrum.
    """
    sig = _quarter_nyquist_sig()
    out = lowpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) < 0.1


def test_highpass_passes_quarter_nyquist():
    """
    What:   A high-pass filter passes a signal at fs/8 with negligible attenuation.
    Input:  Quarter-Nyquist signal (5512.5 Hz), cutoff=1000 Hz, default Q.
    Output: steady-state RMS ratio (output / input) > 0.95.
    Why:    5512.5 Hz is well into the passband of a 1000 Hz high-pass; this
            complements the half-Nyquist passband test by confirming flat gain
            one octave lower.
    """
    sig = _quarter_nyquist_sig()
    out = highpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_bandpass_passes_quarter_nyquist():
    """
    What:   A band-pass filter centred at fs/8 passes the quarter-Nyquist signal.
    Input:  Quarter-Nyquist signal (5512.5 Hz), cutoff=fs/8=5512.5 Hz, Q=4.
    Output: steady-state RMS ratio (output / input) > 0.85.
    Why:    Same rationale as the half-Nyquist centre-frequency test, but at a
            lower frequency. Confirms the BP centre-frequency placement is correct
            across different parts of the spectrum.
    """
    sig = _quarter_nyquist_sig()
    out = bandpass(sig, cutoff=FS / 8, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) > 0.85


# ── Notch filter ──────────────────────────────────────────────────────────────

def test_notch_passes_below_cutoff():
    """
    What:   A sine well below the cutoff passes through with negligible attenuation.
    Input:  100 Hz sine, cutoff=1000 Hz, default Q.
    Output: steady-state RMS ratio (output / input) > 0.95.
    Why:    The notch only removes energy at the centre frequency; a signal a decade
            below the notch should be essentially unaffected.
    """
    sig = _sine(100.0)
    out = notch(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_notch_passes_above_cutoff():
    """
    What:   A sine well above the cutoff passes through with negligible attenuation.
    Input:  10000 Hz sine, cutoff=1000 Hz, default Q.
    Output: steady-state RMS ratio (output / input) > 0.95.
    Why:    Mirrors the below-cutoff test; the notch must reject only the band
            around f₀ and pass everything else, including signals above it.
    """
    sig = _sine(10000.0)
    out = notch(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) > 0.95


def test_notch_attenuates_center():
    """
    What:   A sine at the centre frequency is driven to near-zero.
    Input:  1000 Hz sine, cutoff=1000 Hz, Q=4.
    Output: steady-state RMS ratio (output / input) < 0.01.
    Why:    The notch biquad has a zero exactly on the unit circle at ω₀, so the
            steady-state response to a sinusoid at that frequency is zero. The
            transient at Q=4 decays in ≈1 ms, well before the 50 ms skip window.
    """
    sig = _sine(1000.0)
    out = notch(sig, cutoff=1000.0, fs=FS, Q=4.0)
    assert _steady_rms(out) / _steady_rms(sig) < 0.01


def test_notch_passes_dc():
    """
    What:   A DC signal passes through a notch filter at unity gain.
    Input:  DC at amplitude=1.0, cutoff=1000 Hz.
    Output: steady-state RMS of output ≈ 1.0 (within 0.01).
    Why:    H(0) = 1 for the notch: b0+b1+b2 = a0+a1+a2 (zeros are not at DC).
            A notch that attenuated DC would be removing the wrong frequency.
    """
    out = notch(_dc(1.0), cutoff=1000.0, fs=FS)
    assert _steady_rms(out) == pytest.approx(1.0, abs=0.01)


def test_notch_higher_Q_narrows_notch():
    """
    What:   A frequency close to but outside a narrow notch is attenuated less
            than the same frequency through a wide notch.
    Input:  1200 Hz sine (200 Hz above a 1000 Hz notch centre); Q=1 vs Q=8.
    Output: RMS ratio at Q=8 > RMS ratio at Q=1.
    Why:    Q = f₀ / bandwidth; Q=1 gives BW=1000 Hz, placing 1200 Hz inside the
            notch. Q=8 gives BW=125 Hz, placing 1200 Hz well outside it. A high-Q
            notch is more selective and passes adjacent frequencies with less loss.
    """
    sig = _sine(1200.0)
    ratio_low_Q = _steady_rms(notch(sig, cutoff=1000.0, fs=FS, Q=1.0)) / _steady_rms(sig)
    ratio_high_Q = _steady_rms(notch(sig, cutoff=1000.0, fs=FS, Q=8.0)) / _steady_rms(sig)
    assert ratio_high_Q > ratio_low_Q


# ── All-pass filter ───────────────────────────────────────────────────────────

def test_allpass_preserves_rms_at_low_freq():
    """
    What:   A sine well below the cutoff passes through at unity gain.
    Input:  100 Hz sine, cutoff=1000 Hz, default Q.
    Output: steady-state RMS ratio (output / input) ≈ 1.0 (within 0.02).
    Why:    |H(ω)| = 1 for all ω by definition; a ratio other than 1 would mean
            the coefficient normalisation introduces unintended gain or loss.
    """
    sig = _sine(100.0)
    out = allpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(1.0, abs=0.02)


def test_allpass_preserves_rms_at_high_freq():
    """
    What:   A sine well above the cutoff passes through at unity gain.
    Input:  10000 Hz sine, cutoff=1000 Hz, default Q.
    Output: steady-state RMS ratio (output / input) ≈ 1.0 (within 0.02).
    Why:    Complements the low-frequency case; |H(ω)| = 1 must hold across the
            full band, not just near DC.
    """
    sig = _sine(10000.0)
    out = allpass(sig, cutoff=1000.0, fs=FS)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(1.0, abs=0.02)


def test_allpass_unity_magnitude():
    """
    What:   The FFT of the all-pass impulse response has magnitude ≈ 1 at every bin.
    Input:  Impulse, cutoff=1000 Hz, default Q.
    Output: |FFT(h)[k]| ≈ 1.0 for all bins k > 0 (within 0.01).
    Why:    The impulse response IS the system's frequency response kernel; its FFT
            gives the full frequency response in one shot. A flat magnitude confirms
            the all-pass property directly, independent of any specific test tone.
    """
    _, imp = generate_impulse(fs=FS, duration=DURATION)
    h = allpass(imp, cutoff=1000.0, fs=FS)
    mag = np.abs(np.fft.rfft(h))
    assert np.allclose(mag[1:], 1.0, atol=0.01)


# ── Low-shelf filter ──────────────────────────────────────────────────────────

def test_lowshelf_boosts_dc():
    """
    What:   A low-shelf with positive gain raises DC to the expected amplitude.
    Input:  DC at amplitude=1.0, cutoff=1000 Hz, gain_db=6.
    Output: steady-state RMS ≈ 10^(6/20) ≈ 2.0 (within 0.05).
    Why:    H(0) = A² = 10^(gain_db/20) exactly for the low-shelf. A result
            significantly different from the expected linear gain would indicate
            the A or α coefficients are computed incorrectly.
    """
    expected = 10 ** (6.0 / 20)
    out = lowshelf(_dc(1.0), cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) == pytest.approx(expected, abs=0.05)


def test_lowshelf_cuts_dc():
    """
    What:   A low-shelf with negative gain reduces DC to the expected amplitude.
    Input:  DC at amplitude=1.0, cutoff=1000 Hz, gain_db=−6.
    Output: steady-state RMS ≈ 10^(−6/20) ≈ 0.5 (within 0.05).
    Why:    Confirms both boost (positive gain_db) and cut (negative gain_db)
            directions are implemented correctly — they require different A values
            and the formula must handle A < 1 without sign errors.
    """
    expected = 10 ** (-6.0 / 20)
    out = lowshelf(_dc(1.0), cutoff=1000.0, fs=FS, gain_db=-6.0)
    assert _steady_rms(out) == pytest.approx(expected, abs=0.05)


def test_lowshelf_unity_at_nyquist():
    """
    What:   A low-shelf leaves the Nyquist signal at unity gain regardless of
            gain_db.
    Input:  Nyquist signal (fs/2), cutoff=1000 Hz, gain_db=6.
    Output: steady-state RMS ratio (output / input) ≈ 1.0 (within 0.01).
    Why:    H(π) = 1 exactly for the low-shelf: numerator and denominator both
            evaluate to 4A(1 + cos_w0) at z = −1, cancelling to unity. The shelf
            only affects the low-frequency end; Nyquist must always pass unchanged.
    """
    sig = _nyquist_sig()
    out = lowshelf(sig, cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(1.0, abs=0.01)


def test_lowshelf_boosts_low_freq():
    """
    What:   A low-shelf with positive gain boosts a sine well below the cutoff.
    Input:  100 Hz sine (1 decade below cutoff), cutoff=1000 Hz, gain_db=6.
    Output: steady-state RMS ratio (output / input) > 1.5.
    Why:    100 Hz is deep in the shelf region; the gain there should be close to
            the DC gain of A² ≈ 2.0. A ratio below 1.5 would mean the shelf
            transition is placed much lower than the specified cutoff.
    """
    sig = _sine(100.0)
    out = lowshelf(sig, cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) / _steady_rms(sig) > 1.5


# ── High-shelf filter ─────────────────────────────────────────────────────────

def test_highshelf_unity_at_dc():
    """
    What:   A high-shelf leaves DC at unity gain regardless of gain_db.
    Input:  DC at amplitude=1.0, cutoff=1000 Hz, gain_db=6.
    Output: steady-state RMS of output ≈ 1.0 (within 0.01).
    Why:    H(0) = 1 exactly for the high-shelf: numerator and denominator both
            evaluate to 4A(1 − cos_w0) at z = 1, cancelling to unity. The shelf
            only affects the high-frequency end; DC must always pass unchanged.
    """
    out = highshelf(_dc(1.0), cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) == pytest.approx(1.0, abs=0.01)


def test_highshelf_boosts_nyquist():
    """
    What:   A high-shelf with positive gain raises the Nyquist signal to the
            expected amplitude.
    Input:  Nyquist signal (fs/2), cutoff=1000 Hz, gain_db=6.
    Output: steady-state RMS ratio (output / input) ≈ 10^(6/20) ≈ 2.0
            (within 0.05).
    Why:    H(π) = A² = 10^(gain_db/20) exactly for the high-shelf; Nyquist is the
            design frequency for the gain target, making it the tightest possible
            check of the shelf amplitude.
    """
    expected = 10 ** (6.0 / 20)
    sig = _nyquist_sig()
    out = highshelf(sig, cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(expected, abs=0.05)


def test_highshelf_cuts_nyquist():
    """
    What:   A high-shelf with negative gain reduces the Nyquist signal to the
            expected amplitude.
    Input:  Nyquist signal (fs/2), cutoff=1000 Hz, gain_db=−6.
    Output: steady-state RMS ratio (output / input) ≈ 10^(−6/20) ≈ 0.5
            (within 0.05).
    Why:    Mirrors the boost test for the cut direction. H(π) = A² still holds;
            with A < 1 the gain is a reduction.
    """
    expected = 10 ** (-6.0 / 20)
    sig = _nyquist_sig()
    out = highshelf(sig, cutoff=1000.0, fs=FS, gain_db=-6.0)
    assert _steady_rms(out) / _steady_rms(sig) == pytest.approx(expected, abs=0.05)


def test_highshelf_boosts_high_freq():
    """
    What:   A high-shelf with positive gain boosts a sine well above the cutoff.
    Input:  10000 Hz sine (1 decade above cutoff), cutoff=1000 Hz, gain_db=6.
    Output: steady-state RMS ratio (output / input) > 1.5.
    Why:    10000 Hz is deep in the shelf region; the gain there should be close to
            the Nyquist gain of A² ≈ 2.0. A ratio below 1.5 would mean the shelf
            transition is placed much higher than the specified cutoff.
    """
    sig = _sine(10000.0)
    out = highshelf(sig, cutoff=1000.0, fs=FS, gain_db=6.0)
    assert _steady_rms(out) / _steady_rms(sig) > 1.5
