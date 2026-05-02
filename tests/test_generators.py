import numpy as np
import pytest
from ano_lrn.generators import (
    generate_chirp,
    generate_dc,
    generate_impulse,
    generate_multi_tone,
    generate_pink_noise,
    generate_sawtooth,
    generate_sine,
    generate_square,
    generate_step,
    generate_triangle,
    generate_white_noise,
)

FS = 44100
DURATION = 1.0
N = int(FS * DURATION)

PERIODIC_GENERATORS = [generate_sine, generate_square, generate_sawtooth, generate_triangle]
SEEDED_GENERATORS = [generate_white_noise, generate_pink_noise]


def _dominant_frequency(wave: np.ndarray, fs: int) -> float:
    spectrum = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(len(wave), d=1.0 / fs)
    return float(freqs[np.argmax(spectrum)])


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Each periodic generator: sine, square, sawtooth, triangle.
@pytest.fixture(params=PERIODIC_GENERATORS, ids=lambda f: f.__name__)
def periodic_gen(request):
    return request.param


# Each noise generator that accepts a seed: white, pink.
@pytest.fixture(params=SEEDED_GENERATORS, ids=lambda f: f.__name__)
def seeded_gen(request):
    return request.param


# Sub-unity, unity, and above-unity amplitude to cover scaling behaviour.
@pytest.fixture(params=[0.5, 1.0, 2.0])
def amplitude(request):
    return request.param


# Bass, mid, and upper-mid frequencies within the audio band.
@pytest.fixture(params=[110.0, 440.0, 880.0])
def freq(request):
    return request.param


# Two non-default sample rates paired with their durations.
@pytest.fixture(params=[(22050, 0.5), (48000, 2.0)], ids=["22050Hz-0.5s", "48000Hz-2.0s"])
def fs_and_duration(request):
    return request.param


# Three non-zero phase offsets: 45°, 90°, and 180°.
@pytest.fixture(params=[np.pi / 4, np.pi / 2, np.pi])
def phase(request):
    return request.param


# Low, balanced, and high duty cycles.
@pytest.fixture(params=[0.25, 0.5, 0.75])
def duty(request):
    return request.param


# Three distinct seeds to verify reproducibility across multiple RNG states.
@pytest.fixture(params=[0, 42, 123])
def seed(request):
    return request.param


# Adjacent seed pairs: each pair should produce different output.
@pytest.fixture(params=[(0, 1), (42, 43)])
def seed_pair(request):
    return request.param


# Delays spanning the signal: start, early, mid, and late.
@pytest.fixture(params=[0.0, 0.1, 0.5, 0.9])
def delay(request):
    return request.param


# Onsets spanning the signal: start, quarter, half, and three-quarters.
@pytest.fixture(params=[0.0, 0.25, 0.5, 0.75])
def onset(request):
    return request.param


# Both supported sweep methods.
@pytest.fixture(params=["linear", "logarithmic"])
def chirp_method(request):
    return request.param


# Negative, zero, and positive amplitudes to verify no silent rectification.
@pytest.fixture(params=[-1.0, 0.0, 0.5, 1.0, 2.0])
def dc_amplitude(request):
    return request.param


# Single tone, two-tone, and three-tone mixes at well-separated frequencies.
@pytest.fixture(
    params=[[440.0], [220.0, 880.0], [110.0, 440.0, 1760.0]],
    ids=["1-tone", "2-tone", "3-tone"],
)
def multi_tone_freqs(request):
    return request.param


# ── Shared: periodic generators ───────────────────────────────────────────────

def test_periodic_output_shapes(periodic_gen):
    """
    What:   Each periodic generator returns a time axis and a signal array.
    Input:  Default parameters (440 Hz, fs=44100, duration=1.0).
    Output: Both t and w are 1-D arrays of length N = fs * duration.
    Why:    All downstream consumers expect this shape contract; a wrong length
            would silently corrupt any analysis or plotting that follows.
    """
    t, w = periodic_gen()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_periodic_amplitude_range(periodic_gen, amplitude):
    """
    What:   The signal stays within the declared amplitude envelope.
    Input:  Each periodic generator called with amplitude in {0.5, 1.0, 2.0}.
    Output: Every sample satisfies -amplitude <= w[n] <= amplitude.
    Why:    Amplitude is the gain contract; exceeding it would clip downstream
            processing stages that trust the declared range.
    """
    _, w = periodic_gen(amplitude=amplitude)
    assert np.max(w) <= amplitude + 1e-9
    assert np.min(w) >= -amplitude - 1e-9


def test_periodic_dominant_frequency(periodic_gen, freq):
    """
    What:   The FFT peak of the output lands at the requested frequency.
    Input:  Each periodic generator called with freq in {110, 440, 880} Hz.
    Output: The bin with the highest magnitude is within 1 Hz of freq.
    Why:    The fundamental must match the requested pitch; a wrong frequency
            makes the signal useless as a calibrated audio test tone.
    """
    _, w = periodic_gen(freq=freq)
    assert _dominant_frequency(w, FS) == pytest.approx(freq, abs=1.0)


def test_periodic_sample_rate(periodic_gen, fs_and_duration):
    """
    What:   Output length scales correctly with non-default sample rate and duration.
    Input:  Each periodic generator with (fs=22050, duration=0.5) and (fs=48000, duration=2.0).
    Output: len(t) == len(w) == int(fs * duration) for each combination.
    Why:    Generators must work at any standard sample rate; a fixed internal
            constant would silently produce wrong-length buffers at other rates.
    """
    fs, duration = fs_and_duration
    t, w = periodic_gen(fs=fs, duration=duration)
    assert t.shape == w.shape == (int(fs * duration),)


# ── generate_sine ─────────────────────────────────────────────────────────────

def test_sine_phase_shifts_output(phase):
    """
    What:   A non-zero phase produces a waveform distinct from the zero-phase version.
    Input:  phase in {π/4, π/2, π}; both calls use default freq and amplitude.
    Output: The two arrays are not element-wise close.
    Why:    Phase offset is the only parameter that distinguishes a cosine from a
            sine; if it had no effect, phase-sensitive DSP tests would be invalid.
    """
    _, w0 = generate_sine(phase=0.0)
    _, w1 = generate_sine(phase=phase)
    assert not np.allclose(w0, w1)


# ── generate_square ───────────────────────────────────────────────────────────

def test_square_values_are_binary(amplitude):
    """
    What:   Every sample of the square wave is exactly +amplitude or -amplitude.
    Input:  amplitude in {0.5, 1.0, 2.0}; default freq and duration.
    Output: np.unique(w) == {-amplitude, +amplitude}.
    Why:    A square wave has no intermediate values by definition; any ramp or
            transition sample would indicate a bug in the np.where threshold logic.
    """
    _, w = generate_square(amplitude=amplitude)
    assert np.all((w == amplitude) | (w == -amplitude))


def test_square_duty_cycle(duty):
    """
    What:   The fraction of positive samples matches the requested duty cycle.
    Input:  duty in {0.25, 0.5, 0.75}; default freq and amplitude.
    Output: (count of w > 0) / N is within 2% of duty.
    Why:    Duty cycle directly sets harmonic content; a wrong ratio produces a
            different waveform even if the frequency is correct.
    """
    _, w = generate_square(duty=duty)
    assert np.sum(w > 0) / N == pytest.approx(duty, abs=0.02)


# ── generate_triangle ─────────────────────────────────────────────────────────

def test_triangle_peaks_reach_amplitude(amplitude):
    """
    What:   The triangle wave actually reaches its declared peak values.
    Input:  freq=100 Hz (dense sampling for accurate peak capture),
            amplitude in {0.5, 1.0, 2.0}.
    Output: max(w) ≈ +amplitude and min(w) ≈ -amplitude within 0.01.
    Why:    The amplitude range test only checks that the signal doesn't exceed
            the bound; this confirms the wave uses its full range and isn't
            attenuated by a formula error.
    """
    _, w = generate_triangle(freq=100.0, amplitude=amplitude)
    assert np.max(w) == pytest.approx(amplitude, abs=0.01)
    assert np.min(w) == pytest.approx(-amplitude, abs=0.01)


# ── Shared: seeded noise generators ──────────────────────────────────────────

def test_noise_output_shapes(seeded_gen):
    """
    What:   Each noise generator returns arrays of the expected length.
    Input:  Default parameters (fs=44100, duration=1.0).
    Output: t.shape == w.shape == (N,).
    Why:    Same shape contract as the periodic generators; noise buffers must
            be the same length as any signal they are mixed or convolved with.
    """
    t, w = seeded_gen()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_noise_amplitude_range(seeded_gen, amplitude):
    """
    What:   Noise output is bounded by the declared amplitude.
    Input:  Each noise generator with amplitude in {0.5, 1.0, 2.0} and seed=0.
    Output: max(|w|) <= amplitude.
    Why:    Amplitude scaling must cap the output; an unbounded noise burst
            would saturate any downstream gain stage under test.
    """
    _, w = seeded_gen(amplitude=amplitude, seed=0)
    assert np.max(np.abs(w)) <= amplitude + 1e-9


def test_noise_reproducible_with_seed(seeded_gen, seed):
    """
    What:   Two calls with the same seed return identical arrays.
    Input:  Each noise generator with seed in {0, 42, 123}.
    Output: np.allclose(w1, w2) is True.
    Why:    Reproducibility is required for deterministic regression tests;
            if the seed were ignored, tests would be non-deterministic.
    """
    _, w1 = seeded_gen(seed=seed)
    _, w2 = seeded_gen(seed=seed)
    assert np.allclose(w1, w2)


def test_noise_different_seeds_differ(seeded_gen, seed_pair):
    """
    What:   Two calls with different seeds return distinct arrays.
    Input:  Each noise generator with adjacent seed pairs {(0,1), (42,43)}.
    Output: np.allclose(w1, w2) is False.
    Why:    Confirms the seed is actually consumed by the RNG; a generator that
            ignores the seed would pass the reproducibility test but produce
            only one sequence regardless of the seed value.
    """
    seed_a, seed_b = seed_pair
    _, w1 = seeded_gen(seed=seed_a)
    _, w2 = seeded_gen(seed=seed_b)
    assert not np.allclose(w1, w2)


# ── generate_pink_noise ───────────────────────────────────────────────────────

def test_pink_noise_zero_dc():
    """
    What:   Pink noise has no DC offset.
    Input:  seed=0, default fs and duration.
    Output: |mean(w)| < 0.01.
    Why:    The FFT shaping step explicitly zeros bin 0 (the DC component).
            A nonzero mean would bias any filter under test toward a false
            steady-state offset.
    """
    _, w = generate_pink_noise(seed=0)
    assert abs(np.mean(w)) < 0.01


# ── generate_impulse ──────────────────────────────────────────────────────────

def test_impulse_output_shapes():
    """
    What:   The impulse generator returns arrays of the expected length.
    Input:  Default parameters (delay=0.0, fs=44100, duration=1.0).
    Output: t.shape == w.shape == (N,).
    Why:    Same shape contract as all other generators.
    """
    t, w = generate_impulse()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_impulse_single_nonzero_sample():
    """
    What:   Exactly one sample in the output is non-zero.
    Input:  Default parameters.
    Output: np.sum(w != 0) == 1.
    Why:    A Dirac delta has energy at a single point; multiple nonzero samples
            would make it a pulse rather than an impulse, invalidating any
            impulse-response measurement that uses it.
    """
    _, w = generate_impulse()
    assert np.sum(w != 0) == 1


def test_impulse_delay(delay):
    """
    What:   The nonzero sample lands at the sample index corresponding to delay.
    Input:  delay in {0.0, 0.1, 0.5, 0.9} seconds; default amplitude=1.0.
    Output: w[int(delay * FS)] == 1.0 and all other samples are zero.
    Why:    The delay parameter must convert precisely to a sample index; a
            rounding error here shifts the impulse response in time and
            corrupts latency measurements.
    """
    _, w = generate_impulse(delay=delay)
    idx = int(delay * FS)
    assert w[idx] == 1.0
    assert np.sum(w != 0) == 1


def test_impulse_amplitude(amplitude):
    """
    What:   The peak of the impulse equals the requested amplitude.
    Input:  amplitude in {0.5, 1.0, 2.0}; default delay=0.0.
    Output: max(w) == amplitude and exactly one sample is nonzero.
    Why:    Amplitude scales the impulse gain; a wrong value would scale the
            entire measured impulse response by the wrong factor.
    """
    _, w = generate_impulse(amplitude=amplitude)
    assert np.max(w) == pytest.approx(amplitude)
    assert np.sum(w != 0) == 1


# ── generate_step ─────────────────────────────────────────────────────────────

def test_step_output_shapes():
    """
    What:   The step generator returns arrays of the expected length.
    Input:  Default parameters (onset=0.0, fs=44100, duration=1.0).
    Output: t.shape == w.shape == (N,).
    Why:    Same shape contract as all other generators.
    """
    t, w = generate_step()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_step_onset(onset):
    """
    What:   All samples before the onset index are 0; all samples from the
            onset index onward equal amplitude.
    Input:  onset in {0.0, 0.25, 0.5, 0.75} seconds; amplitude=1.0.
    Output: w[:int(onset*FS)] == 0 and w[int(onset*FS):] == 1.0.
    Why:    The onset time must map exactly to a sample boundary; an off-by-one
            error would place the transition at the wrong sample, shifting any
            transient or DC measurement by one sample.
    """
    _, w = generate_step(onset=onset, amplitude=1.0)
    idx = int(onset * FS)
    assert np.all(w[:idx] == 0.0)
    assert np.all(w[idx:] == pytest.approx(1.0))


def test_step_amplitude(amplitude):
    """
    What:   The steady-state value of the step equals the requested amplitude.
    Input:  onset=0.0 (entire signal is active), amplitude in {0.5, 1.0, 2.0}.
    Output: Every sample equals amplitude.
    Why:    Amplitude sets the DC level under test; a wrong value would give
            incorrect steady-state conditions to any DC-rejection filter.
    """
    _, w = generate_step(onset=0.0, amplitude=amplitude)
    assert np.all(w == pytest.approx(amplitude))


# ── generate_chirp ────────────────────────────────────────────────────────────

def test_chirp_output_shapes():
    """
    What:   The chirp generator returns arrays of the expected length.
    Input:  Default parameters (f_start=20, f_end=20000, fs=44100, duration=1.0).
    Output: t.shape == w.shape == (N,).
    Why:    Same shape contract as all other generators.
    """
    t, w = generate_chirp()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_chirp_amplitude_range(amplitude):
    """
    What:   The chirp signal stays within the declared amplitude envelope.
    Input:  amplitude in {0.5, 1.0, 2.0}; default sweep range and duration.
    Output: Every sample satisfies -amplitude <= w[n] <= amplitude.
    Why:    Same amplitude contract as all other generators; a chirp that
            exceeds its declared range would clip downstream processing.
    """
    _, w = generate_chirp(amplitude=amplitude)
    assert np.max(w) <= amplitude + 1e-9
    assert np.min(w) >= -amplitude - 1e-9


def test_chirp_linear_and_log_differ():
    """
    What:   The "linear" and "logarithmic" methods produce different waveforms.
    Input:  Default sweep range (20–20000 Hz) and duration, one call per method.
    Output: The two arrays are not element-wise close.
    Why:    Confirms the method parameter is actually applied; if both paths
            produced the same output, logarithmic sweep tests would be silently
            running linear sweeps instead.
    """
    _, w_log = generate_chirp(method="logarithmic")
    _, w_lin = generate_chirp(method="linear")
    assert not np.allclose(w_log, w_lin)


def test_chirp_frequency_increases_over_time(chirp_method):
    """
    What:   The dominant frequency in the first tenth of the signal is lower
            than in the last tenth.
    Input:  f_start=100 Hz, f_end=4000 Hz, duration=2.0 s, both sweep methods.
    Output: _dominant_frequency(w[:seg]) < _dominant_frequency(w[-seg:]).
    Why:    An upward sweep is the defining property of this chirp; a wrong
            sign in the phase formula would produce a downward sweep that still
            passes all other tests.
    """
    _, w = generate_chirp(f_start=100.0, f_end=4000.0, duration=2.0, method=chirp_method)
    seg = len(w) // 10
    assert _dominant_frequency(w[:seg], FS) < _dominant_frequency(w[-seg:], FS)


# ── generate_dc ───────────────────────────────────────────────────────────────

def test_dc_output_shapes():
    """
    What:   The DC generator returns arrays of the expected length.
    Input:  Default parameters (amplitude=1.0, fs=44100, duration=1.0).
    Output: t.shape == w.shape == (N,).
    Why:    Same shape contract as all other generators.
    """
    t, w = generate_dc()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_dc_constant_value(dc_amplitude):
    """
    What:   Every sample of the DC signal equals the requested amplitude.
    Input:  dc_amplitude in {-1.0, 0.0, 0.5, 1.0, 2.0}.
    Output: np.all(w == dc_amplitude).
    Why:    Negative and zero values are included to confirm there is no hidden
            abs() or rectification in the implementation that would make a
            "negative DC" signal appear as positive or zero.
    """
    _, w = generate_dc(amplitude=dc_amplitude)
    assert np.all(w == pytest.approx(dc_amplitude))


# ── generate_multi_tone ───────────────────────────────────────────────────────

def test_multi_tone_output_shape(multi_tone_freqs):
    """
    What:   The multi-tone generator returns arrays of the expected length.
    Input:  1-, 2-, and 3-tone frequency lists; default fs and duration.
    Output: t.shape == w.shape == (N,).
    Why:    Same shape contract as all other generators; a filter or analysis
            stage receiving a wrong-length buffer will silently misalign frames.
    """
    t, w = generate_multi_tone(freqs=multi_tone_freqs)
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_multi_tone_peak_amplitude(multi_tone_freqs, amplitude):
    """
    What:   The peak absolute value of the output equals the requested amplitude.
    Input:  Each tone-count variant crossed with amplitude in {0.5, 1.0, 2.0}.
    Output: max(|w|) == amplitude within floating-point tolerance.
    Why:    The generator normalises the raw mix to ±1 before scaling; this test
            confirms the normalisation is applied regardless of how many tones
            are summed — without it, a 3-tone mix could reach ±3.
    """
    _, w = generate_multi_tone(freqs=multi_tone_freqs, amplitude=amplitude)
    assert np.max(np.abs(w)) == pytest.approx(amplitude, abs=1e-9)


def test_multi_tone_contains_expected_frequencies():
    """
    What:   The FFT of the output has a strong peak near each specified frequency.
    Input:  freqs=[220, 880, 3520] Hz; default fs=44100 and duration=1.0.
    Output: For each f in freqs, the FFT bin at f has amplitude > 30% of the
            global FFT peak.
    Why:    Confirms every requested tone is present; a bug that silently drops
            one tone (e.g., skipping the last element in a list) would not be
            caught by shape or amplitude-range tests alone.
    """
    freqs = [220.0, 880.0, 3520.0]
    _, w = generate_multi_tone(freqs=freqs)
    spectrum = np.abs(np.fft.rfft(w))
    freq_axis = np.fft.rfftfreq(len(w), d=1.0 / FS)
    peak = np.max(spectrum)
    for f in freqs:
        idx = int(np.argmin(np.abs(freq_axis - f)))
        assert spectrum[idx] > 0.3 * peak
