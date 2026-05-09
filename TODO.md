# TODO

## Up next

### Add notebooks for all source modules

Add a Jupyter notebook for each source file below (a notebook for `filters/svf.py` already exists at `notebooks/svf_filters.ipynb`). Each notebook should mirror the corresponding plot script where one exists, or demonstrate the module's API with annotated cells showing typical use, parameter sweeps, and any relevant visualisations that show off the capabilities of the algorithms.

**`adaptive/`**

- `notebooks/adaptive_lms.ipynb` — `src/python/adaptive/lms.py`
- `notebooks/adaptive_nlms.ipynb` — `src/python/adaptive/nlms.py`

**`delay/`**

- `notebooks/delay_line.ipynb` — `src/python/delay/line.py`
- `notebooks/delay_comb.ipynb` — `src/python/delay/comb.py`

**`filters/`**

- `notebooks/filters_helpers.ipynb` — `src/python/filters/_helpers.py`
- `notebooks/filters_eq.ipynb` — `src/python/filters/eq.py`
- `notebooks/filters_ladder.ipynb` — `src/python/filters/ladder.py`
- `notebooks/filters_shelving.ipynb` — `src/python/filters/shelving.py`
- `notebooks/filters_utility.ipynb` — `src/python/filters/utility.py`

**`generators/`**

- `notebooks/generators_helpers.ipynb` — `src/python/generators/_helpers.py`
- `notebooks/generators_noise.ipynb` — `src/python/generators/noise.py`
- `notebooks/generators_periodic.ipynb` — `src/python/generators/periodic.py`
- `notebooks/generators_reference.ipynb` — `src/python/generators/reference.py`
- `notebooks/generators_sweep.ipynb` — `src/python/generators/sweep.py`
- `notebooks/generators_transient.ipynb` — `src/python/generators/transient.py`

**`mixing/`**

- `notebooks/mixing_blend.ipynb` — `src/python/mixing/blend.py`
- `notebooks/mixing_level.ipynb` — `src/python/mixing/level.py`

**`modulator/`**

- `notebooks/modulator_amplitude.ipynb` — `src/python/modulator/amplitude.py`
- `notebooks/modulator_pitch.ipynb` — `src/python/modulator/pitch.py`

**`nonlinear/`**

- `notebooks/nonlinear_clipping.ipynb` — `src/python/nonlinear/clipping.py`
- `notebooks/nonlinear_shaping.ipynb` — `src/python/nonlinear/shaping.py`

**`source_filter/`**

- `notebooks/source_filter_formant.ipynb` — `src/python/source_filter/formant.py`
- `notebooks/source_filter_lpc.ipynb` — `src/python/source_filter/lpc.py`

**`source_separation/`**

- `notebooks/source_separation_hpss.ipynb` — `src/python/source_separation/hpss.py`
- `notebooks/source_separation_wiener.ipynb` — `src/python/source_separation/wiener.py`

**`spatial/`**

- `notebooks/spatial_stereo.ipynb` — `src/python/spatial/stereo.py`
- `notebooks/spatial_precedence.ipynb` — `src/python/spatial/precedence.py`

**`spectral/`**

- `notebooks/spectral_features.ipynb` — `src/python/spectral/features.py`
- `notebooks/spectral_processing.ipynb` — `src/python/spectral/processing.py`

**`time_frequency/`**

- `notebooks/time_frequency_transform.ipynb` — `src/python/time_frequency/transform.py`
- `notebooks/time_frequency_analysis.ipynb` — `src/python/time_frequency/analysis.py`

**`time_segment/`**

- `notebooks/time_segment_framing.ipynb` — `src/python/time_segment/framing.py`
- `notebooks/time_segment_windowing.ipynb` — `src/python/time_segment/windowing.py`

**`virtual_analog/`**

- `notebooks/virtual_analog_filters.ipynb` — `src/python/virtual_analog/filters.py`
- `notebooks/virtual_analog_distortion.ipynb` — `src/python/virtual_analog/distortion.py`
- `notebooks/virtual_analog_gate.ipynb` — `src/python/virtual_analog/gate.py`

**`warping/`**

- `notebooks/warping_resampling.ipynb` — `src/python/warping/resampling.py`
- `notebooks/warping_stretching.ipynb` — `src/python/warping/stretching.py`

---

### Add Karplus-Strong plucked string synthesis

Add `pluck(freq, fs, duration, damping, pickup)` to a new `src/python/physical/string.py` submodule with a `src/python/physical/__init__.py`. The Karplus-Strong algorithm excites a delay line (length = fs / freq samples) with a burst of white noise, then recirculates the output through a one-pole lowpass (the "loss filter") each cycle: `y[n] = damping * 0.5 * (y[n−D] + y[n−D−1])`. The `pickup` parameter (0–1) selects the read position along the delay line as a fraction of its length, changing the harmonic balance (mid-pickup cuts even harmonics as in a guitar neck vs. bridge pickup). Return a `(t, signal)` tuple following the generator convention. Add tests verifying: dominant frequency is within 2 Hz of the target, signal decays to below −40 dB within `duration`, and `pickup=0.5` has weaker even harmonics than `pickup=0.1`. Add `plot_physical_string.py` and `docs/physical/string.md`. This builds directly on `src/python/delay/line.py` and is one of the most important physical modeling algorithms for VCV Rack.

### Add chorus and flanger effects

Add `chorus(signal, fs, rate, depth, mix)` and `flanger(signal, fs, rate, depth, feedback, mix)` to `src/python/delay/modulated.py`. Both use a sinusoidally modulated delay line — flanger in the 0.5–5 ms range (comb-filter notches that sweep through the spectrum) and chorus in the 10–30 ms range (pitch detuning that thickens the sound). The modulated read position requires sub-sample interpolation; use the same Lagrange approach planned for `fractional_delay_line` (see that TODO). Flanger adds a `feedback` path from output back into the delay buffer, deepening the notches. Add tests verifying the comb-filter notch frequency matches the instantaneous delay at a static modulation position, and that chorus output RMS is within 3 dB of input. This TODO depends on "Add fractional delay line" being completed first.

### Add bandlimited oscillators (BLEP)

The current `generate_square` and `generate_sawtooth` in `src/python/generators/periodic.py` use naive waveforms that alias heavily above a few kHz. Add bandlimited variants using the PolyBLEP (polynomial band-limited step) technique: compute the ideal discontinuity time within a sample period and subtract a precomputed correction polynomial from the samples immediately around each edge. Target a spurious-free dynamic range (SFDR) of at least 60 dB across the audible band at all pitches up to 10 kHz. This is critical for the VCV Rack port — aliased oscillators produce intermodulation in polyphonic patches.

### Add fractional delay line

`delay_line` in `src/python/delay/line.py` only supports integer sample delays. Add a `fractional_delay_line(signal, delay_samples)` function that accepts non-integer `delay_samples` and uses Lagrange polynomial interpolation (order 3 or 4) to compute sub-sample delays. This is needed for accurate vibrato, chorus, and flanger effects where modulation sweeps continuously through fractional positions. The existing `vibrato` function in `modulator/pitch.py` uses its own inline linear interpolation; it should be rewritten to delegate to the new function once it exists.

### Verify pink noise spectral slope

The `generate_pink_noise` function claims 1/f shaping but has no test verifying it. Add a test in `tests/generators/test_noise.py` that generates a long pink noise sequence (≥ 4 seconds at 44100 Hz), computes the power spectral density in octave bands via `np.fft.rfft`, and asserts that the per-octave power slope is between −4 and −2 dB (centred on the ideal −3 dB/octave). A regression to white noise would give 0 dB/octave and fail. Use geometric band centres from 100 Hz to 10 kHz to stay well away from DC and Nyquist where the shaping accuracy degrades.

### Add ADSR envelope generator

Add `generate_adsr(attack, decay, sustain, release, fs, duration)` to `src/python/generators/transient.py`. The four segments should be implemented as linear ramps (or optional exponential curves for decay and release). Output should be a `(t, envelope)` tuple following the existing generator convention. Add tests verifying: peak reaches 1.0 at the end of attack, the sustain level is held correctly, release reaches 0 at exactly the specified time. Add a matching `plot_generators_envelope.py` example showing several ADSR shapes and their response to a gated sine input.

### Add Schroeder reverb

Add `reverb(signal, fs, room_size, damping, wet)` to a new `src/python/reverb/` package using the classic Schroeder topology: four parallel feedback comb filters whose delay times are mutually prime (to avoid spectral coloration), followed by two allpass diffusers in series. Delay times should scale with `room_size` so the reverberation time (RT60) changes predictably. `damping` controls a one-pole lowpass in each comb feedback loop. Document the allpass and comb coefficient derivation from RT60 in `docs/reverb/`. This builds directly on the delay/comb infrastructure already in `src/python/delay/`.

### Add one-pole filter

Add `one_pole_lp(signal, cutoff, fs)` and `one_pole_hp(signal, cutoff, fs)` to `src/python/filters/` as a new `one_pole.py` submodule. The one-pole lowpass has a single coefficient α = 1 − exp(−2π·cutoff/fs) and update rule y[n] = α·x[n] + (1−α)·y[n−1]; the highpass is y[n] = x[n] − one_pole_lp(x[n]). Compared to the biquad filters in `filters/eq.py`, the one-pole has no resonance, costs one multiply-add per sample, and requires only one word of state — exactly what VCV Rack firmware needs to smooth knob and CV values without audible zipper noise. Add tests verifying the −3 dB point is within 5% of `cutoff`, the rolloff slope is −20 dB/decade between one and four octaves above cutoff, and the highpass rejects DC. Export from `filters/__init__.py`. Add `docs/filters/one_pole.md` with the coefficient derivation from the bilinear-free continuous-time pole at s = −2π·cutoff.

### Add wavefolder distortion

Add `wavefold(signal, gain)` to `src/python/virtual_analog/distortion.py`. A wavefolder reflects the signal back each time it would exceed ±1, producing mirror images of the waveform rather than clipping it: while any sample exceeds 1, apply `sample = 2 − sample`; mirror symmetrically for negative values. At `gain = 1` the signal is folded once per half-cycle; at `gain = 2` twice; and so on. Higher fold counts produce richer harmonic content than clipping because the fold adds a new partial each time a boundary is crossed. Add tests verifying: output is bounded to [−1, +1] for all gains, FFT harmonic energy above 2× the fundamental increases monotonically with `gain`, and `gain = 0` produces silence. Add `plot_virtual_analog_distortion.py` showing a sine wave folded at gains 0.5, 1, 2, and 4. Wavefolding is a defining feature of Buchla synthesizers and is directly relevant to the VCV Rack VA module set.

### Add slew limiter

Add `slew(signal, rise_time, fall_time, fs)` to `src/python/modulator/` as a new `slew.py` submodule. A slew limiter caps the rate of change of a signal: the output can rise at most `1 / (rise_time · fs)` per sample and fall at most `1 / (fall_time · fs)` per sample. When the input rises faster than `rise_time` allows, the output lags with a constant-slope ramp; same for falls. Equal `rise_time` and `fall_time` give symmetric portamento. Implement sample-by-sample with a single state variable. Add tests verifying: on a unit step input the output reaches 0.9 within `rise_time` ± 5%, on a falling edge the output reaches 0.1 within `fall_time` ± 5%, and a signal varying more slowly than the slew rate passes unchanged. Add `plot_modulator_slew.py` showing portamento applied to a stepped pitch sequence. Slew limiters are among the most-used utility modules in VCV Rack and a prerequisite for realistic portamento in the C++ port.

### Add sample-and-hold

Add `sample_and_hold(signal, trigger, fs)` to `src/python/modulator/` as a new `sample_hold.py` submodule. The module samples `signal` on every rising edge of `trigger` (transition from ≤ 0 to > 0) and holds that value until the next trigger. With a noise input and a clock trigger this produces the classic "random stepped voltages" of modular synthesis; with a pitched signal it quantizes continuous pitch to a staircase. Implement with a single held-value state variable and a per-sample edge detector (`trigger[n] > 0 and trigger[n-1] <= 0`). Return a `(t, output)` tuple following the generator convention. Add tests verifying: the output is constant between triggers (zero first-difference on non-trigger samples), the output matches the input value at the trigger moment (within floating-point precision), and no change occurs when the trigger stays below zero. Add `plot_modulator_sample_hold.py` example. Sample-and-hold is a fundamental modular-synthesis utility and a direct building block for random-voltage sources in the VCV Rack port.

### Add vector synthesis oscillator

Add `vector_osc(freq, fs, duration, position, pwm)` to a new `src/python/generators/vector.py` submodule. The oscillator blends four waveforms — saw, pulse (with PWM), triangle, and sine — using a 2D `position` parameter (two values in [0, 1]) that maps to a unit square, with each corner assigned to one generator. Bilinear interpolation between the four waveforms gives continuous timbral morphing. The `pwm` parameter (0–1) controls pulse width of the pulse waveform. Return a `(t, signal)` tuple following the generator convention. Add tests verifying: at each corner position the output matches (within 5%) the expected pure waveform, at centre position the RMS is within 3 dB of any corner, and PWM=0.5 gives a standard square wave. Add `plot_generators_vector.py` showing waveform output at the four corners and several intermediate positions. Export from `generators/__init__.py`.

### Add vector synthesis oscillator with FM and phase distortion

Extend the vector oscillator concept with frequency modulation and phase distortion. Add `vector_osc_fm(freq, fs, duration, position, pwm, fm_ratio, fm_depth, pd_amount)` to `src/python/generators/vector.py`. FM is applied by modulating the instantaneous frequency with a sine operator at `fm_ratio × freq` and depth `fm_depth` (in Hz); phase distortion reshapes the phase accumulator using a piecewise linear map controlled by `pd_amount` (0 = no distortion, 1 = maximum reshape) before evaluating each waveform. Both FM and phase distortion are applied before the bilinear blend so all four waveforms are affected uniformly. Add tests verifying: at `fm_depth = 0` and `pd_amount = 0` output matches the basic `vector_osc` within floating-point precision, FM sideband energy increases monotonically with `fm_depth` (measured by FFT), and `pd_amount = 0.5` shifts the dominant spectral centroid relative to `pd_amount = 0`. Add `plot_generators_vector_fm.py` showing spectrograms at several FM depth and PD amount combinations.

### Add phaser effect

Add `phaser(signal, fs, rate, depth, stages)` to a new `src/python/effects/phaser.py` submodule with `src/python/effects/__init__.py`. A phaser passes the signal through `stages` allpass biquad sections (default 4) whose cutoff is modulated by a sine LFO at `rate` Hz; the LFO sweeps each stage's cutoff over a range of `depth` octaves around a fixed centre of fs/8. The phase-shifted copy is mixed 50/50 with the dry signal, producing moving comb-filter notches as the allpass phase transition sweeps through the spectrum. Add tests verifying: output RMS is within 1 dB of input RMS (allpass sections are magnitude-preserving), the output with `rate = 0` and `depth = 0` equals the input at 50% mix, and output shape matches input shape. Add `plot_effects_phaser.py` showing the spectrogram of a broadband signal through the phaser at rate = 0.5 Hz. This builds directly on the allpass biquad in `filters/eq.py` and the LFO pattern from `modulator/amplitude.py`.

### Add envelope follower

Add `envelope_follow(signal, fs, attack_time, release_time, mode)` to `src/python/modulator/` as a new `envelope.py` submodule. An envelope follower tracks the amplitude contour of a signal by full-wave rectification followed by a one-pole AR smoother: during a rise the pole uses `α_attack = 1 − exp(−1 / (attack_time · fs))` and during a fall it uses `α_release`. The optional `mode` parameter selects `"peak"` (rectify then smooth) or `"rms"` (square, smooth, then square-root). Export from `modulator/__init__.py`. Add tests verifying: on a constant-amplitude sine the output converges to within 5% of the true amplitude within `attack_time` seconds, the output decays below 10% of peak within `release_time` seconds after the signal stops, and `mode="rms"` matches `np.sqrt(np.mean(signal**2))` for a stationary sine. Add `plot_modulator_envelope.py` showing the follower tracking the amplitude of a drum-style burst, and `docs/modulator/envelope.md`. This module is a prerequisite for the dynamics compressor TODO and for CV-to-trigger conversion in VCV Rack.

### Add dynamics compressor/limiter

Add `compress(signal, fs, threshold_db, ratio, attack_time, release_time, knee_db, makeup_db)` and `limit(signal, fs, threshold_db, attack_time, release_time)` to a new `src/python/dynamics/` package (`compressor.py` submodule, `__init__.py` re-exporting both). The compressor uses a feedforward RMS level detector (the envelope follower from `modulator/envelope.py`) to compute the gain reduction in dB: for levels above `threshold_db + knee_db / 2`, apply `GR = (level − threshold_db) · (1 − 1/ratio)`; within the soft knee, blend with a quadratic curve. Convert gain reduction to a linear gain and multiply sample-by-sample. The limiter is a compressor with `ratio = ∞` and a hard knee. Add tests verifying: a sine 6 dB above threshold is attenuated by approximately `6 · (1 − 1/ratio)` dB in steady state, a sine below threshold passes unchanged, `makeup_db` shifts output level by the expected amount, and the limiter output never exceeds `threshold_db + 0.5 dB`. Add `plot_dynamics_compressor.py` showing gain reduction curves and before/after waveforms. Add `docs/dynamics/compressor.md` with the gain-computer equations.

### Add noise gate

Add `noise_gate(signal, fs, threshold_db, attack_time, release_time, hold_time)` to `src/python/dynamics/` (alongside the planned compressor) as a new `gate.py` submodule. The gate opens when the signal level (measured by a peak envelope follower) exceeds `threshold_db` and closes when it falls below `threshold_db − 6 dB` (fixed hysteresis). After the level falls below threshold, the gate holds open for `hold_time` seconds before beginning the release ramp, preventing chattering on decaying sounds. Implement as a state machine with four states: closed, attack, open, release. Return a `(t, output)` tuple. Add tests verifying: a sine above threshold passes unchanged (gain = 1) in steady state, a sine below threshold is fully attenuated (< −60 dB) in steady state, a burst above threshold causes the output to reach 90% of steady-state within `attack_time ± 20%`, and `hold_time` delays the onset of release by the correct number of samples. Add `plot_dynamics_gate.py` and `docs/dynamics/gate.md`. This is distinct from `virtual_analog/gate.py` (the Buchla vactrol lowpass gate); it is a conventional dynamics gate for noise reduction and gated drums.

### Add multi-shape LFO

Add `lfo(rate, fs, duration, shape, phase, unipolar)` to `src/python/modulator/` as a new `lfo.py` submodule. The function generates a low-frequency oscillator waveform at `rate` Hz for `duration` seconds. The `shape` parameter selects from `"sine"`, `"triangle"`, `"sawtooth"`, `"reverse_saw"`, `"square"`, and `"sample_hold"` (random stepped, retriggered at `rate`). `phase` sets the initial phase offset in radians. `unipolar` (bool) shifts the range from [−1, +1] to [0, 1]. Return a `(t, lfo_signal)` tuple. Add tests verifying: a sine LFO at `rate = 1 Hz` completes exactly one cycle in 1 second, a square LFO is exactly +1 for the first half-cycle and −1 for the second, a triangle LFO rises and falls linearly, and `unipolar = True` produces output in [0, 1]. Add `plot_modulator_lfo.py` showing all shapes in a single panel. Export from `modulator/__init__.py`. An LFO with multiple shapes is the most fundamental modulation source in any synthesizer and is required by chorus, phaser, tremolo, and vibrato implementations in this codebase.

### Add wavetable oscillator

Add `wavetable_osc(freq, fs, duration, table, interp)` to a new `src/python/generators/wavetable.py` submodule. The function accepts a 1-D NumPy array `table` (one cycle of any waveform, arbitrary length) and generates audio by advancing a floating-point phase accumulator at `freq / fs` cycles per sample, reading the table with interpolation. The `interp` parameter selects `"linear"` (two-point lerp, default) or `"cubic"` (four-point Hermite spline, smoother at low table sizes). Provide `make_table_sine`, `make_table_saw`, `make_table_square`, and `make_table_triangle` helpers so callers do not need to build tables manually. Add tests verifying: a 2048-point sine table produces a signal whose FFT peak is at `freq` within one bin, THD+N with a 2048-point sine table is below −60 dB, and the output length matches the generator convention. Add `plot_generators_wavetable.py` showing morphing between table shapes during playback. Export from `generators/__init__.py`. Wavetable lookup is the foundation of commercial synthesizers such as the Waldorf Wave and Serum, and is the most general oscillator architecture for VCV Rack.

### Add supersaw / unison oscillator

Add `supersaw(freq, fs, duration, voices, detune, spread)` to `src/python/generators/` as a new `unison.py` submodule. The supersaw sums `voices` sawtooth oscillators detuned symmetrically around `freq`: voice offsets are evenly spaced over ±`detune` cents (1 cent = `freq · (2^(1/1200) − 1)` Hz). The `spread` parameter (0–1) controls stereo panning of the voices: at 0 all voices are mono, at 1 voices are spread evenly left to right. Return a `(t, L, R)` tuple of three arrays. Use bandlimited sawtooth generation to suppress aliasing (defer to the BLEP oscillator TODO if not yet done, otherwise use naive saw with a note in the docstring). Add tests verifying: the dominant FFT peak is at `freq`, the number of distinguishable peaks near `freq` matches `voices`, at `detune = 0` all voices merge and the signal is `voices` times louder than a single oscillator, and `spread = 1` produces non-identical L and R arrays. Add `plot_generators_unison.py` and `docs/generators/unison.md`.

### Add oscillator hard sync

Add `hard_sync(master_freq, slave_freq, fs, duration, slave_shape)` to `src/python/generators/` as a new `sync.py` submodule. In hard sync, the slave oscillator's phase is reset to zero on every rising zero-crossing of the master, regardless of where the slave is in its own cycle. This creates a discontinuity that adds bright, buzzy harmonics whose timbre depends on the ratio `slave_freq / master_freq`. The `slave_shape` parameter selects `"saw"`, `"square"`, or `"sine"`. Return a `(t, master, slave_synced)` tuple. Add tests verifying: when `slave_freq == master_freq` the output matches a free-running slave, when `slave_freq = 2 × master_freq` the output repeats with period equal to the master, and FFT energy at the fundamental increases monotonically as `slave_freq` increases above `master_freq`. Add `plot_generators_sync.py` showing a sweep of `slave_freq` from 1× to 4× master. Export from `generators/__init__.py`. Hard sync is one of the most recognisable sounds of vintage analogue synthesis and a standard VCV Rack building block.

### Add sub-octave / frequency divider

Add `sub_octave(signal, fs, divisions)` to `src/python/generators/` as a new `suboctave.py` submodule. A sub-octave generator produces square waves at integer fractions of the input signal's frequency by counting zero-crossings: for `divisions = 2` a flip-flop is toggled on every rising zero-crossing, producing a square wave at half the input frequency. The `divisions` parameter accepts a single integer or a list of integers for multiple simultaneous sub-octaves; in the list case, return a `(t, outputs)` tuple where `outputs` is a list of arrays. Add tests verifying: a 440 Hz sine input produces a 220 Hz square for `divisions = 2` and a 110 Hz square for `divisions = 4`, the output length matches the input, and the duty cycle of the sub-octave square is within 5% of 50% for a sine input. Add `plot_generators_suboctave.py` showing the original tone and two sub-octave voices mixed. Export from `generators/__init__.py`. Sub-octave dividers are a cornerstone of organ synthesis and bass enrichment in modular patches; the divide-by-2 architecture maps directly to 1-bit hardware logic in VCV Rack.

### Add YIN pitch detector

Add `detect_pitch_yin(signal, fs, frame_size, hop_size, threshold)` to a new `src/python/analysis/` package (`pitch.py` submodule, `__init__.py` re-exporting). YIN estimates the fundamental frequency per frame using the difference function `d(τ) = Σ (x[n] − x[n+τ])²`, computing the cumulative mean normalised form `d′(τ)`, and locating the first minimum below `threshold` (default 0.1) using parabolic interpolation for sub-sample lag accuracy. Return a `(times, f0)` pair where `times` are frame-centre times in seconds and `f0[i]` is the estimated pitch in Hz (NaN when no pitch is detected). Add tests verifying: a pure sine at 440 Hz is detected within 2 Hz, a 110 Hz sine is detected within 2 Hz, a white-noise input returns NaN for most frames, and the output length equals `⌈len(signal) / hop_size⌉`. Add `plot_analysis_pitch.py` showing the pitch track of a frequency sweep. Add `docs/analysis/pitch.md` documenting the algorithm. YIN is the standard pitch detection algorithm for monophonic signals and a key building block for pitch-tracking oscillators and tuners in VCV Rack.

### Add pitch quantizer

Add `quantize_pitch(signal, scale, root, glide_time, fs)` to `src/python/modulator/` as a new `quantizer.py` submodule. The quantizer maps each sample (treated as a 1V/octave CV in the range [−5, +5] V) to the nearest allowed pitch in `scale` (e.g. `"major"`, `"minor"`, `"chromatic"`, or a custom list of semitone offsets from `root`). `root` is the tonic as a MIDI pitch number (60 = C4). `glide_time` applies the slew limiter from `modulator/slew.py` to the quantized output so pitch changes are smoothed; set to 0 for hard quantization. Return a `(t, output)` tuple. Provide built-in scale definitions for major, natural minor, pentatonic, and chromatic. Add tests verifying: all output values lie on scale tones within floating-point rounding, a chromatic input sweep produces steps of exactly 1/12 V, and `glide_time > 0` causes transitions to take the expected time. Add `plot_modulator_quantizer.py` and `docs/modulator/quantizer.md`. Pitch quantizers are among the most-used utility modules in VCV Rack for constraining generative CV to musical scales.

### Add modal resonator bank

Add `modal_bank(excitation, fs, frequencies, decay_times, amplitudes)` to `src/python/physical/` as a new `resonator.py` submodule (alongside the planned Karplus-Strong `string.py`). A modal bank models a vibrating body as a sum of damped sinusoidal resonators. Each mode is a second-order resonator implemented as a biquad with poles at `freq_k · exp(−π · freq_k / (decay_k · fs))`, excited by `excitation`. The three list parameters define each mode's frequency, decay time in seconds, and relative amplitude. Return a `(t, output)` tuple. Add tests verifying: a single-mode bank at 440 Hz excited by an impulse produces a decaying sinusoid whose dominant FFT peak is within 2 Hz of 440, two modes at 440 and 880 Hz both appear in the FFT, and a longer `decay_time` produces a signal that takes proportionally longer to fall below −40 dB. Add `plot_physical_resonator.py` showing a bell-like sound with 8 inharmonic modes. Add `docs/physical/resonator.md`. Modal synthesis is the standard physical model for metallophones, bells, and resonant plates — all targets for VCV Rack percussion modules.

### Add granular synthesizer

Add `granulate(signal, fs, grain_duration, grain_density, pitch_shift, scatter, window)` to a new `src/python/granular/` package (`synthesis.py` submodule, `__init__.py` re-exporting). The function reads overlapping short segments ("grains") from random positions within a jitter window of `scatter` seconds, each grain windowed with the specified window type (`"hann"` by default) and pitch-shifted by resampling at a ratio corresponding to `pitch_shift` semitones before mixing into the output. `grain_density` sets the average number of concurrent grains. Return a `(t, output)` tuple the same length as the input. Add tests verifying: at `pitch_shift = 0` and `scatter = 0` the output power is within 6 dB of the input, pitch-shifted output has its dominant frequency shifted by approximately the correct semitone ratio (within 10%), and `grain_density = 1` produces no zero-energy regions. Add `plot_granular_synthesis.py` showing spectrograms at several scatter and pitch-shift settings. Add `docs/granular/synthesis.md`. Granular synthesis is central to modern textural sound design and time/pitch manipulation in VCV Rack.

### Add convolution reverb

Add `convolve_reverb(signal, impulse_response, wet)` to `src/python/reverb/` (alongside the planned Schroeder reverb) as `convolution.py`. Use `scipy.signal.fftconvolve` for fast overlap-add convolution of `signal` with `impulse_response`, normalise the output to prevent clipping, then blend: `out = (1 − wet) · signal + wet · convolved`. Also provide `make_synthetic_ir(duration, fs, rt60, early_reflections)` which generates a synthetic IR as a decaying exponential noise burst with optional early-reflection spikes, so tests do not require external IR files. Add tests verifying: convolving with a unit impulse returns the original signal within floating-point rounding, `wet = 0` returns the dry signal exactly, `wet = 1` returns the fully convolved signal, and RT60 of the output with the synthetic IR is within 20% of the specified value. Add `plot_reverb_convolution.py` and `docs/reverb/convolution.md`. Convolution reverb gives access to measured acoustic spaces and is widely used in VCV Rack via externally loaded IR files.

### Add channel vocoder

Add `vocoder(carrier, modulator_signal, fs, num_bands, attack_time, release_time)` to `src/python/effects/` as a new `vocoder.py` submodule (alongside the planned phaser). A channel vocoder analyses the spectral envelope of `modulator_signal` using a bank of `num_bands` bandpass filters spread logarithmically between 100 Hz and `fs/2`, extracts the amplitude envelope of each band using the envelope follower from `modulator/envelope.py`, then applies those envelopes to the corresponding bands of `carrier`. Return a `(t, output)` tuple. Add tests verifying: with `carrier = modulator_signal` (self-vocoding) the output RMS is within 3 dB of the input, the dominant formant of a vowel modulator is preserved in the output (spectral centroid within 10%), and `num_bands` can range from 8 to 32 without errors. Add `plot_effects_vocoder.py` showing a swept-sine carrier vocoded by a noise modulator. Add `docs/effects/vocoder.md` documenting the filter bank design.

### Add spectral freeze effect

Add `spectral_freeze(signal, fs, freeze_at, hop_size, window_size)` to `src/python/effects/` as a new `freeze.py` submodule. The function analyses `signal` via STFT, identifies the frame nearest `freeze_at` seconds, and generates an output signal of the same length by repeatedly synthesising from the frozen frame with randomised frame-to-frame phase increments (phase vocoder maintenance) to avoid metallic flanging artefacts. Return a `(t, output)` tuple. Add tests verifying: the dominant frequency of the output matches the dominant frequency of `signal` at `freeze_at` within one FFT bin, the output RMS is within 3 dB of the frozen frame's RMS, and `freeze_at = 0` freezes the first frame without errors. Add `plot_effects_freeze.py` showing a chirp frozen at its midpoint. Add `docs/effects/freeze.md` documenting the phase vocoder maintenance step. This builds on `time_frequency/transform.py` (STFT/ISTFT) and introduces the phase vocoder pattern needed for future pitch-shifting and time-stretching refinements.

---

## Completed

- Added docstrings to every test function explaining why the test is appropriate, and added TODO comments in each file where coverage gaps were identified.
- Organized generators and filters by type with section comments, updated docs, and split `plot_generators.py` / `plot_filters.py` into seven per-type example scripts.
- Completed the Audio EQ Cookbook filter set by adding notch, all-pass, low-shelf, and high-shelf biquad filters with tests, plots, and updated docs.
- Verified `moog_ladder` 4-pole rolloff slope with an impulse-response FFT test asserting −27 to −16 dB/octave between 2× and 4× cutoff.
- Added `plots/plot_filters_utility.py` showing `dc_block` frequency response, DC rejection, and audio passthrough.
- Added `lowpass_gate` (Buchla 292c/292e) with vactrol model, three modes, 9 tests, doc, and example.
- Added `svf` (Chamberlin SVF): LP/BP/HP/notch outputs in one pass, per-sample cutoff control, stability clamp, 18 tests, `docs/filters/svf.md`, `plots/plot_filters_svf.py`.
- Added `zdf_svf` (Zavalishin TPT ZDF SVF): exact −3 dB at cutoff for fc/fs up to 0.4, stable to Nyquist, KVL identity verified, 23 tests, TPT derivation in `docs/filters/svf.md`, `plots/plot_filters_zdf_svf.py`.
- Filled in missing test coverage across all 33 test files: 67+ new assertions covering waveform shape, boundary conditions, spectral identities (Chamberlin SVF, Moog ladder slope, formant peaks, LMS convergence, Wiener masking, HPSS contrast), and parameter edge cases. Total suite grew to 485 tests.
