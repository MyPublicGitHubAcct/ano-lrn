# TODO

## Up next

### Fill in missing test coverage

The following coverage gaps were identified when documenting the test suite. Each entry names a missing assertion that would catch a real defect.

**`tests/generators/test_periodic.py`**

- `generate_sawtooth` rises monotonically within each cycle (current tests check FFT peak and amplitude but not waveform shape).
- `generate_sine` at exactly 0 Hz produces a constant signal (DC boundary case).

**`tests/generators/test_noise.py`**

- `generate_white_noise` has flat power spectral density across octave bands.

**`tests/generators/test_transient.py`**

- `generate_step` with `onset=0.0` and `amplitude=0.0` returns all zeros.
- `generate_impulse` with `delay >= duration` raises an error or clips gracefully (currently silently produces a zero signal).

**`tests/generators/test_sweep.py`**

- Instantaneous frequency at `t=0` is close to `f_start` and at `t=duration` is close to `f_end` (requires short-time FFT or analytic signal).
- `f_start == f_end` produces a pure tone identical to `generate_sine` at that frequency.

**`tests/generators/test_reference.py`**

- `generate_multi_tone` with a single frequency matches `generate_sine` within floating-point precision.
- `generate_dc` with `amplitude=0.0` returns all zeros.

**`tests/filters/test_eq.py`**

- Lowpass −3 dB point is within 5% of cutoff (requires impulse-response FFT sweep across a fine frequency grid).
- Bandpass bandwidth (−3 dB points) equals `cutoff/Q`, confirming the Audio EQ Cookbook derivation.

**`tests/filters/test_shelving.py`**

- Shelf transition frequency: gain at `cutoff` should be approximately `gain_db/2` dB (the −3 dB midpoint of the shelf slope).
- `lowshelf` does not affect frequencies well above cutoff (> 1 octave): H(2·cutoff) ≈ 1 regardless of `gain_db`.

**`tests/filters/test_ladder.py`**

- 4-pole rolloff slope is between −16 and −27 dB/octave between 1× and 2× cutoff (mirrors `test_moog_ladder_four_pole_rolloff_slope` in `virtual_analog/test_filters.py` but targets the `filters/` module).
- Resonance peak frequency matches cutoff to within 10%.

**`tests/filters/test_utility.py`**

- `dc_block` with near-zero cutoff behaves as an all-pass and does not attenuate audio-band content.
- Two cascaded `dc_block` calls do not cause instability or excessive attenuation compared to a single call.

**`tests/filters/test_svf.py`**

- Chamberlin SVF LP −3 dB point is within 10% of cutoff (the Chamberlin formulation has slight frequency error at high fc/fs ratios).
- LP + BP + HP = input (Chamberlin identity), analogous to the KVL test in `test_zdf_svf.py`.

**`tests/virtual_analog/test_distortion.py`**

- `analog_saturate` with `drive=1` is close to identity for small signals (`|x| ≪ 1`), confirming small-signal gain ≈ 1.
- `diode_clip` with `threshold=0` clips all positive samples to zero while leaving negative samples unchanged.

**`tests/virtual_analog/test_gate.py`**

- `mode="both"` matches the expected product of independent amplitude and filter paths.
- `attack_time` is respected: after a step open, output reaches 90% of steady-state within `attack_time ± 20%`.

**`tests/adaptive/test_lms.py`**

- Output satisfies `out = desired − error` sample-by-sample.
- `mu=0` produces weights that remain all-zero (no adaptation).
- Very large `mu` causes divergence (error increases), confirming the step-size stability bound is not silently clamped.

**`tests/adaptive/test_nlms.py`**

- `mu=0` produces weights that remain all-zero.
- `epsilon` prevents division-by-zero when the reference signal is silent (all zeros).

**`tests/delay/test_line.py`**

- `delay >= len(signal)` produces an all-zero output (signal pushed entirely beyond the buffer).
- Only integer delays are accepted; fractional delay support is a separate TODO.

**`tests/delay/test_comb.py`**

- `feedback_delay` with negative feedback produces alternating-sign echoes (polarity inversion every bounce).
- `feedback >= 1.0` raises an error or produces bounded output (unrestricted feedback causes the IIR to go unstable).

**`tests/mixing/test_blend.py`**

- `mix` with all-zero weights returns a zero array.
- `crossfade` at intermediate positions satisfies `out = (1−pos)·a + pos·b` for non-constant signals (linearity check).

**`tests/mixing/test_level.py`**

- `normalize` applied twice is idempotent (no hidden state or denormal accumulation).
- `gain` applied to a silent signal returns all zeros (no NaN from `0 × 10^(g/20)`).

**`tests/modulator/test_amplitude.py`**

- Tremolo LFO period: at `rate=1 Hz`, envelope completes one full cycle within 1 second.
- `ring_modulate` by a carrier at the signal frequency produces DC and 2× the frequency (product identity).

**`tests/modulator/test_pitch.py`**

- `depth_samples=0` returns output equal to the input (no modulation).
- Vibrato produces a frequency-modulated output: instantaneous frequency varies around the base frequency by approximately `rate × depth_samples` Hz.
- Very large `depth_samples` (e.g. half the signal length) does not cause index-out-of-bounds or NaN.

**`tests/nonlinear/test_shaping.py`**

- `bitcrush` output remains within [−1, +1] for any input within [−1, +1] (quantisation must not overflow the range).
- `waveshape` with a Chebyshev polynomial of order n produces energy primarily at n× the fundamental.

**`tests/source_filter/test_formant.py`**

- Two formants both produce spectral peaks at their respective frequencies (currently only a single formant is exercised).
- Narrower `bandwidth` produces a sharper spectral peak (measured as energy concentration in a ±50 Hz window).
- `formant_filter` with an empty `formant_freqs` list returns the input unchanged.

**`tests/source_filter/test_lpc.py`**

- `lpc_synthesize` driven by an impulse produces a signal whose dominant frequency matches the original signal's fundamental (round-trip test).
- `order=1` produces a single-sample prediction (first-order AR model).

**`tests/source_separation/test_hpss.py`**

- A pure sine is classified predominantly as harmonic (`_rms(h) ≫ _rms(p)`).
- A mix of a sine and a short noise burst: percussive component contains the majority of the transient energy.
- `kernel_size` affects separation sharpness (larger kernel = stronger harmonic/percussive contrast).

**`tests/source_separation/test_wiener.py`**

- Partial estimate (`estimate = mixture × 0.5`) verifying monotone suppression (output RMS between 0 and the full-pass level).
- Frequency-selective estimate (only the low half of the spectrum) verifying only low-frequency content is passed through.

**`tests/spatial/test_stereo.py`**

- `stereo_widen` with `width > 1` increases mid-side spread (side signal L − R must have higher RMS than unprocessed side).

**`tests/spatial/test_precedence.py`**

- `delay=0` produces identical L and R channels (no precedence effect).

**`tests/spectral/test_features.py`**

- `spectral_centroid` of a pure sine at frequency `f` returns values close to `f` (within one FFT bin) across all steady-state frames.
- `spectral_flux` spikes at a note onset (abrupt change from silence to a tone), confirming its use as an onset detector.

**`tests/spectral/test_processing.py`**

- `spectral_gate` preserves frequency content of a signal that passes the threshold (dominant frequency must remain the same).
- Mixed signal (loud tone + quiet noise): tone passes and noise is suppressed when threshold is set between their levels.

**`tests/time_frequency/test_analysis.py`**

- `spectrogram` values equal 20·log10(|STFT|) or 10·log10(|STFT|²) — confirm which power/amplitude convention is used.
- `spectrogram` of a sine at frequency `f` has its peak column consistently at the expected FFT bin across all time frames.

**`tests/time_segment/test_windowing.py`**

- `apply_window("hamming")` has endpoints near 0.08 (Hamming window does not go fully to zero, distinguishing it from Hann).
- All supported window types produce outputs within [−1, +1] when applied to a unit-amplitude input.

**`tests/warping/test_resampling.py`**

- Resampling a very short signal (e.g. 1 sample) does not crash or produce NaN.
- Non-integer ratio (e.g. 44100 → 48000): output length is `round(len(sig) × target_fs / orig_fs)` and frequency content is preserved.

**`tests/warping/test_stretching.py`**

- `pitch_shift` by exactly 12 semitones raises the dominant frequency to approximately 2× (within 10%), verifying the semitone-to-ratio conversion.
- `time_stretch` preserves the dominant frequency: stretched signal pitch must equal original pitch to within one FFT bin.

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

---

## Completed

- Added docstrings to every test function explaining why the test is appropriate, and added TODO comments in each file where coverage gaps were identified.
- Organized generators and filters by type with section comments, updated docs, and split `plot_generators.py` / `plot_filters.py` into seven per-type example scripts.
- Completed the Audio EQ Cookbook filter set by adding notch, all-pass, low-shelf, and high-shelf biquad filters with tests, plots, and updated docs.
- Verified `moog_ladder` 4-pole rolloff slope with an impulse-response FFT test asserting −27 to −16 dB/octave between 2× and 4× cutoff.
- Added `examples/plot_filters_utility.py` showing `dc_block` frequency response, DC rejection, and audio passthrough.
- Added `lowpass_gate` (Buchla 292c/292e) with vactrol model, three modes, 9 tests, doc, and example.
- Added `svf` (Chamberlin SVF): LP/BP/HP/notch outputs in one pass, per-sample cutoff control, stability clamp, 18 tests, `docs/filters/svf.md`, `examples/plot_filters_svf.py`.
- Added `zdf_svf` (Zavalishin TPT ZDF SVF): exact −3 dB at cutoff for fc/fs up to 0.4, stable to Nyquist, KVL identity verified, 23 tests, TPT derivation in `docs/filters/svf.md`, `examples/plot_filters_zdf_svf.py`.
