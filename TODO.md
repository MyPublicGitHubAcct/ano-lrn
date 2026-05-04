# TODO

## Up next

### Add documentation for tests

Describe why each test is appropriate and complete for the function being tested.  Where it is not, add TODO items to rememdy the absences.

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

### Add phaser effect

Add `phaser(signal, fs, rate, depth, stages)` to a new `src/python/effects/phaser.py` submodule with `src/python/effects/__init__.py`. A phaser passes the signal through `stages` allpass biquad sections (default 4) whose cutoff is modulated by a sine LFO at `rate` Hz; the LFO sweeps each stage's cutoff over a range of `depth` octaves around a fixed centre of fs/8. The phase-shifted copy is mixed 50/50 with the dry signal, producing moving comb-filter notches as the allpass phase transition sweeps through the spectrum. Add tests verifying: output RMS is within 1 dB of input RMS (allpass sections are magnitude-preserving), the output with `rate = 0` and `depth = 0` equals the input at 50% mix, and output shape matches input shape. Add `plot_effects_phaser.py` showing the spectrogram of a broadband signal through the phaser at rate = 0.5 Hz. This builds directly on the allpass biquad in `filters/eq.py` and the LFO pattern from `modulator/amplitude.py`.

---

## Completed

- Organized generators and filters by type with section comments, updated docs, and split `plot_generators.py` / `plot_filters.py` into seven per-type example scripts.
- Completed the Audio EQ Cookbook filter set by adding notch, all-pass, low-shelf, and high-shelf biquad filters with tests, plots, and updated docs.
- Verified `moog_ladder` 4-pole rolloff slope with an impulse-response FFT test asserting −27 to −16 dB/octave between 2× and 4× cutoff.
- Added `examples/plot_filters_utility.py` showing `dc_block` frequency response, DC rejection, and audio passthrough.
- Added `lowpass_gate` (Buchla 292c/292e) with vactrol model, three modes, 9 tests, doc, and example.
- Added `svf` (Chamberlin SVF): LP/BP/HP/notch outputs in one pass, per-sample cutoff control, stability clamp, 18 tests, `docs/filters/svf.md`, `examples/plot_filters_svf.py`.
- Added `zdf_svf` (Zavalishin TPT ZDF SVF): exact −3 dB at cutoff for fc/fs up to 0.4, stable to Nyquist, KVL identity verified, 23 tests, TPT derivation in `docs/filters/svf.md`, `examples/plot_filters_zdf_svf.py`.
