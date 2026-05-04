# TODO

## Up next

### Add documentation for tests

Describe why each test is appropriate and complete for the function being tested.  Where it is not, add TODO items to rememdy the absences.

### Add a state-variable filter (SVF)

Add `svf(signal, cutoff, fs, resonance, mode)` to `src/python/filters/` as a new `svf.py` submodule. Use the Zölzer/Chamberlin topology: two integrator stages in a feedback loop with a resonance path, solved sample-by-sample. The key advantage over the biquad filters already in `filters/eq.py` is that a single pass produces lowpass, bandpass, and highpass outputs simultaneously at no extra cost, and the cutoff and resonance are directly controllable per-sample (making them suitable for audio-rate modulation). Add tests verifying the −12 dB/octave LP slope, that the BP output peaks at cutoff, and that the HP output passes above cutoff while rejecting below. Export from `filters/__init__.py`. This is an essential building block for the VCV Rack port where filter cutoff is routinely modulated by LFOs and envelopes.

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

---

## Completed

- Organized generators and filters by type with section comments, updated docs, and split `plot_generators.py` / `plot_filters.py` into seven per-type example scripts.
- Completed the Audio EQ Cookbook filter set by adding notch, all-pass, low-shelf, and high-shelf biquad filters with tests, plots, and updated docs.
- Verified `moog_ladder` 4-pole rolloff slope with an impulse-response FFT test asserting −27 to −16 dB/octave between 2× and 4× cutoff.
- Added `examples/plot_filters_utility.py` showing `dc_block` frequency response, DC rejection, and audio passthrough.
- Added `lowpass_gate` (Buchla 292c/292e) with vactrol model, three modes, 9 tests, doc, and example.
