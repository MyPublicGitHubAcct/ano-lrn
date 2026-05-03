# TODO

## Up next

### Verify moog_ladder 4-pole rolloff slope

The existing `moog_ladder` tests in `tests/test_filters.py` confirm shape, DC pass, low-freq pass, high-freq attenuation, and resonance boost, but none verify the 24 dB/octave rolloff rate that defines a 4-pole ladder filter. Add a test that measures attenuation at one octave above cutoff and at two octaves above cutoff, then asserts the per-octave slope is approximately −24 dB (e.g. within ±3 dB). Without this, a regression to a 2-pole implementation would pass all current tests.

### Add plot_filters_utility.py example

Every filter group has a matching example script (`plot_filters_eq.py`, `plot_filters_shelving.py`, `plot_filters_ladder.py`), but `dc_block` — the only utility filter — has no plot. Add `examples/plot_filters_utility.py` showing the `dc_block` frequency response at several cutoff frequencies (e.g. 5 Hz, 20 Hz, 80 Hz), following the same structure as the other plot scripts (frequency-response panel via `freqz` or impulse response + FFT).

---

## Completed

### Organize generators and filters by type

Added section comments to `src/python/generators.py` (periodic, noise, transient, sweep, reference) and `src/python/filters.py` (EQ/parametric, shelving); moved `generate_multi_tone` into the periodic group. Rewrote `docs/generators.md` and `docs/filters.md` with per-type sections and summary tables. Replaced the monolithic `examples/plot_generators.py` and `examples/plot_filters.py` with seven per-type scripts: `plot_generators_periodic.py`, `plot_generators_noise.py`, `plot_generators_transient.py`, `plot_generators_sweep.py`, `plot_generators_reference.py`, `plot_filters_eq.py`, `plot_filters_shelving.py`.

### Complete the Audio EQ Cookbook filter set

Added notch, all-pass, low-shelf, and high-shelf biquad filters to `src/python/filters.py`. Each has a private `_coeffs` helper and a public function. Tests added to `tests/test_filters.py` (185 total passing). Plots added to `examples/plot_filters.py` as a second 2×4 figure. DSP theory documented in `docs/filters.md`. README table updated to list all seven filters.
