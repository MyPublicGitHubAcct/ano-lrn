# TODO

## Up next

### Add documentation for tests

Describe why each test is appropriate and complete for the function being tested.  Where it is not, add TODO items to rememdy the absences.

### Add a lowpass gate

Write an accurate emulation of an analog lowpass gate, specifically a Buchla 292c/292e.

---

## Completed

### Organize generators and filters by type

Added section comments to `src/python/generators.py` (periodic, noise, transient, sweep, reference) and `src/python/filters.py` (EQ/parametric, shelving); moved `generate_multi_tone` into the periodic group. Rewrote `docs/generators.md` and `docs/filters.md` with per-type sections and summary tables. Replaced the monolithic `examples/plot_generators.py` and `examples/plot_filters.py` with seven per-type scripts: `plot_generators_periodic.py`, `plot_generators_noise.py`, `plot_generators_transient.py`, `plot_generators_sweep.py`, `plot_generators_reference.py`, `plot_filters_eq.py`, `plot_filters_shelving.py`.

### Complete the Audio EQ Cookbook filter set

Added notch, all-pass, low-shelf, and high-shelf biquad filters to `src/python/filters.py`. Each has a private `_coeffs` helper and a public function. Tests added to `tests/test_filters.py` (185 total passing). Plots added to `examples/plot_filters.py` as a second 2×4 figure. DSP theory documented in `docs/filters.md`. README table updated to list all seven filters.

### Verify moog_ladder 4-pole rolloff slope

Added `test_moog_ladder_four_pole_rolloff_slope` to `tests/virtual_analog/test_filters.py`. Uses a 131072-sample impulse response, measures FFT magnitude at 2× and 4× cutoff, and asserts the per-octave slope is between −27 and −16 dB/octave. The Huovilainen approximation gives ≈ −21 dB/oct at those points (due to digital warping); a 2-pole regression would give ≈ −12 dB/oct and fail the upper bound.

### Add plot_filters_utility.py example

Added `examples/plot_filters_utility.py` showing `dc_block` frequency response vs cutoff, comparison with biquad highpass, DC rejection, sub-sonic removal, and audio passthrough across a 2×3 subplot grid.
