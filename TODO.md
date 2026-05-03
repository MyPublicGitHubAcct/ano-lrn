# TODO

## Up next

---

## Completed

### Organize generators and filters by type

Added section comments to `src/python/generators.py` (periodic, noise, transient, sweep, reference) and `src/python/filters.py` (EQ/parametric, shelving); moved `generate_multi_tone` into the periodic group. Rewrote `docs/generators.md` and `docs/filters.md` with per-type sections and summary tables. Replaced the monolithic `examples/plot_generators.py` and `examples/plot_filters.py` with seven per-type scripts: `plot_generators_periodic.py`, `plot_generators_noise.py`, `plot_generators_transient.py`, `plot_generators_sweep.py`, `plot_generators_reference.py`, `plot_filters_eq.py`, `plot_filters_shelving.py`.

### Complete the Audio EQ Cookbook filter set

Added notch, all-pass, low-shelf, and high-shelf biquad filters to `src/python/filters.py`. Each has a private `_coeffs` helper and a public function. Tests added to `tests/test_filters.py` (185 total passing). Plots added to `examples/plot_filters.py` as a second 2×4 figure. DSP theory documented in `docs/filters.md`. README table updated to list all seven filters.
