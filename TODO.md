# TODO

## Up next

### Organize generators and filters by type

Group each generator and filter under a named type category (e.g. periodic, noise, transient for generators; shelving, eq, dynamic for filters):

- Add a `type` label or grouping to each generator in `src/python/generators.py` and each filter in `src/python/filters.py`
- Reorganize `docs/generators.md` and `docs/filters.md` so each section is grouped by type, with a summary table per group
- Split `examples/plot_filters.py` and `examples/plot_generators.py` into per-type example scripts (e.g. `examples/plot_filters_eq.py`, `examples/plot_generators_noise.py`) so each script demonstrates one coherent family of signals or filters

---

## Completed

### Complete the Audio EQ Cookbook filter set

Added notch, all-pass, low-shelf, and high-shelf biquad filters to `src/python/filters.py`. Each has a private `_coeffs` helper and a public function. Tests added to `tests/test_filters.py` (185 total passing). Plots added to `examples/plot_filters.py` as a second 2×4 figure. DSP theory documented in `docs/filters.md`. README table updated to list all seven filters.
