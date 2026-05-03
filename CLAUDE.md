# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

DSP algorithm development in Python, targeting eventual port to C++ (VCV Rack modules). The Python layer is for prototyping and validation; correctness is proven here before porting.

## Commands

```sh
uv sync                        # install dependencies and create .venv
uv run pytest                  # run all tests
uv run pytest tests/test_foo.py::test_bar  # run a single test
uv run python examples/<script>.py         # run an example script
```

To import `src/` modules in scripts outside of pytest, first install the package in editable mode:

```sh
uv pip install -e .
```

(pytest adds `src/` to `sys.path` automatically via `pythonpath = ["src"]` in `pyproject.toml`; editable install is only needed for standalone scripts.)

## Architecture

All generator functions share the same signature shape: positional DSP parameters (`freq`, `fs`, `duration`, `amplitude`, …) followed by signal-specific options. They return a tuple of `(t, signal)` — a time axis array and one or more signal arrays, all sampled at `fs` Hz over `duration` seconds. The private `_time_axis(fs, duration)` helper in [src/python/generators.py](src/python/generators.py) is the canonical way to build the time axis.

- [src/python/generators.py](src/python/generators.py) — all test signal generators (sine, square, sawtooth, triangle, noise, impulse, step, chirp, DC)
- [src/python/filters.py](src/python/filters.py) — biquad lowpass, highpass, bandpass filters (Audio EQ Cookbook)
- [examples/](examples/) — standalone runnable scripts; not part of the package
- [tests/](tests/) — pytest tests; use FFT-based frequency analysis to verify spectral correctness
- [docs/generators.md](docs/generators.md) — DSP concepts for each generator (spectra, formulas, use cases)
- [docs/filters.md](docs/filters.md) — biquad theory, coefficient derivation, and per-filter frequency response behavior

Tests validate shapes, amplitude bounds, and dominant frequency via FFT (`np.fft.rfft`). Follow this pattern when adding tests for new generators.
