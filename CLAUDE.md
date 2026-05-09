# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

DSP algorithm development in Python, targeting eventual port to C++ (VCV Rack modules). The Python layer is for prototyping and validation; correctness is proven here before porting.

## Commands

```sh
uv sync                        # install dependencies and create .venv
uv run pytest                  # run all tests
uv run pytest tests/test_foo.py::test_bar  # run a single test
uv run python plots/<script>.py            # run a plot script
```

To import `src/` modules in scripts outside of pytest, first install the package in editable mode:

```sh
uv pip install -e .
```

(pytest adds `src/` to `sys.path` automatically via `pythonpath = ["src"]` in `pyproject.toml`; editable install is only needed for standalone scripts.)

## Architecture

All generator functions share the same signature shape: positional DSP parameters (`freq`, `fs`, `duration`, `amplitude`, …) followed by signal-specific options. They return a tuple of `(t, signal)` — a time axis array and one or more signal arrays, all sampled at `fs` Hz over `duration` seconds. The private `_time_axis(fs, duration)` helper in [src/python/generators/_helpers.py](src/python/generators/_helpers.py) is the canonical way to build the time axis.

Each top-level module under `src/python/` is a package with submodules. The `__init__.py` re-exports the public API so callers can import directly from the package (e.g. `from python.generators import generate_sine`).

### Source modules

| Package | Submodules | Contents |
| ------- | ---------- | -------- |
| [generators/](src/python/generators/) | `periodic`, `noise`, `transient`, `sweep`, `reference` | Sine, square, sawtooth, triangle, noise, impulse, step, chirp, DC |
| [filters/](src/python/filters/) | `eq`, `shelving`, `ladder`, `utility` | Biquad EQ (Audio EQ Cookbook), shelving filters, Moog ladder (ZDF), utility filters |
| [virtual_analog/](src/python/virtual_analog/) | `filters`, `distortion`, `gate` | VA filter emulations, waveshaping distortion, lowpass gate (vactrol) |
| [adaptive/](src/python/adaptive/) | `lms`, `nlms` | LMS and NLMS adaptive filters |
| [delay/](src/python/delay/) | `line`, `comb`, `modulated` | Delay line, comb filter, chorus and flanger |
| [mixing/](src/python/mixing/) | `blend`, `level` | Signal blending, level/gain utilities |
| [modulator/](src/python/modulator/) | `amplitude`, `pitch` | AM, pitch modulation |
| [nonlinear/](src/python/nonlinear/) | `clipping`, `shaping` | Hard/soft clipping, waveshaping |
| [source_filter/](src/python/source_filter/) | `formant`, `lpc` | Formant synthesis, LPC |
| [source_separation/](src/python/source_separation/) | `hpss`, `wiener` | Harmonic-percussive separation, Wiener filtering |
| [spatial/](src/python/spatial/) | `stereo`, `precedence` | Stereo panning, precedence effect |
| [spectral/](src/python/spectral/) | `features`, `processing` | Spectral feature extraction, spectral processing |
| [time_frequency/](src/python/time_frequency/) | `transform`, `analysis` | STFT/ISTFT, time-frequency analysis |
| [time_segment/](src/python/time_segment/) | `framing`, `windowing` | Block framing, window functions |
| [warping/](src/python/warping/) | `resampling`, `stretching` | Sample-rate conversion, time stretching |

### Supporting directories

- [plots/](plots/) — standalone `plot_<module>_<submodule>.py` scripts; not part of the package
- [tests/](tests/) — pytest tests mirroring the package structure; use FFT-based frequency analysis to verify spectral correctness
- [docs/](docs/) — per-module subdirectories with DSP theory, coefficient derivation, and frequency response notes

Tests validate shapes, amplitude bounds, and dominant frequency via FFT (`np.fft.rfft`). Follow this pattern when adding tests for new modules.
