# ano-lrn

DSP algorithm development and testing in Python, targeting eventual port to C++ (VCV Rack).

## Project Structure

```
src/       # DSP source modules
tests/     # pytest test files
examples/  # standalone example scripts
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package and environment manager

## Setup

1. **Clone the repo**

   ```sh
   git clone <repo-url>
   cd ano-lrn
   ```

2. **Create the virtual environment and install dependencies**

   ```sh
   uv sync
   ```

   This reads `.python-version` (Python 3.9) and `pyproject.toml`, creates `.venv/`, and installs all dependencies.

3. **Optional: install the project in editable mode**

   ```sh
   uv pip install -e .
   ```

   This step is only needed if you want to import modules from `src/` in scripts run outside of pytest — for example, standalone scripts in `examples/` or an interactive REPL. pytest already adds `src/` to its Python path automatically (via `pythonpath = ["src"]` in `pyproject.toml`), so this step is not required to run tests.

## Test Signals

`src/python/generators.py` provides a set of standard audio DSP test signals. All functions return a time axis `t` and one or more signal arrays sampled at `fs` (default 44100 Hz) over `duration` seconds.

| Function | Description |
|---|---|
| `generate_sine` | Single sine wave with `freq`, `amplitude`, and `phase` control |
| `generate_square` | Square wave with configurable `duty` cycle (0–1; default 0.5) |
| `generate_sawtooth` | Sawtooth wave that rises linearly from −1 to +1 per period |
| `generate_triangle` | Symmetric triangle wave that ramps linearly between −1 and +1 |
| `generate_white_noise` | Uniform random noise with a flat frequency spectrum; accepts an optional `seed` for reproducibility |
| `generate_pink_noise` | 1/f noise shaped via FFT, normalized to ±1; more spectrally natural than white noise for perceptual testing |
| `generate_impulse` | Single non-zero sample (Dirac delta) at `delay` seconds; use this to measure an algorithm's impulse response |
| `generate_step` | Unit step that switches from 0 to `amplitude` at `onset` seconds; useful for testing transient and DC behaviour |
| `generate_chirp` | Frequency sweep from `f_start` to `f_end`; `method="logarithmic"` (default) gives perceptually uniform spacing across the audio band, `method="linear"` gives uniform Hz spacing |
| `generate_dc` | Constant signal at `amplitude`; useful for testing DC rejection and offset handling |
| `generate_multi_tone` | Normalized sum of sinusoids at the given `freqs` list; peak is scaled to `amplitude`; use this to verify a filter selectively passes or rejects specific frequencies |
| `generate_nyquist` | Alternating `−1, 1, −1, 1, …` — a cosine at exactly fs/2; the highest representable frequency in a discrete system |
| `generate_half_nyquist` | Pattern `−1, 0, 1, 0, …` — a cosine at fs/4 |
| `generate_quarter_nyquist` | Pattern `−1, −0.707, 0, 0.707, 1, …` — a cosine at fs/8 |

## Filters

`src/python/filters.py` provides three second-order biquad filters based on the Audio EQ Cookbook formulas. Each function takes a signal array and returns a filtered array of the same length.

| Function | Description |
|---|---|
| `lowpass(signal, cutoff, fs, Q=0.707)` | 2nd-order Butterworth-style low-pass; passes below `cutoff`, attenuates above; passes DC at unity gain |
| `highpass(signal, cutoff, fs, Q=0.707)` | 2nd-order Butterworth-style high-pass; passes above `cutoff`, attenuates below; rejects DC |
| `bandpass(signal, cutoff, fs, Q=1.0)` | 2nd-order band-pass with constant 0 dB peak gain; bandwidth = `cutoff / Q`; rejects DC |

`Q` controls the sharpness of the transition: `Q=0.707` gives a maximally-flat (Butterworth) response; higher values produce a resonant peak near the cutoff frequency and a narrower passband for the band-pass filter.

## Running Tests

```sh
uv run pytest
```

## Running Examples

```sh
uv run python examples/<script>.py
```
