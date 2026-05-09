# ano-lrn

DSP algorithm development and testing in Python, targeting eventual port to C++ (VCV Rack).

## Project Structure

```text
src/         # DSP source modules (Python packages)
tests/       # pytest test files
plots/       # standalone plot scripts
notebooks/   # Jupyter notebooks (one per submodule)
docs/        # DSP theory and coefficient derivation notes
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

   This step is only needed if you want to import modules from `src/` in scripts run outside of pytest — for example, standalone scripts in `plots/` or an interactive REPL. pytest already adds `src/` to its Python path automatically (via `pythonpath = ["src"]` in `pyproject.toml`), so this step is not required to run tests.

## Generators (`generators/`)

`src/python/generators/` provides standard audio DSP test signals. All functions return a `(t, signal)` tuple sampled at `fs` (default 44100 Hz) over `duration` seconds.

| Function | Description |
| --- | --- |
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

## Filters (`filters/`)

`src/python/filters/` provides biquad filters and state-variable filters. Each function takes a signal array and returns a filtered array of the same length.

| Function | Description |
| --- | --- |
| `lowpass(signal, cutoff, fs, Q=0.707)` | 2nd-order Butterworth-style low-pass; passes DC at unity gain |
| `highpass(signal, cutoff, fs, Q=0.707)` | 2nd-order Butterworth-style high-pass; rejects DC |
| `bandpass(signal, cutoff, fs, Q=1.0)` | 2nd-order band-pass; constant 0 dB peak gain; bandwidth = `cutoff / Q` |
| `notch(signal, cutoff, fs, Q=1.0)` | 2nd-order notch (band-reject); null at `cutoff`; higher Q = narrower notch |
| `allpass(signal, cutoff, fs, Q=0.707)` | 2nd-order all-pass; unity magnitude at all frequencies |
| `lowshelf(signal, cutoff, fs, gain_db=6.0)` | 2nd-order low shelf; boosts or cuts below `cutoff` by `gain_db` dB |
| `highshelf(signal, cutoff, fs, gain_db=6.0)` | 2nd-order high shelf; boosts or cuts above `cutoff` by `gain_db` dB |
| `dc_block(signal, cutoff=20.0, fs=44100)` | 1st-order DC blocker; structural zero at DC; −3 dB at `cutoff` Hz |
| `moog_ladder(signal, cutoff, fs, resonance=0.0)` | 4-pole Moog ladder filter (Huovilainen model); −24 dB/oct rolloff |
| `svf(signal, cutoff, fs, q=0.707, mode='lowpass')` | Chamberlin state-variable filter; LP/BP/HP/notch in one pass; supports per-sample cutoff array |
| `zdf_svf(signal, cutoff, fs, q=0.707, mode='lowpass')` | Zavalishin TPT zero-delay-feedback SVF; exact −3 dB at cutoff for fc/fs up to 0.4 |

`Q` controls resonance: `Q=0.707` gives a maximally-flat (Butterworth) response; higher values produce a resonant peak near cutoff.

## Effects

Each effect family lives in its own package under `src/python/`. All effect functions take a signal array as their first argument and return a processed signal (or a tuple of signals for stereo outputs).

### Delay (`delay/`)

| Function | Key parameters | Description |
| --- | --- | --- |
| `delay_line(signal, delay_samples)` | `delay_samples` | Pure integer-sample delay |
| `fractional_delay_line(signal, delay_samples, order=3)` | `delay_samples`, `order` | Lagrange-interpolated fractional delay; `delay_samples` may be a scalar or per-sample array; `order` 3 (4-point) or 4 (5-point) |
| `feedback_delay(signal, delay_samples, feedback=0.5)` | `feedback` | IIR feedback comb filter; exponential echoes |
| `comb_filter(signal, delay_samples, gain=0.5)` | `gain` | FIR feedforward comb; peaks and notches at fs/D intervals |

### Modulator (`modulator/`)

| Function | Key parameters | Description |
| --- | --- | --- |
| `tremolo(signal, rate, depth, fs)` | `rate`, `depth` | Sinusoidal LFO amplitude modulation |
| `ring_modulate(signal, carrier_freq, fs)` | `carrier_freq` | Multiply by cosine; produces sidebands |
| `vibrato(signal, rate, depth_samples, fs)` | `rate`, `depth_samples` | Fractional-delay pitch modulation |

### Nonlinear (`nonlinear/`)

| Function | Key parameters | Description |
| --- | --- | --- |
| `hard_clip(signal, threshold=1.0)` | `threshold` | Symmetrical amplitude clipping |
| `soft_clip(signal, drive=1.0)` | `drive` | Tanh saturation; bounded output |
| `waveshape(signal, coeffs)` | `coeffs` | Polynomial waveshaping with explicit harmonic control |
| `bitcrush(signal, bits=8)` | `bits` | Amplitude quantisation to N-bit resolution |

### Spatial (`spatial/`)

| Function | Key parameters | Returns | Description |
| --- | --- | --- | --- |
| `pan(signal, position)` | `position` ∈ [−1, +1] | `(left, right)` | Equal-power stereo panning |
| `stereo_widen(left, right, width=1.0)` | `width` | `(left, right)` | M/S stereo width control |
| `haas(signal, delay_samples)` | `delay_samples` | `(left, right)` | Haas precedence effect |

### Time Segment (`time_segment/`)

| Function | Key parameters | Description |
| --- | --- | --- |
| `apply_window(signal, window_type='hann')` | `window_type` | Multiply signal by window function |
| `frame(signal, frame_size, hop_size)` | `frame_size`, `hop_size` | Segment into overlapping 2D frame array |
| `overlap_add(frames, hop_size, window_type='hann')` | `hop_size` | OLA reconstruction from frames |

### Time-Frequency (`time_frequency/`)

| Function | Key parameters | Returns | Description |
| --- | --- | --- | --- |
| `stft(signal, frame_size=2048, hop_size=512, window='hann')` | `frame_size`, `hop_size` | complex 2D array | Short-time Fourier transform |
| `istft(S, hop_size=512, window='hann')` | `hop_size` | 1D signal | Inverse STFT via overlap-add |
| `spectrogram(signal, frame_size=2048, hop_size=512)` | `frame_size`, `hop_size` | real 2D array | Magnitude spectrogram in dB |

### Source-Filter (`source_filter/`)

| Function | Key parameters | Description |
| --- | --- | --- |
| `lpc_coeffs(signal, order=12)` | `order` | Autocorrelation LPC analysis; returns predictor coefficients |
| `lpc_synthesize(excitation, coeffs)` | `coeffs` | All-pole synthesis filter (speech vocoder) |
| `formant_filter(signal, formant_freqs, bandwidths, fs)` | `formant_freqs`, `bandwidths` | Cascade of 2nd-order resonators at formant frequencies |

### Adaptive (`adaptive/`)

| Function | Key parameters | Returns | Description |
| --- | --- | --- | --- |
| `lms(desired, reference, filter_order=32, mu=0.01)` | `filter_order`, `mu` | `(output, error, weights)` | LMS adaptive FIR filter |
| `nlms(desired, reference, filter_order=32, mu=0.5, eps=1e-8)` | `mu`, `eps` | `(output, error, weights)` | Normalised LMS; stable across input levels |

### Spectral (`spectral/`)

| Function | Key parameters | Returns | Description |
| --- | --- | --- | --- |
| `spectral_centroid(signal, fs, frame_size=2048, hop_size=512)` | `fs` | Hz array | Frequency-weighted mean per frame |
| `spectral_flux(signal, frame_size=2048, hop_size=512)` | `frame_size`, `hop_size` | array | Frame-to-frame magnitude change (onset detection) |
| `spectral_gate(signal, threshold_db, frame_size=2048, hop_size=512)` | `threshold_db` | signal | Suppress STFT bins below threshold |

### Warping (`warping/`)

| Function | Key parameters | Description |
| --- | --- | --- |
| `resample(signal, orig_fs, target_fs)` | `orig_fs`, `target_fs` | Polyphase sample-rate conversion |
| `time_stretch(signal, rate, frame_size=2048, hop_size=512)` | `rate` | Phase-vocoder time stretch; rate > 1 = slower |
| `pitch_shift(signal, semitones, fs, frame_size=2048, hop_size=512)` | `semitones` | Phase-vocoder pitch shift; duration unchanged |

### Virtual Analog (`virtual_analog/`)

| Function | Key parameters | Description |
| --- | --- | --- |
| `moog_ladder(signal, cutoff, fs=44100, resonance=0.0)` | `cutoff`, `resonance` | 4-pole Moog ladder filter (VA formulation) |
| `diode_clip(signal, threshold=0.7)` | `threshold` | Asymmetric diode clipping (hard positive / soft negative) |
| `analog_saturate(signal, drive=1.0)` | `drive` | 3rd-order polynomial tube saturation |
| `lowpass_gate(signal, control, fs, mode='both')` | `control`, `mode` | Buchla 292c/292e vactrol lowpass gate; `mode` ∈ `'lp'`, `'vca'`, `'both'` |

### Mixing (`mixing/`)

| Function | Key parameters | Description |
| --- | --- | --- |
| `gain(signal, gain_db)` | `gain_db` | Scale amplitude by dB amount |
| `mix(signals, weights=None)` | `weights` | Weighted sum of multiple signals |
| `crossfade(signal_a, signal_b, position)` | `position` ∈ [0, 1] | Linear blend between two signals |
| `normalize(signal, target_db=-3.0)` | `target_db` | Scale peak to target dB level |

### Source Separation (`source_separation/`)

| Function | Key parameters | Returns | Description |
| --- | --- | --- | --- |
| `hpss(signal, fs=44100, frame_size=2048, hop_size=512, kernel_size=31)` | `kernel_size` | `(harmonic, percussive)` | Median-filter harmonic-percussive separation |
| `wiener_filter(mixture, source_estimate, frame_size=2048, hop_size=512)` | — | signal | Wiener mask source extraction |

## Running Tests

```sh
uv run pytest
```

## Running Plots

```sh
uv run python plots/<script>.py
```

## Running Notebooks

```sh
uv run jupyter notebook notebooks/<notebook>.ipynb
```
