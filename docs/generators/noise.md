# Noise Generators

Stochastic signals with broadband spectral energy. Both generators accept an optional `seed` parameter for reproducible output.

| Generator | Key parameters | Spectral shape |
| --- | --- | --- |
| `generate_white_noise` | `seed` | Flat (equal energy per Hz) |
| `generate_pink_noise` | `seed` | −3 dB/octave (equal energy per octave) |

Both return `(t, signal)` — a time axis and signal array sampled at `fs` Hz over `duration` seconds.

---

## `generate_white_noise`

Uniform random samples drawn from `[−A, +A]`. Has a **flat power spectral density** — equal energy per Hz across the band.

White noise is spectrally uniform but perceptually biased toward high frequencies because musical pitch intervals are logarithmic. For broadband filter testing where spectral flatness matters, white noise is the right choice; for perceptual tests, pink noise is more natural.

An optional `seed` parameter makes the output reproducible for regression tests.

---

## `generate_pink_noise`

1/f noise: spectral density falls at **−3 dB/octave**, giving equal energy per octave rather than per Hz. This matches the long-term average spectrum of speech and music, making it a more realistic test signal for perceptual work.

Implementation uses FFT shaping:

1. Generate white noise in the time domain
2. Take the FFT
3. Scale each bin by `1/sqrt(f)` (which makes power proportional to `1/f`)
4. Zero the DC bin
5. Inverse FFT back to the time domain
6. Normalize to ±1

The `seed` parameter controls the underlying white noise RNG.

---

## Parameter Ranges

| Parameter | Range | Notes |
| --- | --- | --- |
| `fs` | 8000–192000 Hz | Any standard audio sample rate; output length is `int(fs · duration)` samples |
| `duration` | > 0 s | |
| `amplitude` | ≥ 0 | Peak amplitude; 0 produces silence; RNG output is scaled to [−amplitude, +amplitude] |
| `seed` | `None` or any int | `None` → non-reproducible; any integer gives a fixed RNG state for deterministic tests |
