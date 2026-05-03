# Test Signal Generators

All generators live in `src/python/generators.py`. They share a common signature shape and return `(t, signal)` — a time axis and a signal array, both sampled at `fs` Hz over `duration` seconds.

---

## Periodic waveforms

### Sine (`generate_sine`)

A pure sinusoid — the only waveform with energy at exactly one frequency. Used as the reference signal when you need to isolate a single frequency (e.g. filter passband/stopband tests).

```text
x(t) = A · sin(2π f t + φ)
```

The spectrum of an ideal sine is a single impulse at `f`. Any energy elsewhere in the FFT output indicates distortion or aliasing.

---

### Square (`generate_square`)

A square wave with configurable duty cycle. A 50% duty cycle square wave contains only **odd harmonics**:

```text
x(t) = (4A/π) · Σ sin(2π(2k−1)f t) / (2k−1),  k = 1, 2, 3, …
```

Harmonics decay as `1/n`. Duty cycle `d ≠ 0.5` introduces even harmonics as well. Implementation uses a phase-comparison approach: the instantaneous phase `(2π f t) mod 2π` is compared against `d · 2π` to select +1 or −1 each sample.

Use cases: testing harmonic content handling, PWM synthesis, hard-clipping behavior.

---

### Sawtooth (`generate_sawtooth`)

Rises linearly from −1 to +1 over each period, then resets. Contains **all harmonics** (both odd and even):

```text
x(t) = (2A/π) · Σ (−1)^(k+1) sin(2π k f t) / k,  k = 1, 2, 3, …
```

Harmonics decay as `1/n`. The implementation maps `(f·t) mod 1` into `[−1, +1]` via `2·frac − 1`. No anti-aliasing is applied; Gibbs ringing is present near the discontinuity.

Use cases: subtractive synthesis test signals, testing aliasing at high frequencies.

---

### Triangle (`generate_triangle`)

A symmetric ramp that goes up then down within each period. Contains only **odd harmonics**, decaying as `1/n²` (faster than square or sawtooth):

```text
x(t) = (8A/π²) · Σ (−1)^k sin(2π(2k+1)f t) / (2k+1)²,  k = 0, 1, 2, …
```

The implementation folds the sawtooth: `|2·frac − 1|` gives a triangle in `[0, 1]`, then `2·|…| − 1` rescales to `[−1, +1]`.

Use cases: testing low-order harmonic response; smoother test tone than square.

---

## Noise

### White noise (`generate_white_noise`)

Uniform random samples drawn from `[−A, +A]`. Has a **flat power spectral density** — equal energy per Hz across the band.

White noise is spectrally uniform but perceptually biased toward high frequencies because musical pitch intervals are logarithmic. For broadband filter testing where spectral flatness matters, white noise is the right choice; for perceptual tests, pink noise is more natural.

An optional `seed` parameter makes the output reproducible for regression tests.

---

### Pink noise (`generate_pink_noise`)

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

## Deterministic test signals

### Impulse (`generate_impulse`)

A single non-zero sample (Dirac delta approximation) at time `delay`. All other samples are zero.

The Fourier transform of a Dirac delta is a constant — **flat spectrum at all frequencies**. This means that running an impulse through any LTI system and taking the FFT of the output gives the system's full frequency response in one shot. This is how `examples/plot_filters.py` derives filter frequency responses.

```text
x[n] = A · δ[n − delay·fs]
```

Use cases: measuring impulse responses, deriving frequency responses, unit testing filter shapes.

---

### Step (`generate_step`)

Switches from 0 to `amplitude` at time `onset` and holds. The integral of an impulse; its spectrum rolls off as `1/f`.

The step response of a system reveals how it handles sudden transitions: overshoot and ringing indicate underdamped poles; slow rise indicates heavy low-pass filtering. DC gain can be read directly from the settled output value.

```text
x[n] = A · u[n − onset·fs]
```

Use cases: testing transient behavior, measuring DC gain, verifying filter stability after a level jump.

---

### Chirp (`generate_chirp`)

A frequency sweep from `f_start` to `f_end` over `duration` seconds. Two methods are available:

**Logarithmic (default)** — instantaneous frequency increases exponentially:
```text
f(t) = f_start · e^(k·t),   k = ln(f_end / f_start) / T
phase(t) = 2π f_start (e^(k·t) − 1) / k
```
Each octave takes the same wall-clock time, matching the logarithmic spacing of the musical scale. Preferred for audio-band frequency response measurements.

**Linear** — instantaneous frequency increases linearly:
```text
f(t) = f_start + (f_end − f_start) · t / T
phase(t) = 2π (f_start · t + (f_end − f_start) · t² / 2T)
```
Equal Hz per second; concentrates dwell time at low frequencies relative to linear frequency spacing.

Use cases: sweeping a filter's passband, visualizing time-frequency behavior via spectrogram.

---

### DC (`generate_dc`)

A constant signal at `amplitude`. Has energy only at 0 Hz.

```text
x[n] = A  for all n
```

Use cases: testing DC rejection (high-pass, band-pass filters must drive this to zero); testing DC pass-through (low-pass filters must preserve it at unity gain); verifying that an algorithm does not introduce or remove a DC offset.

---

### Multi-tone (`generate_multi_tone`)

Sum of pure sinusoids at the frequencies in `freqs`, normalized so the peak amplitude equals `amplitude`:

```text
raw(t) = Σ sin(2π fₖ t)
x(t) = amplitude · raw(t) / max(|raw(t)|)
```

Peak normalization uses the actual maximum of the discrete signal (not the theoretical worst-case) so the output amplitude is exact to floating-point precision.

Use cases: verifying that a filter selectively passes or blocks specific frequencies in one pass; testing intermodulation distortion; characterizing linearity when multiple tones are present simultaneously.

---

## Digital frequency references

These generators produce exact discrete-time cosine sequences at fixed fractions of the sample rate. They are computed from sample indices rather than a continuous time axis, so each sample value is mathematically exact (no floating-point drift accumulates over a long buffer).

All three start at phase π — i.e., `−cos(2π·freq·n/fs)` — so the first sample is always −1 and the pattern is immediately readable in a sample-level debugger.

### Nyquist (`generate_nyquist`)

```text
x[n] = −cos(π n) = (−1)^(n+1)   →   −1, 1, −1, 1, …
```

A cosine at exactly **fs/2** — the highest frequency a discrete system can represent. Every sample is ±1 with no intermediate values. A low-pass filter at any cutoff below fs/2 must attenuate this signal; a high-pass filter near fs/2 must pass it.

Use cases: verifying low-pass stopband attenuation at the extreme edge; testing high-pass passband gain; checking for Nyquist-frequency aliasing artifacts.

---

### Half-Nyquist (`generate_half_nyquist`)

```text
x[n] = −cos(π n / 2)   →   −1, 0, 1, 0, −1, 0, 1, 0, …
```

A cosine at **fs/4** — halfway between DC and Nyquist. The zero crossings at every other sample make the period immediately visible: four samples per cycle. This sits squarely in the middle of the representable frequency range, making it a clean reference for checking filter behavior away from both extremes.

Use cases: verifying filter gain at the midpoint of the spectrum; testing phase response at fs/4; a convenient sanity-check frequency for any filter whose cutoff is near fs/4.

---

### Quarter-Nyquist (`generate_quarter_nyquist`)

```text
x[n] = −cos(π n / 4)   →   −1, −√2/2, 0, √2/2, 1, √2/2, 0, −√2/2, …
```

A cosine at **fs/8** — one quarter of the Nyquist frequency. The eight-sample period and the ±√2/2 ≈ ±0.707 values at the ±45° points are a recognizable fingerprint in a sample dump.

Use cases: filter response measurements in the lower quarter of the spectrum; checking that a high-pass filter attenuates this signal relative to half-Nyquist; testing oversampled paths where fs/8 of the output rate corresponds to a musically relevant frequency.
