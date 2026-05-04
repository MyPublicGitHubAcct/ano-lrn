# Periodic Generators

Repeating waveforms at a fixed fundamental frequency. All harmonics are exact multiples of `freq`; no anti-aliasing is applied.

| Generator | Key parameters | Spectral content |
| --- | --- | --- |
| `generate_sine` | `freq`, `phase` | Single tone at `freq` |
| `generate_square` | `freq`, `duty` | Odd harmonics at duty=0.5, all harmonics otherwise; 1/n decay |
| `generate_sawtooth` | `freq` | All harmonics; 1/n decay |
| `generate_triangle` | `freq` | Odd harmonics; 1/n² decay |
| `generate_multi_tone` | `freqs` | Tones at each frequency in `freqs`, peak-normalized |

All return `(t, signal)` — a time axis and signal array sampled at `fs` Hz over `duration` seconds.

---

## `generate_sine`

A pure sinusoid — the only waveform with energy at exactly one frequency.

```text
x(t) = A · sin(2π f t + φ)
```

The spectrum of an ideal sine is a single impulse at `f`. Any energy elsewhere in the FFT output indicates distortion or aliasing. The `phase` parameter shifts the waveform; at `phase = π/2` the result is a cosine.

**Use cases:** reference signal for filter passband/stopband tests; carrier for modulation; single-frequency RMS calibration.

---

## `generate_square`

A square wave with configurable duty cycle. A 50% duty cycle square wave contains only **odd harmonics**:

```text
x(t) = (4A/π) · Σ sin(2π(2k−1)f t) / (2k−1),  k = 1, 2, 3, …
```

Harmonics decay as `1/n`. Duty cycle `d ≠ 0.5` introduces even harmonics as well. Implementation uses a phase-comparison approach: the instantaneous phase `(2π f t) mod 2π` is compared against `d · 2π` to select +1 or −1 each sample.

**Use cases:** testing harmonic content handling, PWM synthesis, hard-clipping behavior.

---

## `generate_sawtooth`

Rises linearly from −1 to +1 over each period, then resets. Contains **all harmonics** (both odd and even):

```text
x(t) = (2A/π) · Σ (−1)^(k+1) sin(2π k f t) / k,  k = 1, 2, 3, …
```

Harmonics decay as `1/n`. The implementation maps `(f·t) mod 1` into `[−1, +1]` via `2·frac − 1`. No anti-aliasing is applied; Gibbs ringing is present near the discontinuity.

**Use cases:** subtractive synthesis test signals, testing aliasing at high frequencies.

---

## `generate_triangle`

A symmetric ramp that goes up then down within each period. Contains only **odd harmonics**, decaying as `1/n²` (faster than square or sawtooth):

```text
x(t) = (8A/π²) · Σ (−1)^k sin(2π(2k+1)f t) / (2k+1)²,  k = 0, 1, 2, …
```

The implementation folds the sawtooth: `|2·frac − 1|` gives a triangle in `[0, 1]`, then `2·|…| − 1` rescales to `[−1, +1]`.

**Use cases:** testing low-order harmonic response; smoother test tone than square.

---

## `generate_multi_tone`

Sum of pure sinusoids at the frequencies in `freqs`, normalized so the peak amplitude equals `amplitude`:

```text
raw(t) = Σ sin(2π fₖ t)
x(t) = amplitude · raw(t) / max(|raw(t)|)
```

Peak normalization uses the actual maximum of the discrete signal (not the theoretical worst-case) so the output amplitude is exact to floating-point precision.

**Use cases:** verifying that a filter selectively passes or blocks specific frequencies in one pass; testing intermodulation distortion; characterizing linearity when multiple tones are present simultaneously.

---

## Parameter Ranges

| Parameter | Applies to | Range | Notes |
| --- | --- | --- | --- |
| `freq` | all periodic | 0 < freq < fs/2 | No anti-aliasing; aliasing artifacts appear when freq > fs/4; practical audio range 20–20000 Hz |
| `fs` | all | 8000–192000 Hz | Any standard audio sample rate; output length is `int(fs · duration)` samples |
| `duration` | all | > 0 s | |
| `amplitude` | all | any real | 0 produces silence; negative values are valid and invert polarity |
| `phase` | sine | any real (radians) | Values outside [0, 2π] are valid; sin wraps automatically |
| `duty` | square | [0, 1] | 0 = all −amplitude; 1 = all +amplitude; 0.5 = symmetric square wave (odd harmonics only) |
| `freqs` | multi_tone | non-empty list | Each frequency subject to the same freq constraints above |
