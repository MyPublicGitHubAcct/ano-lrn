# Modulated Delay Effects

Both chorus and flanger use a sinusoidally modulated fractional delay line; they differ in delay range and whether a feedback path is present.

---

## `chorus`

Chorus mixes the dry signal with a delayed copy whose delay time oscillates around a center value:

```text
delay(t) = delay_ms · fs/1000 + depth_ms · fs/1000 · sin(2π · rate · t)
wet[n]   = fractional_delay(x, delay(t))
y[n]     = (1 − mix) · x[n] + mix · wet[n]
```

The center delay (typically 20–30 ms) is large enough that the modulated copy sounds like a slightly detuned second voice rather than a comb filter. The result is a "thickening" effect: the slight pitch variation created by the changing delay smears the spectrum without an obvious periodic sweep. At `depth_ms = 0` and `mix = 1` the effect degenerates to a pure fractional delay. At `mix = 0` the output equals the input.

**Vectorised implementation:** the full delay array is computed in advance and passed to `fractional_delay_line`, which uses Lagrange polynomial interpolation (4-point stencil, order 3).

**Typical settings:** rate 0.5–2 Hz, depth_ms 1–5 ms, delay_ms 20–30 ms, mix 0.3–0.7.

---

## `flanger`

Flanger uses a much shorter modulated delay (0.5–5 ms) and feeds the delay output back into the delay input. At each sample:

```text
delayed[n]  = linear_interp(buf, n − delay(t))
buf[n]      = x[n] + feedback · delayed[n]
y[n]        = (1 − mix) · x[n] + mix · delayed[n]
```

The mix of dry and delayed signals creates a feedforward comb with notches at:

```text
f_notch = (2k + 1) · fs / (2 · D)    k = 0, 1, 2, …
```

As the LFO sweeps `D`, the notch comb sweeps through the spectrum — the characteristic "jet plane" whoosh. Positive `feedback` deepens and sharpens the notches by adding IIR resonance:

```text
H(z) ≈ z^(−D) / (1 − feedback · z^(−D))
```

Stability requires `|feedback| < 1`.

**Sample-by-sample loop:** the feedback path from `delayed[n]` back into `buf[n]` creates a dependency that prevents vectorisation. The implementation uses a circular buffer with linear interpolation.

**Typical settings:** rate 0.1–1 Hz, depth_ms 1–3 ms, delay_ms 1–5 ms, feedback 0.3–0.8, mix 0.5.

---

## Chorus vs. Flanger

| Property | Chorus | Flanger |
| --- | --- | --- |
| Center delay | 20–30 ms | 0.5–5 ms |
| Effect character | Detuned doubling | Sweeping comb notches |
| Feedback | No | Yes (optional) |
| Notch spacing | Not audible (too sparse) | fs / D (sweeps through spectrum) |
| Implementation | Vectorised via `fractional_delay_line` | Sample loop (circular buffer) |

---

## Parameter Ranges

**`chorus`**

| Parameter | Range | Notes |
| --- | --- | --- |
| `rate` | 0.1–5 Hz | LFO frequency; musical range 0.5–2 Hz |
| `depth_ms` | 0–10 ms | LFO swing; 0 gives a fixed delay |
| `delay_ms` | 5–50 ms | Center delay; 20–30 ms is the classic chorus zone |
| `mix` | [0, 1] | 0 = dry only, 1 = wet only |
| `fs` | 8000–192000 Hz | Any standard audio sample rate |

**`flanger`**

| Parameter | Range | Notes |
| --- | --- | --- |
| `rate` | 0.05–2 Hz | LFO frequency; 0.1–0.5 Hz is typical |
| `depth_ms` | 0–delay_ms | LFO swing; should not exceed `delay_ms` to keep delay > 0 |
| `delay_ms` | 0.5–10 ms | Center delay; 1–5 ms is the classic flanger zone |
| `feedback` | (−1, 1) | Deepens notches; 0 = feedforward only; negative = "barber pole" sweep |
| `mix` | [0, 1] | 0 = dry only, 1 = wet only |
| `fs` | 8000–192000 Hz | Any standard audio sample rate |
