# Tape Echo / Tape Delay

A tape echo models a magnetic tape loop stretched between a record head and one or more playback heads.  The time between the two heads determines the primary delay; the mechanical imperfections of the tape transport — wow, flutter, and tape saturation — give the effect its characteristic warmth and instability.

---

## Architecture

```text
x[n] ──► record head ──► tape loop ──► playback head ──► y[n]
              ▲                                │
              └──────── feedback · sat ────────┘
```

1. **Primary delay** — a fractional-delay read of the input, offset by `delay_time · fs` samples.
2. **Wow** — a slow sinusoidal perturbation of the read position (< 2 Hz), modelling motor-speed drift.
3. **Flutter** — a faster sinusoidal perturbation (8–12 Hz), modelling capstan and pinch-roller eccentricity.
4. **Feedback** — the playback output is soft-clipped and mixed back into the record-head input, producing a decaying echo trail.

---

## Wow and Flutter Model

The per-sample delay curve is:

```text
D[n] = delay_time · fs
     + wow_depth · fs · sin(2π · wow_rate · n / fs)
     + flutter_depth · fs · sin(2π · flutter_rate · n / fs)
```

where `wow_depth` and `flutter_depth` are given in seconds.  The curve is clamped to a minimum of 1 sample to prevent a read/write collision.

Changing the delay by `ΔD` samples from one sample to the next creates an instantaneous pitch shift:

```text
Δf / f ≈ −dD/dn
```

For sinusoidal modulation at rate `r` and depth `d`:

```text
|Δf|_peak ≈ f · 2π · r · d
```

At 440 Hz with flutter at 10 Hz and depth 1 ms this gives ±27.6 Hz peak deviation — easily audible.

---

## Feedback Path and Saturation

Each iteration through the tape loop multiplies the current echo by `feedback` before adding it to the write pointer.  `saturation` applies soft-clipping via `tanh` to model tape oxide magnetisation limits:

```text
fb_signal[n] = tanh(echo[n] · (1 + saturation · 4))
write[n]     = x[n] + feedback · fb_signal[n]
```

At `saturation = 0` the feedback path is linear.  As `saturation → 1` successive echoes are increasingly compressed, preventing runaway amplitude growth and adding harmonic richness on loud transients.

Stability requires `|feedback| < 1`.  With saturation active the effective feedback is always reduced (tanh bounds the signal), so higher nominal feedback values are safe.

---

## Implementation

Because all echoes in a real tape loop pass through the *same* physical tape, the wow and flutter curve is identical for every repeat.  The implementation exploits this by reusing a single pre-computed `delay_curve` array:

```python
for each echo pass k:
    current = fractional_delay_line(current, delay_curve)  # same curve every pass
    output += current
    current = soft_clip(current) * feedback
    if max(|current|) < 1e-8: break
```

This is mathematically equivalent to a circular-buffer implementation but avoids a sample-by-sample Python loop.  `fractional_delay_line` uses a 4-point Lagrange stencil (order 3) for sub-sample accuracy.

---

## Parameter Reference

| Parameter | Default | Range | Notes |
| --- | --- | --- | --- |
| `delay_time` | 0.2 s | 0.01–2 s | Tape-loop length; musical range 50–600 ms |
| `feedback` | 0.4 | [0, 1) | 0 = single echo; 0.9 = very long tail |
| `flutter_rate` | 10 Hz | 8–12 Hz | Capstan/pinch-roller eccentricity |
| `flutter_depth` | 0.5 ms | 0–5 ms | Typical tape machines: 0.1–1 ms |
| `wow_rate` | 0.5 Hz | 0.1–2 Hz | Motor speed drift |
| `wow_depth` | 2 ms | 0–20 ms | Typical tape machines: 0.5–5 ms |
| `saturation` | 0.0 | [0, 1] | 0 = clean; 0.3 = classic warmth; 1 = heavy drive |

---

## Comparison with Chorus and Flanger

| | Chorus | Flanger | Tape Echo |
| --- | --- | --- | --- |
| Delay range | 20–30 ms | 0.5–5 ms | 50–600 ms |
| Modulation | Single LFO | Single LFO | Wow + flutter (two LFOs) |
| Feedback | No | Yes (IIR) | Yes (with saturation) |
| Saturation | No | No | Yes (tape oxide) |
| Dry signal | Mixed in | Mixed in | Returned separately |
| Loop | Vectorised | Sample loop | Iterative passes |
