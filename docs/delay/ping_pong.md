# Ping-Pong Delay

A ping-pong delay is a stereo effect that routes successive echoes to alternating channels. The first echo appears in the right channel at `delay_time` seconds; the second in the left channel at `2 × delay_time`; and so on, each attenuated by `feedback`.

---

## `ping_pong_delay`

```python
ping_pong_delay(signal, fs, delay_time, feedback=0.5, mix=0.5) -> (t, L, R)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `signal` | `ndarray` | Mono input array |
| `fs` | `int` | Sample rate in Hz |
| `delay_time` | `float` | Echo period in seconds |
| `feedback` | `float` | Gain per bounce, `[0, 1)`. 0 = single echo, 0.9 = many decaying bounces |
| `mix` | `float` | `0` = fully dry (both channels), `1` = fully wet |

Returns `(t, L, R)` — a time axis and left/right output arrays, all the same length as `signal`.

---

## Signal Flow

The input signal passes through a chain of fractional delay lines. After each delay of `delay_time × fs` samples, the echo is routed to the next channel and attenuated by `feedback` before being fed into the next delay:

```text
x[n] ──→ [Delay D] ──→ R_wet (echo 1, amplitude = 1)
               ↓ × feedback
          [Delay D] ──→ L_wet (echo 2, amplitude = feedback)
               ↓ × feedback
          [Delay D] ──→ R_wet (echo 3, amplitude = feedback²)
               ↓ × feedback
              ...
```

The wet channels and dry signal are blended by `mix`:

```text
L[n] = (1 − mix) · x[n] + mix · L_wet[n]
R[n] = (1 − mix) · x[n] + mix · R_wet[n]
```

At `mix = 0` both channels carry the dry signal with no echoes. At `mix = 1` the dry signal is fully replaced by the alternating echo trail.

---

## Echo Pattern

For an impulse input at `t = 0`, the output consists of:

| Echo | Channel | Time | Amplitude |
| --- | --- | --- | --- |
| 1st | R | `delay_time` | `1` |
| 2nd | L | `2 × delay_time` | `feedback` |
| 3rd | R | `3 × delay_time` | `feedback²` |
| 4th | L | `4 × delay_time` | `feedback³` |
| … | … | `k × delay_time` | `feedback^(k−1)` |

The series is finite in practice: once the echo amplitude falls below `1e-8` (or the delay pushes the echo beyond the signal length), the loop terminates.

---

## Fractional Delay

Echo positions are resolved by `fractional_delay_line` using 4-point Lagrange polynomial interpolation. This allows `delay_time` to correspond to a non-integer number of samples while preserving sub-sample accuracy, which is essential when `delay_time × fs` is not an integer (virtually always the case for musically meaningful delay times).

---

## Implementation

The implementation is fully vectorised. Instead of a per-sample circular buffer, successive echoes are computed by applying `fractional_delay_line` to the previous echo in a short loop:

```python
current = signal
for k in range(500):
    current = fractional_delay_line(current, D)
    (R_wet if k % 2 == 0 else L_wet) += current
    current *= feedback
    if max(|current|) < 1e-8:
        break
```

Because each iteration passes a progressively smaller (and further right-shifted) array through `fractional_delay_line`, the loop terminates rapidly — typically within `ceil(log(1e-8) / log(feedback))` iterations for a given `feedback`, or earlier when the delay exceeds the signal length.

---

## Parameter Ranges

| Parameter | Range | Notes |
| --- | --- | --- |
| `delay_time` | 0.01–1 s | 30–200 ms is the typical musical range |
| `feedback` | [0, 1) | 0 = single echo, 0.5 = several audible bounces, 0.9 = long decay |
| `mix` | [0, 1] | 0.5 is a common starting point |
| `fs` | 8000–192000 Hz | Any standard audio sample rate |

---

## VCV Rack Notes

Ping-pong delay is among the most commonly patched stereo effects in VCV Rack. In a C++ port:

- Replace the vectorised `fractional_delay_line` calls with a pair of circular ring buffers (one per channel direction), each of length `delay_time × fs + guard` samples.
- Maintain two write pointers advanced one sample per tick; the read pointer for each buffer lags by `delay_time × fs` samples.
- Cross-feed: the output of the right buffer feeds the write input of the left buffer (scaled by `feedback`), and vice versa.
- Use 4-point or linear fractional interpolation on the read pointer when `delay_time × fs` is non-integer (which it always is in practice).
