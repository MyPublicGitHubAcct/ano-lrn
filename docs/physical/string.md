# Karplus-Strong Plucked String

## Algorithm

The Karplus-Strong algorithm models a vibrating string as a recirculating delay line excited by a burst of noise:

1. Fill a ring buffer of length `D = round(fs / freq)` with white noise.
2. For each output sample, apply a one-pole averaging loss filter:

```
y[n] = damping * 0.5 * (y[n-D] + y[n-D-1])
```

3. Feed `y[n]` back into the delay buffer, replacing the sample from `D` steps ago.

The averaging (`* 0.5`) is a lowpass filter with a zero at Nyquist, so high frequencies decay faster than low frequencies — matching the behaviour of a real string where stiffness damps upper partials first. The `damping` coefficient scales the overall energy loss per cycle; values close to 1 produce long sustain, lower values decay quickly.

## Delay Line Length and Pitch

The fundamental period is determined by the buffer length `D`:

```
f0 ≈ fs / D
```

Because `D` is an integer, the actual pitch is `fs / round(fs / freq)`, which may differ slightly from the target. At `fs = 44100` and `freq = 440 Hz`, `D = 100` gives an exact 441 Hz; for precise tuning at arbitrary pitches, a fractional-delay tail is needed (see `physical/waveguide.py`).

## Pickup Position

The `pickup` parameter (0–1) selects a read position at `round(pickup * D)` samples behind the write pointer, simulating a pickup mounted at a fraction of the string length from the bridge.

A pickup at position `p` from the bridge implements a comb filter whose zeros fall at harmonics `k` satisfying:

```
sin(π · k · p) = 0   →   k = 1/p, 2/p, 3/p, …
```

At `pickup = 0.5` (string midpoint), even harmonics (2, 4, 6, …) are at nodes and are cancelled — the output sounds warm and round, like a guitar neck pickup. At `pickup ≈ 0.1` (near the bridge), the comb zeros fall outside the audible range, so all harmonics are present — the output is bright and nasal, like a bridge pickup.

## Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `freq`    | —       | > 0 Hz | Fundamental frequency |
| `fs`      | —       | Hz     | Sample rate |
| `duration`| —       | s      | Output length |
| `damping` | 0.99    | 0–1    | Loss per cycle; lower = faster decay |
| `pickup`  | 0.1     | 0–1    | Fractional read position along delay line |
| `seed`    | 0       | int    | RNG seed for reproducible excitation |

## Relation to Infrastructure

`pluck` builds on:
- `delay/line.py` — the ring buffer concept (implemented inline here for performance)
- `generators/_helpers.py` — `_time_axis` for the output time vector

The `physical/waveguide.py` module extends this model with bidirectional delay lines and fractional-delay tuning for exact pitch accuracy at any frequency.

## References

- Karplus, K. & Strong, A. (1983). *Digital Synthesis of Plucked-String and Drum Timbres*. Computer Music Journal, 7(2), 43–55.
- Jaffe, D. & Smith, J. O. (1983). *Extensions of the Karplus-Strong Plucked-String Algorithm*. Computer Music Journal, 7(2), 56–69.
