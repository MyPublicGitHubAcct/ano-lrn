# ADSR Envelope Generator

An ADSR envelope shapes the amplitude of a signal over time using four segments: **Attack**, **Decay**, **Sustain**, and **Release**.

```
1.0 ─────┐                          ← peak
         │\                         
sustain ─│ ─────────────────┐       ← sustain level held
         │                   \      
0.0 ─────┘─────────────────────────
         A    D        S      R
```

---

## `generate_adsr`

```python
generate_adsr(attack, decay, sustain, release, fs, duration, curve="linear")
    -> (t, envelope)
```

| Parameter | Description |
| --- | --- |
| `attack` | Rise time from 0 to 1.0 (seconds) |
| `decay` | Fall time from 1.0 to `sustain` level (seconds) |
| `sustain` | Level held after decay (0–1) |
| `release` | Fall time from `sustain` to 0.0 (seconds) |
| `fs` | Sample rate (Hz) |
| `duration` | Total output length (seconds) |
| `curve` | `"linear"` (default) or `"exponential"` |

Returns `(t, envelope)` — a time axis and amplitude envelope, both of length `int(fs × duration)`.

---

## Segment equations

### Attack (always linear)

```
env[n] = n / n_a,     0 ≤ n < n_a
```

where `n_a = int(attack × fs)`. The last attack sample equals **1.0** exactly.

### Decay

**Linear:**

```
env[n] = 1 + (sustain − 1) · n / (n_d − 1),     n_a ≤ n < n_a + n_d
```

**Exponential** (`curve="exponential"`):

```
u[n] = n / (n_d − 1)
c[n] = (1 − exp(−k·u[n])) / (1 − exp(−k)),     k = 5
env[n] = 1 + (sustain − 1) · c[n]
```

The factor `k = 5` gives roughly 5 time constants over the segment; `c[0] = 0` and `c[n_d−1] = 1` exactly, so both endpoints are guaranteed.

### Sustain

```
env[n] = sustain,     n_a + n_d ≤ n < N − n_r
```

### Release

**Linear:**

```
env[n] = sustain · (1 − n / (n_r − 1)),     N − n_r ≤ n < N
```

**Exponential** — same normalized curve as decay, mapped from `sustain` to `0`:

```
env[n] = sustain · (1 − c[n])
```

The last release sample equals **0.0** exactly for both curve types.

---

## Segment lengths

```
n_a = min(int(attack  × fs), N)
n_d = min(int(decay   × fs), N − n_a)
n_r = min(int(release × fs), N − n_a − n_d)
n_s = N − n_a − n_d − n_r
```

Clamping prevents any segment from overrunning the buffer. If `attack + decay + release > duration`, the sustain segment is dropped (`n_s = 0`).

---

## Linear vs exponential curves

The exponential shape uses a normalized `(1 − e^{−kt})` curve that hits both endpoints exactly:

```
c(t) = (1 − e^{−kt}) / (1 − e^{−k}),     t ∈ [0, 1]
```

- At `t = 0`: `c = 0`
- At `t = 1`: `c = 1`

For decay (falling from 1 to sustain), larger `k` produces a faster initial fall and a longer tail — closer to the behaviour of a capacitor discharging through a resistor. `k = 5` is a reasonable default that sounds natural for percussive sounds.

---

## Use cases

| Shape | attack | decay | sustain | release | Character |
| --- | --- | --- | --- | --- | --- |
| Pluck | short | long | 0 | short | Guitar, harpsichord |
| Pad | long | medium | 0.8 | long | Strings, choir |
| Lead | short | short | 0.7 | medium | Synth lead |
| Percussion | 0 | medium | 0 | short | Drum, bell |

The envelope output is designed to be multiplied sample-by-sample with any audio signal:

```python
t, env = generate_adsr(attack=0.02, decay=0.1, sustain=0.7, release=0.3,
                       fs=44100, duration=1.0)
t, sine = generate_sine(freq=440, fs=44100, duration=1.0)
output = sine * env
```
