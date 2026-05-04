# ZDF Moog Ladder Filter

## `moog_ladder`

Emulates the transistor-ladder filter in the Minimoog synthesizer — a 4-pole (24 dB/octave) lowpass with voltage-controlled resonance, including self-oscillation at maximum resonance.

| Parameter | Range | Effect |
| --- | --- | --- |
| `cutoff` | 20–20000 Hz | −3 dB point of each individual 1-pole stage |
| `resonance` | 0–1 | Feedback coefficient k = resonance × 4 |

---

### Topology

Four identical first-order ZDF (zero-delay feedback) lowpass stages are chained in series, with the output of the final stage fed back to the input:

```text
x ──(−k·y4)──▶ [stage 1] ──▶ [stage 2] ──▶ [stage 3] ──▶ [stage 4] ──▶ y4
       ▲                                                          │
       └──────────────────────────────────────────────────────────┘
```

Using the Huovilainen tanh-based approximation, each stage update is:

```text
g = 1 − exp(−2π · fc)
x_fb = x[n] − k · s[3]
s[0] += g · (tanh(x_fb) − tanh(s[0]))
s[1] += g · (tanh(s[0]) − tanh(s[1]))
s[2] += g · (tanh(s[1]) − tanh(s[2]))
s[3] += g · (tanh(s[2]) − tanh(s[3]))
y[n] = s[3]
```

The tanh nonlinearity in each stage limits resonance amplitude and produces the characteristic Moog saturation. k = 4·resonance; self-oscillation occurs at k ≥ 4.

---

### Frequency response

- **H(0)** ≈ 1 at resonance = 0; decreases with resonance due to feedback
- **Roll-off**: −80 dB/decade (−24 dB/octave) — the defining characteristic of a 4-pole filter
- **Resonant peak**: rises from −12 dB at k = 0 to self-oscillation at k = 4

---

### Comparison with biquad lowpass

| Property | `lowpass` (biquad) | `moog_ladder` |
| --- | --- | --- |
| Order | 2nd | 4th |
| Roll-off | −40 dB/decade | −80 dB/decade |
| Resonance control | Q factor | k = resonance × 4 |
| Self-oscillation | No | Yes (at resonance = 1) |
| Implementation | `scipy.signal.lfilter` | Sample-by-sample loop |
