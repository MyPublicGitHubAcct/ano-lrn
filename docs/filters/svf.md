# Chamberlin State-Variable Filter (SVF)

## `svf`

A two-integrator resonant filter that produces lowpass, bandpass, highpass, and notch outputs simultaneously in a single sample-by-sample pass. Cutoff and resonance are simple scalar multipliers in the update loop, so both can be changed every sample without instability — making this topology ideal for VCV Rack patches where LFOs or envelopes modulate filter cutoff at audio rate.

| Parameter | Range | Effect |
| --- | --- | --- |
| `cutoff` | 20–fs/4 Hz (clamped) | Centre frequency of all four output ports |
| `resonance` | 0–1 | 0 = Butterworth (q = √2), approaching 1 = self-oscillation (q → 0) |
| `mode` | `'lp'`, `'bp'`, `'hp'`, `'notch'`, `'all'` | Selects output port; `'all'` returns `(lp, bp, hp)` tuple |

---

### Topology

Zölzer/Chamberlin form — two first-order integrators in a feedback loop with a damping path:

```text
         ┌─────────────────────────────────┐
x ─(−)──▶ hp ──×f──▶ [+]──▶ bp ──×f──▶ [+]──▶ lp
    ▲           │     ▲            │     ▲
    │  (−q·bp)──┘     └── bp ─────┘     └── lp ─┐
    └────────────────────────────────────────────┘
                          notch = lp + hp
```

Per-sample update (Zölzer DAFX, 2nd ed., §2.2):

```text
f  = 2 sin(π · cutoff / fs)        # frequency coefficient
q  = √2 · (1 − resonance)          # damping coefficient

lp[n] = lp[n−1] + f · bp[n−1]
hp[n] = x[n] − lp[n] − q · bp[n−1]
bp[n] = f · hp[n] + bp[n−1]
notch[n] = lp[n] + hp[n]
```

---

### Stability and the frequency clamp

The filter is stable when `f(f + 2q) < 4`, which sets an upper cutoff limit of approximately `fc_max = fs · arcsin((√(q²+4) − q) / 2) / π`. At `resonance = 0` (maximum damping, `q = √2`) this limit is roughly `fs / 5.8`. The implementation clamps `f` to `√(q²+4) − q` so the filter cannot diverge regardless of the `cutoff` argument, but frequency accuracy degrades above the stability boundary. The ZDF SVF (`zdf_svf`) does not have this limitation.

---

### Frequency response

| Mode | Slope below cutoff | Slope above cutoff | At cutoff |
| --- | --- | --- | --- |
| LP | 0 dB (passband) | −12 dB/octave | −3 dB (Butterworth at resonance=0) |
| BP | +12 dB/octave | −12 dB/octave | Peak (height grows with resonance) |
| HP | −12 dB/octave | 0 dB (passband) | −3 dB (Butterworth at resonance=0) |
| Notch | 0 dB | 0 dB | Deep null |

---

### Resonance mapping

`q = √2 · (1 − resonance)` maps the 0–1 control range to the damping coefficient:

| `resonance` | `q` | `Q = 1/q` | Character |
| --- | --- | --- | --- |
| 0.0 | √2 ≈ 1.414 | 0.707 | Butterworth — maximally flat passband |
| 0.29 | 1.0 | 1.0 | Slight resonant peak |
| 0.5 | √2/2 ≈ 0.707 | √2 ≈ 1.41 | Audible resonance |
| 0.9 | √2/10 ≈ 0.141 | ~7 | Strongly resonant |
| → 1 | → 0 | → ∞ | Self-oscillation |

---

### Comparison with biquad LP/BP/HP

| Property | `lowpass` / `highpass` / `bandpass` | `svf` |
| --- | --- | --- |
| Outputs per pass | 1 | LP + BP + HP + notch simultaneously |
| Per-sample parameter change | Coefficient recalculation needed | Safe — f and q are loop scalars |
| High-cutoff accuracy | Good | Degrades above ~fs/6 (see `zdf_svf` for fix) |
| Self-oscillation | No | Yes (resonance → 1) |
| Implementation | `scipy.signal.lfilter` | Sample-by-sample Python loop |

---

## `zdf_svf`

Zero-delay-feedback (ZDF) state-variable filter using the Zavalishin topology-preserving transform (TPT). Replaces each integrator's unit delay with a trapezoidal integrator solved by simultaneous substitution, eliminating the one-sample delay in the feedback path. The result is exact cutoff accuracy across the full audio band — the −3 dB point is at exactly `cutoff` Hz even at fc/fs = 0.4, where the Chamberlin SVF loses 15–20% frequency accuracy.

| Parameter | Range | Effect |
| --- | --- | --- |
| `cutoff` | 1 Hz – fs/2 | Exact −3 dB frequency (bilinear pre-warped) |
| `resonance` | > 0 (default 1/√2) | Q factor: 1/√2 = Butterworth, higher = resonant peak |
| `mode` | `'lp'`, `'bp'`, `'hp'`, `'notch'`, `'all'` | Selects output port; `'all'` returns `(lp, bp, hp)` tuple |

---

### Theory: why the Chamberlin SVF has frequency error

#### The continuous-time prototype

The SVF models two cascaded lossless integrators (capacitors in an analog circuit) with a damping feedback path. In continuous time, with normalised cutoff ωc = 1 and damping k = 1/Q:

```text
HP = x − k·BP − LP
BP = (1/s)·HP        ← integrator 1
LP = (1/s)·BP        ← integrator 2
```

Substituting and solving gives the transfer functions H_LP = 1/(s²+ks+1), H_BP = s/(s²+ks+1), H_HP = s²/(s²+ks+1) — a standard 2-pole resonant filter with poles whose natural frequency and damping are set directly by the two parameters.

#### The unit-delay problem in the Chamberlin topology

The Chamberlin update computes lp from the previous bp, then hp from the new lp and the previous bp, then bp from the new hp:

```text
lp[n] = lp[n−1] + f · bp[n−1]       # uses bp from sample n−1
hp[n] = x[n] − lp[n] − q · bp[n−1]  # uses bp from sample n−1
bp[n] = f · hp[n] + bp[n−1]
```

Notice that hp[n] is computed using `bp[n−1]` — a one-sample-old value. In the z-domain this is a `z⁻¹` factor inside the resonance feedback loop. This delay shifts the effective cutoff and introduces a phase error that grows with fc/fs.

At fc = 100 Hz and fs = 44100 Hz the delay is one sample ≈ 22.7 μs, far shorter than one period at 100 Hz (10 ms) — negligible. At fc = 17640 Hz (fc/fs = 0.4), that same 22.7 μs is an appreciable fraction of one period (56.7 μs), producing roughly 15–20% upward frequency drift. This is also why the Chamberlin SVF needs a stability clamp: the embedded `z⁻¹` creates a feedback loop that can go unstable at high cutoffs.

#### Forward Euler vs trapezoidal integration

The Chamberlin integrator approximates `1/s` by the forward Euler rule:

```text
y[n] = y[n−1] + T · x[n−1]     (z-domain: 1/s ≈ T·z⁻¹/(z−1))
```

This maps each analog pole at `s = jΩ` to the digital location `z = 1 + jΩT`, which lies outside the unit circle for all Ω > 0. The approximation `f = 2·sin(π·fc/fs)` in the Chamberlin update is a correction that pulls those poles back toward the unit circle, but it is only accurate for small fc/fs.

The trapezoidal (bilinear) rule replaces `1/s` exactly:

```text
1/s → (T/2) · (z+1)/(z−1)
```

The bilinear transform is a conformal map from the left half of the s-plane onto the interior of the unit circle. Every stable analog pole — regardless of its frequency — maps to a point strictly inside the unit circle. The digital filter therefore inherits the stability of its analog prototype unconditionally, for any fc in (0, fs/2) and any Q > 0.

#### Why g = tan(π·fc/fs) gives the exact digital cutoff

The bilinear transform compresses the entire analog frequency axis (0 to ∞) onto the digital axis (0 to π/T). The mapping is nonlinear:

```text
Ω_analog = (2/T) · tan(ω_digital · T/2)
```

To place the digital cutoff at exactly fc Hz, we pre-warp the analog prototype's cutoff to compensate for this compression. Substituting ω = 2π·fc/fs = π·fc/fs · 2:

```text
g = Ω_analog · T/2 = tan(π · fc / fs)
```

This is the integration coefficient in `zdf_svf`. The pre-warp guarantees that whatever the bilinear transform does to the surrounding frequencies, the −3 dB point lands precisely at fc.

The Chamberlin coefficient `f = 2·sin(π·fc/fs)` approximates `2·tan(π·fc/fs)` only for small arguments (both approach 2π·fc/fs as fc → 0). At fc/fs = 0.4 the values diverge sharply:

```text
2·sin(0.4π) ≈ 1.902
2·tan(0.4π) ≈ 6.155    (actual integration gain needed)
```

The Chamberlin filter uses less than one-third of the required integration gain at that frequency, which is what causes the 15–20% cutoff error.

#### Solving the equations with zero delay (the TPT step)

"Zero-delay feedback" means computing all three outputs — HP, BP, LP — for sample n using only current-sample quantities, with no `z⁻¹` in the feedback path. In the TPT form, each trapezoidal integrator is written as:

```text
output = g · input + state
state  ← 2 · output − state
```

where `state` carries the integrator memory from the previous sample. Writing this for both integrators simultaneously:

```text
BP = g · HP + s1     (BP integrator, state s1)
LP = g · BP + s2     (LP integrator, state s2)
HP = x − k · BP − LP
```

These three equations are linear in the unknowns HP, BP, LP — all at sample n. Substituting BP and LP into the HP equation and collecting terms:

```text
HP + k·g·HP + g²·HP = x − (k+g)·s1 − s2
HP · (1 + k·g + g²) = x − (k+g)·s1 − s2
```

This single equation has HP as the only unknown. Defining `a1 = 1/(1+g·(g+k))` (the inverse denominator, precomputed once per parameter change), `a2 = g·a1`, `a3 = g·a2`, the per-sample update becomes a handful of multiply-adds with no divisions in the loop. The integrator states s1 and s2 update as `s1 ← 2·BP − s1`, `s2 ← 2·LP − s2` — the standard trapezoidal memory step.

Nothing on the right-hand side of any equation belongs to the current sample's outputs. s1 and s2 are memory, not feedback. The resonance path from LP and BP back to HP is resolved algebraically within the same sample — hence "zero delay feedback."

#### Stability

Because the bilinear transform maps the left half-plane to the interior of the unit circle, and the analog SVF is stable for all k > 0, the ZDF SVF is unconditionally stable for any fc ∈ (0, fs/2) and any Q > 0. No clamp on the cutoff argument is needed. The Chamberlin SVF requires the clamp because the embedded `z⁻¹` in the feedback loop creates a region of instability that grows with fc/fs.

#### When to use each

Use `svf` when cutoff stays well below fs/8 and the 0–1 normalised resonance knob is convenient. Use `zdf_svf` when cutoff is modulated at audio rate (LFOs, envelopes sweeping to high frequencies), when an exact −3 dB point at cutoff is required (crossover networks), or when porting to C++ — the ZDF arithmetic is equally simple and has no accuracy or stability caveats.

---

### TPT derivation

The continuous-time SVF integrators `1/s` are discretised by the trapezoidal rule `1/s → (T/2)(z+1)/(z−1)`. In the TPT form each integrator becomes:

```text
output = g · input + state
state  ← 2 · output − state     (i.e. state ← state + 2g · input)
```

where `g = tan(π · cutoff / fs)` — the bilinear pre-warp factor that maps the analogue cutoff ωc exactly to the digital bin at `cutoff` Hz.

Solving the two simultaneous integrator equations for the current sample yields the closed-form per-sample update:

```text
g  = tan(π · cutoff / fs)
k  = 1 / Q                      # damping; k = √2 for Butterworth
a1 = 1 / (1 + g · (g + k))
a2 = g · a1
a3 = g · a2

v3 = x − s2
v1 = a1 · s1 + a2 · v3         # BP output
v2 = s2 + a2 · s1 + a3 · v3   # LP output
s1 ← 2 · v1 − s1
s2 ← 2 · v2 − s2
hp = x − k · v1 − v2           # HP = input − k·BP − LP
```

The KVL identity `LP + k·BP + HP = x` holds sample-by-sample (verified in tests).

The analogous Chamberlin identity is `lp[n] + q·bp[n−1] + hp[n] = x[n]` — it uses the *previous* bp sample, not the current one, because `hp[n]` is computed before `bp[n]` is updated. See the Chamberlin topology section above.

---

### Output port characteristics

Same LP/BP/HP/notch shapes as `svf`, but without the frequency warping that degrades the Chamberlin SVF above fs/4:

| Mode | At cutoff | Above 4×cutoff | Below cutoff/10 |
| --- | --- | --- | --- |
| LP | −3 dB (exact for Butterworth Q) | −48 dB/octave attenuation | — |
| HP | −3 dB (exact) | — | ≈ −40 dB |
| BP | Peak (exact at cutoff) | −12 dB/octave | −12 dB/octave |

---

### Resonance (Q factor)

`resonance` is the Q factor directly (unlike `svf` which uses a 0–1 normalised mapping):

| `resonance` | `k = 1/Q` | Character |
| --- | --- | --- |
| 1/√2 ≈ 0.707 | √2 | Butterworth — maximally flat passband (default) |
| 1.0 | 1.0 | Q=1 — slight resonant peak |
| 2.0 | 0.5 | Audible resonance peak |
| 10.0 | 0.1 | Strongly resonant |
| → ∞ | → 0 | Self-oscillation |

---

### Comparison: `svf` vs `zdf_svf`

| Property | `svf` (Chamberlin) | `zdf_svf` (TPT/ZDF) |
| --- | --- | --- |
| Frequency coefficient | `f = 2 sin(π·fc/fs)` | `g = tan(π·fc/fs)` |
| −3 dB point accuracy | Accurate below ~fs/8; drifts above | Exact up to fs/2 |
| Stability range | `f(f+2q) < 4` (~fs/6 at Q=√2) | Stable for all fc < fs/2 |
| Resonance parameter | Normalised 0–1 | Q factor directly |
| VCV Rack preference | Prototype/low-fc use | Production — preferred for C++ port |
