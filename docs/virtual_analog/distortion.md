# Virtual Analog Distortion

## `diode_clip`

Models asymmetric clipping characteristic of a diode pair:

- **Positive half:** hard clip at `+threshold` (ideal diode forward conduction).
- **Negative half:** soft clip via tanh, modelling the diode's gradual turn-on region.

```text
y[n] = threshold                              if x[n] > threshold
y[n] = x[n]                                  if 0 ≤ x[n] ≤ threshold
y[n] = −threshold · tanh(−x[n] / threshold)  if x[n] < 0
```

The asymmetry generates both even and odd harmonics, giving a character similar to BJT transistor or diode-based guitar overdrive circuits.

**Degenerate case:** `threshold=0` divides by zero in the negative-half formula. NumPy evaluates `0 · tanh(±∞)` as 0, so the output is all zeros (finite) rather than NaN — but the behaviour is implementation-defined. Avoid passing `threshold=0` in production use.

---

## `analog_saturate`

Third-order polynomial approximation to a triode vacuum tube transfer curve:

```text
x_d = clip(x[n] · drive, −1, 1)
y[n] = x_d − x_d³ / 3
```

The input is first driven by `drive` and then clamped to avoid the polynomial growing unbounded. The cubic term introduces only odd harmonics (3rd harmonic dominant). The derivative dy/dx = 1 − x² ≥ 0 everywhere on [−1, 1], so the transfer function is monotone (no foldback distortion).

**Comparison to tanh:** the polynomial model saturates more abruptly than tanh, matching the sharp knee of a triode running into grid-current clipping.

---

## `wavefold`

Wavefolder: instead of clipping a signal when it exceeds ±1, the excess is reflected back (folded) into range. The operation is equivalent to evaluating a triangle wave whose input is the gain-scaled signal:

```text
x[n] = signal[n] · gain
m    = (x[n] + 1) mod 4          # period-4 sawtooth centred on 0
y[n] = m − 1     if m < 2        # rising half: 0 → 1
y[n] = 3 − m     if m ≥ 2        # falling half: 1 → −1
```

This is identical to the iterative reflection rule "while |x| > 1, reflect at the boundary" but avoids a Python loop by computing the periodic fold algebraically in one vectorised step.

**Gain and harmonic content:** a unit-amplitude sine with `gain = 1` just touches ±1 (no folding). At `gain = 2` the waveform folds once per half-cycle, producing odd and even harmonics. At `gain = 3` it folds twice, adding yet more partials. Unlike clipping — which compresses the peak and adds harmonics that decay in amplitude — wavefolding adds a new spectral component each time a fold boundary is crossed, giving a richer and more even distribution of upper partials at moderate fold counts.

**Odd symmetry:** the transfer function satisfies f(−x) = −f(x) for all x and any gain value. This means a pure sine input produces only odd harmonics when the gain is such that the waveform folds an even number of times per half-cycle (i.e. lands symmetrically). In general, odd and even harmonics are both present.

**Buchla context:** wavefolding is the defining nonlinearity of the Buchla 259 Complex Waveform Generator and closely related to the Serge Wave Multiplier. It is directly relevant to the VCV Rack VA module set.
