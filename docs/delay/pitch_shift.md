# Time-Domain Pitch Shifter

`pitch_shift(signal, fs, semitones)` shifts the pitch of a monophonic signal by
a given number of semitones without changing its duration.

---

## Algorithm

### Circular Buffer and Two Read Heads

A circular delay buffer of length `B` (next power of 2 ≥ 50 ms at `fs`) is
filled by a write head advancing at one sample per step.  Two read heads
traverse the buffer; the difference between a head's position and the write
position is called the **grain pointer** `p`, representing the delay (age in
samples) of the sample being read.

The grain pointer advances at `dp = 1 − r` per step, where `r = 2^(semitones/12)`:

| Shift | r | dp | Effect |
|---|---|---|---|
| Up (semitones > 0) | > 1 | < 0 | Delay shrinks; read head catches write head |
| None (semitones = 0) | 1 | 0 | Delay constant; fixed passthrough with B/2 latency |
| Down (semitones < 0) | < 1 | > 0 | Delay grows; read head falls behind write head |

When `p` is constant, the output reads a fixed-age sample and passes through
unchanged.  When `p` decreases (pitch up), the read head catches up to the
write head, effectively reading the input faster — raising the pitch.

### Crossfade via Hann Partition of Unity

Each read head carries a Hann window weight based on its current grain pointer:

```
w(p) = 0.5 · (1 − cos(2π · p / B))
```

Head 1's grain pointer is always B/2 ahead of head 0's:

```
p₁ = (p₀ + B/2) mod B
```

This stagger gives a partition-of-unity property at every sample:

```
w(p₀) + w(p₁) = 0.5(1 − cos θ) + 0.5(1 + cos θ) = 1
```

so the total output amplitude is preserved regardless of where the two heads
are in their respective cycles.  When a head's grain pointer wraps at `p = 0`
or `p = B` (where `w = 0`), it is silent, and the other head (at `p = B/2`,
where `w = 1`) carries full weight — providing a seamless crossfade at every
grain boundary.

### Read-Pointer Speed and Pitch Ratio

The output at step `n` is:

```
y[n] = w(p₀[n]) · x[n − p₀[n]] + w(p₁[n]) · x[n − p₁[n]]
```

Within a single grain (between wraps), `p₀[n] = p₀[0] + dp · n`, so:

```
n − p₀[n] = n − p₀[0] − dp · n = n(1 − dp) + const = r · n + const
```

The read position advances at rate `r` per output step, so the output pitch is
`r × f_input` — the desired pitch ratio.

### Grain Period and Crossfade Rate

The grain pointer takes `B / |dp|` steps to traverse the full buffer:

| Shift | r | Grain period (smp) | At 44100 Hz |
|---|---|---|---|
| +12 st | 2.00 | B / 1.00 = B | 93 ms |
| +7 st | 1.498 | B / 0.498 ≈ 2B | 186 ms |
| −7 st | 0.667 | B / 0.333 ≈ 3B | 279 ms |
| −12 st | 0.50 | B / 0.50 = 2B | 186 ms |

Larger shifts → shorter grain period → more frequent crossfades → more
grain-boundary artefacts.

---

## Startup Latency

Head 0 starts at `p₀ = B/2` (maximum Hann weight).  Its read position at
`n = 0` is `0 − B/2 < 0` — before the start of the signal.  The implementation
uses left-zero-padding of width `B`, so the first `B/2` output samples are
silence.  After that, for `semitones = 0`, `out[B//2:]` equals `signal[:N−B//2]`
to floating-point precision.

---

## Implementation

The vectorised implementation pre-computes grain pointers for all `N` steps
at once and uses NumPy fancy indexing with linear interpolation for sub-sample
accuracy:

```python
n     = np.arange(N, dtype=float)
p0    = (B/2 + (1 - r) * n) % B
p1    = (p0 + B/2) % B
win0  = 0.5 * (1 - np.cos(2π * p0 / B))
win1  = 0.5 * (1 - np.cos(2π * p1 / B))
src0  = n - p0      # fractional sample index in the original signal
...
out   = win0 * s0 + win1 * s1
```

Because there is no recirculation (the pitch shifter is feed-forward), the
entire computation is data-parallel: no sample-by-sample loop is required.

---

## Parameter Reference

| Parameter | Notes |
|---|---|
| `semitones` | Positive = up, negative = down; fractional values (e.g. `0.5`) are valid |
| `fs` | Used only to compute `B = next power of 2 ≥ fs × 0.05` |

---

## Limitations

- **Phase coherence**: At each grain wrap the read position resets by `B`
  samples.  For a pure tone input the phase jump is `2π · f₀ · B / fs` radians.
  When this is not a multiple of `2π`, a low-level sideband appears near the
  fundamental.  This is inherent to the two-head crossfade algorithm and is
  inaudible at musical amplitudes.
- **Polyphony**: The algorithm is monophonic.  Polyphonic pitch shifting
  requires a phase vocoder or per-channel subband processing.
- **Large shifts**: Shifts beyond ±12 semitones produce shorter grain periods
  and more frequent crossfades, increasing artefact density.
