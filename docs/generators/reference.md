# Reference Generators

Fixed-frequency or constant signals with mathematically exact sample values. Useful as regression anchors and for probing specific points in the spectrum.

| Generator | Frequency | First sample | Use case |
| --- | --- | --- | --- |
| `generate_dc` | 0 Hz | +amplitude | DC rejection / pass-through testing |
| `generate_nyquist` | fs/2 | −1 | Low-pass stopband, high-pass passband edge |
| `generate_half_nyquist` | fs/4 | −1 | Midband filter gain check |
| `generate_quarter_nyquist` | fs/8 | −1 | Lower-quarter spectrum, oversampled paths |

All return `(t, signal)` — a time axis and signal array sampled at `fs` Hz over `duration` seconds.

---

## `generate_dc`

A constant signal at `amplitude`. Has energy only at 0 Hz.

```text
x[n] = A  for all n
```

**Use cases:** testing DC rejection (high-pass, band-pass filters must drive this to zero); testing DC pass-through (low-pass filters must preserve it at unity gain); verifying that an algorithm does not introduce or remove a DC offset.

---

## `generate_nyquist`

```text
x[n] = −cos(π n) = (−1)^(n+1)   →   −1, 1, −1, 1, …
```

A cosine at exactly **fs/2** — the highest frequency a discrete system can represent. Every sample is ±1 with no intermediate values. A low-pass filter at any cutoff below fs/2 must attenuate this signal; a high-pass filter near fs/2 must pass it.

**Use cases:** verifying low-pass stopband attenuation at the extreme edge; testing high-pass passband gain; checking for Nyquist-frequency aliasing artifacts.

---

## `generate_half_nyquist`

```text
x[n] = −cos(π n / 2)   →   −1, 0, 1, 0, −1, 0, 1, 0, …
```

A cosine at **fs/4** — halfway between DC and Nyquist. The zero crossings at every other sample make the period immediately visible: four samples per cycle.

**Use cases:** verifying filter gain at the midpoint of the spectrum; testing phase response at fs/4; a convenient sanity-check frequency for any filter whose cutoff is near fs/4.

---

## `generate_quarter_nyquist`

```text
x[n] = −cos(π n / 4)   →   −1, −√2/2, 0, √2/2, 1, √2/2, 0, −√2/2, …
```

A cosine at **fs/8** — one quarter of the Nyquist frequency. The eight-sample period and the ±√2/2 ≈ ±0.707 values at the ±45° points are a recognizable fingerprint in a sample dump.

**Use cases:** filter response measurements in the lower quarter of the spectrum; checking that a high-pass filter attenuates this signal relative to half-Nyquist; testing oversampled paths.
