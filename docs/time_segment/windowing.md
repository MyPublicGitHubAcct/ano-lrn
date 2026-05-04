# Windowing

## `apply_window`

Multiplies a signal element-wise by a smooth window function:

```text
y[n] = x[n] · w[n]
```

Supported window types: `'hann'`, `'hamming'`, `'blackman'`, `'rectangular'`.

Windows are used before FFT analysis to reduce spectral leakage. A rectangular window has maximum leakage (sharp edges cause sinc-shaped sidelobes); the Hann window reduces peak sidelobe levels to ~−32 dB; Blackman reduces them further to ~−58 dB at the cost of a wider main lobe.

**Hann window:** w[n] = 0.5 · (1 − cos(2π·n/(N−1)))

The Hann window is zero at both endpoints, making it ideal for overlap-add synthesis with 50% or 75% overlap.
