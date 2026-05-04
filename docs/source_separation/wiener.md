# Wiener Filter

## `wiener_filter`

The Wiener filter is an optimal linear estimator in the minimum mean-squared-error sense. Given a mixture M and a spectral estimate E of the target source, the frequency-domain mask is:

```text
mask[k, m] = |E[k, m]|² / |M[k, m]|²     (clamped to 1)
output     = ISTFT(mask · M)
```

This weights each time-frequency bin by the estimated signal-to-noise ratio (the ratio of source power to mixture power). Bins where the estimate dominates the mixture pass at near-unity gain; bins where the estimate is weak are suppressed.

**Use cases:** speech enhancement (estimate = a clean speech reference), vocal extraction (estimate = a rough harmonic signal), denoising (estimate = noise-reduced approximation of the target).
