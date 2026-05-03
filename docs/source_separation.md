# Source Separation

`src/python/source_separation.py` implements two STFT-domain source separation algorithms.

## Summary

| Function | Key parameters | Returns | Description |
| --- | --- | --- | --- |
| `hpss` | `signal`, `fs`, `frame_size`, `hop_size`, `kernel_size` | `(harmonic, percussive)` | Harmonic-percussive separation via median filtering |
| `wiener_filter` | `mixture`, `source_estimate`, `frame_size`, `hop_size` | signal | Wiener mask source extraction |

---

### `hpss`

Harmonic-percussive source separation (Fitzgerald 2010) exploits the structural difference between harmonic and percussive components in the spectrogram:

- **Harmonic components** (sustained tones, melody) appear as horizontal ridges (constant frequency over time).
- **Percussive components** (drums, attacks) appear as vertical ridges (instantaneous energy at all frequencies).

**Algorithm:**

1. Compute the STFT magnitude M[k, m].
2. Apply a median filter along the time axis (width = `kernel_size`) to extract the harmonic estimate H[k, m].
3. Apply a median filter along the frequency axis (width = `kernel_size`) to extract the percussive estimate P[k, m].
4. Compute Wiener-style soft masks:

```text
mask_H[k, m] = H²[k, m] / (H²[k, m] + P²[k, m])
mask_P[k, m] = P²[k, m] / (H²[k, m] + P²[k, m])
```

5. Apply masks to the original STFT and invert via ISTFT.

The soft masks satisfy mask_H + mask_P = 1 everywhere, so the two components sum to approximately the original signal.

**kernel_size:** larger values give stronger separation but may introduce artefacts. Typical values: 17–31 for 44100 Hz, 2048-point STFT.

---

### `wiener_filter`

The Wiener filter is an optimal linear estimator in the minimum mean-squared-error sense. Given a mixture M and a spectral estimate E of the target source, the frequency-domain mask is:

```text
mask[k, m] = |E[k, m]|² / |M[k, m]|²     (clamped to 1)
output     = ISTFT(mask · M)
```

This weights each time-frequency bin by the estimated signal-to-noise ratio (the ratio of source power to mixture power). Bins where the estimate dominates the mixture pass at near-unity gain; bins where the estimate is weak are suppressed.

**Use cases:** speech enhancement (estimate = a clean speech reference), vocal extraction (estimate = a rough harmonic signal), denoising (estimate = noise-reduced approximation of the target).
