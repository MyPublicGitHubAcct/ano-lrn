# Harmonic-Percussive Source Separation

## `hpss`

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

**Returns:** `(harmonic, percussive)` — two time-domain signals.
