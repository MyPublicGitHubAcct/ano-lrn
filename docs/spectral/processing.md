# Spectral Processing

## `spectral_gate`

A frequency-domain noise gate: STFT bins whose magnitude is below `threshold_db` are zeroed, then the result is resynthesised via ISTFT:

```text
M[k, m] = 1 if 20·log10(|S[k, m]|) ≥ threshold_db, else 0
ŷ       = ISTFT(M · S)
```

**Note on threshold calibration:** raw STFT bin magnitudes scale with `frame_size`. For frame_size = 2048, a full-scale sine has a peak magnitude of approximately 54 dB; set the threshold accordingly (e.g., 40 dB to suppress bins more than ~14 dB below the peak).

**Use cases:** static noise removal, spectral subtraction pre-processing, source isolation.
