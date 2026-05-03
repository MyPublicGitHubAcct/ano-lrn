# Spectral

`src/python/spectral.py` provides frequency-domain feature extraction and spectral-domain noise suppression.

## Summary

| Function | Key parameters | Returns | Description |
| --- | --- | --- | --- |
| `spectral_centroid` | `signal`, `fs`, `frame_size`, `hop_size` | Hz array | Frequency-weighted mean per frame |
| `spectral_flux` | `signal`, `frame_size`, `hop_size` | array | Frame-to-frame magnitude change |
| `spectral_gate` | `signal`, `threshold_db`, `frame_size`, `hop_size` | signal | Zero spectral bins below threshold |

---

### `spectral_centroid`

The spectral centroid is the frequency-weighted mean of the magnitude spectrum:

```text
C[m] = Σ_k  f[k] · |S[k, m]|  /  Σ_k  |S[k, m]|
```

Returns an array of centroid values in Hz, one per STFT frame. High centroid = energy concentrated at high frequencies (bright/sharp timbre); low centroid = energy at low frequencies (dark/muffled timbre).

**Use cases:** timbre tracking, music genre classification, onset detection auxiliary feature.

---

### `spectral_flux`

Spectral flux measures the rate of change in the magnitude spectrum between consecutive frames:

```text
F[m] = ‖|S[:, m]| − |S[:, m−1]|‖₂
```

Returns an array of flux values (first frame is 0). Peaks in the flux signal correspond to onsets, note attacks, or abrupt timbral transitions. Spectral flux is one of the most effective onset detection features for percussive sounds.

**Rectified flux:** using only positive differences (half-wave rectification) increases onset detection precision by ignoring decays: F⁺[m] = ‖max(|S[:, m]| − |S[:, m−1]|, 0)‖₂.

---

### `spectral_gate`

A frequency-domain noise gate: STFT bins whose magnitude is below `threshold_db` are zeroed, then the result is resynthesised via ISTFT:

```text
M[k, m] = 1 if 20·log10(|S[k, m]|) ≥ threshold_db, else 0
ŷ       = ISTFT(M · S)
```

**Note on threshold calibration:** raw STFT bin magnitudes scale with `frame_size`. For frame_size = 2048, a full-scale sine has a peak magnitude of approximately 54 dB; set the threshold accordingly (e.g., 40 dB to suppress bins more than ~14 dB below the peak).

**Use cases:** static noise removal, spectral subtraction pre-processing, source isolation.
