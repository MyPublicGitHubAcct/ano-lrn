# Spectral Features

## `spectral_centroid`

The spectral centroid is the frequency-weighted mean of the magnitude spectrum:

```text
C[m] = Σ_k  f[k] · |S[k, m]|  /  Σ_k  |S[k, m]|
```

Returns an array of centroid values in Hz, one per STFT frame. High centroid = energy concentrated at high frequencies (bright/sharp timbre); low centroid = energy at low frequencies (dark/muffled timbre).

**Use cases:** timbre tracking, music genre classification, onset detection auxiliary feature.

---

## `spectral_flux`

Spectral flux measures the rate of change in the magnitude spectrum between consecutive frames:

```text
F[m] = ‖|S[:, m]| − |S[:, m−1]|‖₂
```

Returns an array of flux values (first frame is 0). Peaks in the flux signal correspond to onsets, note attacks, or abrupt timbral transitions. Spectral flux is one of the most effective onset detection features for percussive sounds.

**Rectified flux:** using only positive differences (half-wave rectification) increases onset detection precision by ignoring decays: F⁺[m] = ‖max(|S[:, m]| − |S[:, m−1]|, 0)‖₂.
