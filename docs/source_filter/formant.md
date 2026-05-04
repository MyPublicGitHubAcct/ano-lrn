# Formant Filter

## `formant_filter`

Each formant is modelled as a second-order resonator with a peak at frequency f₀ and –3 dB bandwidth bw:

```text
H_k(z) = (1 − R²) / (1 − 2R·cos(θ)·z⁻¹ + R²·z⁻²)
```

where θ = 2π·f₀/fs and R = exp(−π·bw/fs). The cascade of K resonators gives:

```text
H(z) = ∏ₖ H_k(z)
```

The approximate formant frequencies for the vowel /a/ in English are F1 ≈ 800 Hz, F2 ≈ 1200 Hz, F3 ≈ 2500 Hz with bandwidths of 80–120 Hz. Applying this filter to an impulse train produces a synthetic vowel.
