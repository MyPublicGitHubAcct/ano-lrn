# Source-Filter

`src/python/source_filter.py` implements linear predictive coding (LPC) analysis and synthesis, and a cascade of resonator filters for formant modelling.

## Summary

| Function | Key parameters | Returns | Description |
| --- | --- | --- | --- |
| `lpc_coeffs` | `signal`, `order` | coefficient array | Autocorrelation-method LPC analysis |
| `lpc_synthesize` | `excitation`, `coeffs` | signal | All-pole synthesis filter |
| `formant_filter` | `signal`, `formant_freqs`, `bandwidths`, `fs` | signal | Cascade of 2nd-order resonators |

---

### `lpc_coeffs`

Linear predictive coding models the signal as the output of an all-pole filter driven by white noise. The predictor coefficients minimise the mean-squared prediction error:

```text
x̂[n] = a₁·x[n−1] + a₂·x[n−2] + … + aₚ·x[n−p]
```

The autocorrelation method sets up a Toeplitz system of equations (the Yule-Walker equations) and solves for the coefficient vector:

```text
R · a = r
```

where R[i,j] = r[|i−j|] is the autocorrelation matrix and r = [r[1], …, r[p]] is the autocorrelation vector at positive lags. `order` p should be approximately fs / 1000 (one coefficient per kHz) for speech modelling; 12 is a common choice at 12 kHz.

---

### `lpc_synthesize`

Applies the all-pole LPC synthesis filter to an excitation signal:

```text
H(z) = 1 / A(z) = 1 / (1 − a₁·z⁻¹ − a₂·z⁻² − … − aₚ·z⁻ᵖ)
```

The excitation for voiced speech is typically a pulse train; for unvoiced speech it is white noise. Together `lpc_coeffs` + `lpc_synthesize` implement the classic speech synthesis chain: excitation → spectral envelope → output.

**LPC residual:** subtracting the all-pole filtered signal from the original gives the LPC residual, which is an approximately white signal representing the glottal source.

---

### `formant_filter`

Each formant is modelled as a second-order resonator with a peak at frequency f₀ and –3 dB bandwidth bw:

```text
H_k(z) = (1 − R²) / (1 − 2R·cos(θ)·z⁻¹ + R²·z⁻²)
```

where θ = 2π·f₀/fs and R = exp(−π·bw/fs). The cascade of K resonators gives:

```text
H(z) = ∏ₖ H_k(z)
```

The approximate formant frequencies for the vowel /a/ in English are F1 ≈ 800 Hz, F2 ≈ 1200 Hz, F3 ≈ 2500 Hz with bandwidths of 80–120 Hz. Applying this filter to an impulse train produces a synthetic vowel.
