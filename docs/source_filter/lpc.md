# Linear Predictive Coding

## `lpc_coeffs`

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

## `lpc_synthesize`

Applies the all-pole LPC synthesis filter to an excitation signal:

```text
H(z) = 1 / A(z) = 1 / (1 − a₁·z⁻¹ − a₂·z⁻² − … − aₚ·z⁻ᵖ)
```

The excitation for voiced speech is typically a pulse train; for unvoiced speech it is white noise. Together `lpc_coeffs` + `lpc_synthesize` implement the classic speech synthesis chain: excitation → spectral envelope → output.

**LPC residual:** subtracting the all-pole filtered signal from the original gives the LPC residual, which is an approximately white signal representing the glottal source.
