# Adaptive

`src/python/adaptive.py` implements two adaptive FIR filters that adjust their coefficients online to minimise a cost function.

## Summary

| Function | Key parameters | Returns | Description |
| --- | --- | --- | --- |
| `lms` | `desired`, `reference`, `filter_order`, `mu` | `(output, error, weights)` | Least Mean Squares adaptive filter |
| `nlms` | `desired`, `reference`, `filter_order`, `mu`, `eps` | `(output, error, weights)` | Normalised LMS adaptive filter |

---

### `lms`

The LMS algorithm minimises the mean-squared error between the filter output and a desired signal by updating the weight vector in the direction of the negative gradient:

```text
y[n]   = wᵀ[n] · x[n]          (filter output)
e[n]   = d[n] − y[n]            (error signal)
w[n+1] = w[n] + 2μ · e[n] · x[n]  (weight update)
```

where x[n] = [reference[n], reference[n−1], …, reference[n−p+1]]ᵀ is the current input buffer and d[n] is the desired signal.

**Convergence:** the step size μ must satisfy 0 < μ < 1 / (filter_order · E[x²]) for stability. Smaller μ converges more slowly but produces a lower steady-state misadjustment error. The LMS is computationally inexpensive (O(p) per sample) and robust.

**Use cases:** system identification, noise cancellation, echo cancellation (with a delayed microphone reference).

---

### `nlms`

NLMS normalises the step size by the instantaneous input power:

```text
w[n+1] = w[n] + (μ / (‖x[n]‖² + ε)) · e[n] · x[n]
```

This makes convergence speed independent of the input amplitude, allowing a fixed `mu` close to 1.0 that is nearly optimal for a wide range of input levels. The small constant `eps` prevents division by zero for silent input.

**Comparison to LMS:** NLMS converges faster for most signals and is more robust to changes in input level. The trade-off is a slightly higher computational cost (one extra dot-product per sample for the norm).
