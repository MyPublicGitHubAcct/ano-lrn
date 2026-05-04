# NLMS Adaptive Filter

## `nlms`

NLMS normalises the step size by the instantaneous input power:

```text
w[n+1] = w[n] + (μ / (‖x[n]‖² + ε)) · e[n] · x[n]
```

This makes convergence speed independent of the input amplitude, allowing a fixed `mu` close to 1.0 that is nearly optimal for a wide range of input levels. The small constant `eps` prevents division by zero for silent input.

**Returns:** `(output, error, weights)` — all arrays of length equal to `desired`.

---

### Comparison with LMS

| Property | LMS | NLMS |
| --- | --- | --- |
| Step size | Fixed μ | Normalised by input power |
| Convergence rate | Depends on input level | Input-level independent |
| Stability condition | 0 < μ < 1/(p·E[x²]) | 0 < μ < 2 |
| Computation | O(p) | O(p) + one extra dot-product |
| Typical μ | 0.001–0.01 | 0.3–1.0 |

NLMS converges faster for most signals and is more robust to changes in input level. The trade-off is a slightly higher computational cost (one extra dot-product per sample for the norm).
