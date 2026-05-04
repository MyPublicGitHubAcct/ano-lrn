# LMS Adaptive Filter

## `lms`

The LMS algorithm minimises the mean-squared error between the filter output and a desired signal by updating the weight vector in the direction of the negative gradient:

```text
y[n]   = wᵀ[n] · x[n]              (filter output)
e[n]   = d[n] − y[n]               (error signal)
w[n+1] = w[n] + 2μ · e[n] · x[n]  (weight update)
```

where x[n] = [reference[n], reference[n−1], …, reference[n−p+1]]ᵀ is the current input buffer and d[n] is the desired signal.

**Returns:** `(output, error, weights)` — all arrays of length equal to `desired`.

---

### Convergence

The step size μ must satisfy:

```text
0 < μ < 1 / (filter_order · E[x²])
```

for stability. Smaller μ converges more slowly but produces a lower steady-state misadjustment error. The LMS is computationally inexpensive (O(p) per sample) and robust.

**Use cases:** system identification, noise cancellation, echo cancellation (with a delayed microphone reference).
