# Delay Line

## `delay_line`

A causal integer-sample delay:

```text
y[n] = x[n − D]
```

Transfer function: H(z) = z^(−D). All samples before index D are zero (initial rest). This is the simplest possible DSP operation and serves as the building block for the two comb filters (`feedback_delay`, `comb_filter`). It is also used by `spatial.haas`.

**Parameters:** `delay_samples` — integer delay D in samples.

**Use cases:** time-aligning signals, creating echo taps, building more complex delay networks.
