# Transient Generators

Single-event signals used to probe how a system responds to sudden changes.

| Generator | Key parameters | Description |
| --- | --- | --- |
| `generate_impulse` | `delay` | Single non-zero sample; flat spectrum |
| `generate_step` | `onset` | Switches from 0 to `amplitude` and holds; spectrum rolls off as 1/f |

Both return `(t, signal)` — a time axis and signal array sampled at `fs` Hz over `duration` seconds.

---

## `generate_impulse`

A single non-zero sample (Dirac delta approximation) at time `delay`. All other samples are zero.

```text
x[n] = A · δ[n − delay·fs]
```

The Fourier transform of a Dirac delta is a constant — **flat spectrum at all frequencies**. This means that running an impulse through any LTI system and taking the FFT of the output gives the system's full frequency response in one shot. This is how `examples/plot_filters_eq.py` derives filter frequency responses.

**Use cases:** measuring impulse responses, deriving frequency responses, unit testing filter shapes.

---

## `generate_step`

Switches from 0 to `amplitude` at time `onset` and holds. The integral of an impulse; its spectrum rolls off as `1/f`.

```text
x[n] = A · u[n − onset·fs]
```

The step response of a system reveals how it handles sudden transitions: overshoot and ringing indicate underdamped poles; slow rise indicates heavy low-pass filtering. DC gain can be read directly from the settled output value.

**Use cases:** testing transient behavior, measuring DC gain, verifying filter stability after a level jump.
