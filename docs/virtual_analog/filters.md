# Virtual Analog Filters

## `moog_ladder`

The Moog transistor ladder filter is a 4-pole (24 dB/octave) lowpass with nonlinear resonance. The Huovilainen discretisation cascades four one-pole stages with tanh nonlinearities and a feedback path:

```text
g = 1 − exp(−2π · fc)           (one-pole coefficient)
k = 4 · resonance                (feedback gain)

x_fb   = x[n] − k · s₃         (input minus resonance feedback)
s₀    += g · (tanh(x_fb) − tanh(s₀))
s₁    += g · (tanh(s₀)  − tanh(s₁))
s₂    += g · (tanh(s₁)  − tanh(s₂))
s₃    += g · (tanh(s₂)  − tanh(s₃))
y[n]   = s₃
```

At `resonance = 0` the filter is a clean 4th-order lowpass. As resonance approaches 1 the filter self-oscillates at `cutoff`, producing a pure sinusoid. The tanh in each stage limits the resonance amplitude.

**Cutoff tracking:** the filter does not track pitch accurately at high frequencies due to the bilinear approximation; pre-warping (`fc = sin(π·f/fs) / π`) improves accuracy at the cost of a nonlinear cutoff mapping.

---

## Parameter Ranges

| Parameter | Range | Notes |
| --- | --- | --- |
| `cutoff` | 20–20000 Hz | Automatically clamped to [1e-6 · fs, 0.499 · fs]; values outside the audio band are clamped rather than rejected |
| `resonance` | [0, 1) | k = 4 · resonance; self-oscillation occurs at resonance = 1 (k = 4); the tanh nonlinearity limits amplitude growth near this threshold |
| `fs` | 8000–192000 Hz | Any standard audio sample rate |
