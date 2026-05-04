# Clipping

## `hard_clip`

Symmetrically clips the signal to the interval [−threshold, +threshold]:

```text
y[n] = clip(x[n], −T, +T)
```

Clipping introduces discontinuities in the waveform, producing a rich spectrum of all harmonics (both even and odd). With a high threshold the signal passes unchanged; as the threshold decreases, more cycles are squared-off. Commonly used in guitar amp emulations and to prevent digital overload.

**Spectrum:** the clipped waveform can be decomposed into Fourier series. For a symmetrically clipped sine wave the harmonics follow a pattern of odd and even partials depending on the clip depth.

---

## `soft_clip`

Hyperbolic tangent saturation:

```text
y[n] = tanh(drive · x[n])
```

The output is bounded in (−1, +1) for all inputs. Near zero the transfer function is approximately linear (tanh(x) ≈ x). As the input grows the gain compresses, producing a smooth transition from clean to saturated. Higher `drive` values move the knee lower, increasing harmonic distortion at lower input levels.

**Odd symmetry:** tanh(−x) = −tanh(x), so only odd harmonics are generated for a pure sine input (3rd, 5th, 7th, …). This gives a "tube-like" character compared to the even harmonics of asymmetric distortions.
