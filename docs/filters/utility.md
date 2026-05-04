# Utility Filter

## `dc_block`

Removes DC offset and sub-sonic content below `cutoff` Hz while leaving all higher-frequency content unchanged. Unlike the biquad `highpass` filter, `dc_block` is first-order (one pole, one zero) and is specifically designed for DC removal rather than audio-band shaping.

---

### Coefficient derivation

`dc_block` applies the bilinear transform to the first-order analog high-pass H_a(s) = s / (s + ω_c):

```text
k  = tan(π · cutoff / fs)    # bilinear pre-warp
b  = [1/(1+k),  −1/(1+k)]
a  = [1.0,      −(1−k)/(1+k)]
```

The bilinear transform maps the analog −3 dB frequency ω_c exactly to the digital frequency `cutoff` Hz.

---

### Frequency response

- **H(1) = 0** (DC, z = 1): structural zero at z = +1; DC is always completely rejected regardless of `cutoff`
- **H(−1) = 1** (Nyquist, z = −1): unity gain at Nyquist for any `cutoff`
- **−3 dB at `cutoff`**: bilinear pre-warp guarantees the half-power point is exactly at the requested frequency
- **−20 dB/decade** rolloff below `cutoff` (first-order slope)

---

### Comparison with `highpass`

| Property | `highpass` | `dc_block` |
| --- | --- | --- |
| Order | 2nd (biquad) | 1st |
| Roll-off | −40 dB/decade | −20 dB/decade |
| Q control | Yes | No |
| DC rejection | Complete | Complete (structural zero) |
| Typical cutoff | 20 Hz – audio band | < 20 Hz (sub-sonic) |

Use `dc_block` when you only need DC and hum removal and want the flattest possible response in the audio band. Use `highpass` when you need steeper roll-off or Q control for audio filtering.
