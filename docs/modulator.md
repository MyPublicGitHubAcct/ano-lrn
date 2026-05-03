# Modulator

`src/python/modulator.py` implements three time-varying amplitude and frequency modulation effects.

## Summary

| Function | Key parameters | Description |
| --- | --- | --- |
| `tremolo` | `rate`, `depth`, `fs` | Amplitude modulation via sinusoidal LFO |
| `ring_modulate` | `carrier_freq`, `fs` | Multiply signal by a cosine; produces sidebands |
| `vibrato` | `rate`, `depth_samples`, `fs` | Pitch modulation via fractional variable delay |

---

### `tremolo`

Tremolo is periodic amplitude modulation applied by an LFO (low-frequency oscillator):

```text
y[n] = x[n] · (1 − depth/2 · (1 − cos(2π · rate · n/fs)))
```

The LFO envelope oscillates between `1 − depth` (minimum gain) and `1` (maximum gain). At `depth = 0` the signal is unchanged; at `depth = 1` the amplitude dips to zero on each LFO cycle.

**Spectrum:** tremolo produces sidebands at f_signal ± f_LFO, but because the envelope is always positive the carrier frequency is preserved (unlike ring modulation).

---

### `ring_modulate`

Ring modulation multiplies the signal by a cosine:

```text
y[n] = x[n] · cos(2π · carrier · n/fs)
```

By the product-to-sum identity, cos(A)cos(B) = ½(cos(A−B) + cos(A+B)), so a pure tone at f_signal produces sidebands at f_signal − carrier and f_signal + carrier. The carrier itself is suppressed (unlike amplitude modulation with a DC offset). At `carrier_freq = 0` the cosine is 1 at all samples, leaving the signal unchanged.

**Use case:** metallic timbres, robot voice, AM radio demodulation.

---

### `vibrato`

Vibrato modulates pitch by varying the read position in a fractional delay line:

```text
delay[n] = center + depth_samples · sin(2π · rate · n/fs)
y[n]     = x[n − delay[n]]   (linearly interpolated)
```

The delay oscillates around a fixed `center` value, pulling the instantaneous pitch up and down. Sub-sample accuracy is achieved via linear interpolation between adjacent samples. Unlike tremolo (which affects amplitude), vibrato affects pitch.

**Depth–pitch relationship:** depth_samples = (Δf / f_signal) · fs / (2π · rate), where Δf is the peak frequency deviation in Hz.
