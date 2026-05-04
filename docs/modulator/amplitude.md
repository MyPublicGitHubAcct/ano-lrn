# Amplitude Modulation

## `tremolo`

Tremolo is periodic amplitude modulation applied by an LFO (low-frequency oscillator):

```text
y[n] = x[n] · (1 − depth/2 · (1 − cos(2π · rate · n/fs)))
```

The LFO envelope oscillates between `1 − depth` (minimum gain) and `1` (maximum gain). At `depth = 0` the signal is unchanged; at `depth = 1` the amplitude dips to zero on each LFO cycle.

**Spectrum:** tremolo produces sidebands at f_signal ± f_LFO, but because the envelope is always positive the carrier frequency is preserved (unlike ring modulation).

---

## `ring_modulate`

Ring modulation multiplies the signal by a cosine:

```text
y[n] = x[n] · cos(2π · carrier · n/fs)
```

By the product-to-sum identity, cos(A)cos(B) = ½(cos(A−B) + cos(A+B)), so a pure tone at f_signal produces sidebands at f_signal − carrier and f_signal + carrier. The carrier itself is suppressed (unlike amplitude modulation with a DC offset). At `carrier_freq = 0` the cosine is 1 at all samples, leaving the signal unchanged.

**Use case:** metallic timbres, robot voice, AM radio demodulation.

---

## Parameter Ranges

**`tremolo`**

| Parameter | Range | Notes |
| --- | --- | --- |
| `rate` | 0.1–20 Hz | LFO rate; musical range typically 1–8 Hz; `depth = 0` makes `rate` irrelevant |
| `depth` | [0, 1] | `0` = identity (no modulation); `1` = amplitude dips to zero on each LFO trough |
| `fs` | 8000–192000 Hz | Any standard audio sample rate |

**`ring_modulate`**

| Parameter | Range | Notes |
| --- | --- | --- |
| `carrier_freq` | 0 – fs/2 | `0` = identity (cosine is DC = 1); practical metallic range 50–5000 Hz |
| `fs` | 8000–192000 Hz | Any standard audio sample rate |
