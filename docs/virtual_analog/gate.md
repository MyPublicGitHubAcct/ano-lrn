# Virtual Analog — Lowpass Gate

## `lowpass_gate`

The Buchla 292c/292e lowpass gate combines a voltage-controlled 4-pole lowpass filter with a VCA in a single circuit driven by a vactrol (LED + photoresistor optical coupler). The defining characteristic is the vactrol's asymmetric time response: the LED lights quickly on a gate-on edge (fast attack) while the photoresistor resistance decays slowly after the LED dims (slow, mushy release). This gives percussive and plucked sounds a natural bloom without a hard amplitude cut.

### Vactrol model

The photoresistor brightness is tracked by an asymmetric one-pole smoother:

```text
alpha_a = 1 − exp(−1 / (tau_a · fs))      (attack coefficient)
alpha_r = 1 − exp(−1 / (tau_r · fs))      (release coefficient)

if control[n] > brightness[n−1]:
    brightness[n] = brightness[n−1] + alpha_a · (control[n] − brightness[n−1])
else:
    brightness[n] = brightness[n−1] + alpha_r · (control[n] − brightness[n−1])
```

Typical 292c vactrol values: `tau_a ≈ 10 ms`, `tau_r ≈ 150–300 ms`. The slow release is the primary contributor to the gate's "wooden" or "bongo-like" decay.

### Cutoff mapping

Photoresistor resistance is approximately logarithmic in brightness, so the cutoff frequency is mapped exponentially:

```text
cutoff(n) = exp(log(min_cutoff) + brightness[n] · (log(max_cutoff) − log(min_cutoff)))
```

At `brightness = 0` the filter sits at `min_cutoff` (default 20 Hz, effectively closed). At `brightness = 1` it opens to `max_cutoff` (default 8 kHz).

### Filter

Four cascaded one-pole stages (no resonance — the 292 has no Q control):

```text
g    = 1 − exp(−2π · cutoff(n) / fs)
s₀  += g · (x[n] − s₀)
s₁  += g · (s₀   − s₁)
s₂  += g · (s₁   − s₂)
s₃  += g · (s₂   − s₃)
```

Unlike the Moog ladder, there is no tanh nonlinearity and no resonance feedback path. The filter is linear; character comes entirely from the vactrol envelope.

### Modes

| mode | behaviour |
|------|-----------|
| `"both"` (default) | filter + VCA: `y = s₃ · brightness` |
| `"lowpass"` | filter only: `y = s₃`, amplitude not scaled |
| `"amplitude"` | VCA only: `y = x · brightness`, no filtering |

The `"both"` mode is the most characteristic and matches how the 292c behaves when a gate or trigger is patched into the control input.

### Parameters

| parameter | default | meaning |
|-----------|---------|---------|
| `attack_time` | 0.010 s | vactrol attack time constant |
| `release_time` | 0.200 s | vactrol release time constant |
| `max_cutoff` | 8000 Hz | filter cutoff at full open |
| `min_cutoff` | 20 Hz | filter cutoff at closed |

### Use cases

- Percussive plucked/bongo sounds: short gate pulse + `"both"` mode
- Envelope-following filter: feed a rectified audio signal as `control`
- Natural note decay: `release_time` 150–500 ms for organic bloom
