# Pitch Modulation

## `vibrato`

Vibrato modulates pitch by varying the read position in a fractional delay line:

```text
delay[n] = center + depth_samples · sin(2π · rate · n/fs)
y[n]     = x[n − delay[n]]   (linearly interpolated)
```

The delay oscillates around a fixed `center` value, pulling the instantaneous pitch up and down. Sub-sample accuracy is achieved via linear interpolation between adjacent samples. Unlike tremolo (which affects amplitude), vibrato affects pitch.

**Depth–pitch relationship:** depth_samples = (Δf / f_signal) · fs / (2π · rate), where Δf is the peak frequency deviation in Hz.
