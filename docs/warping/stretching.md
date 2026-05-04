# Time Stretching and Pitch Shifting

## `time_stretch`

Phase-vocoder time stretching changes the playback rate without affecting pitch:

```text
rate > 1: output is longer  (slower playback)
rate < 1: output is shorter (faster playback)
```

**Algorithm:**

1. Compute the STFT of the input with analysis hop size H.
2. Resample the time axis: for output frame i, read from source frame i/rate.
3. Accumulate phase increments scaled by the true instantaneous frequency:

```text
Δφ[k]     = angle(S[k, m+1]) − angle(S[k, m]) − ω_k   (wrapped to [−π, π])
ω_k       = 2π · k · H / N                              (expected phase advance)
φ_out[k]  = φ_out[k] + ω_k + Δφ[k]
```

4. Resynthesize via ISTFT using the accumulated phases.

Phase locking produces artifact-free stretching for tonal signals; percussive transients can smear under heavy stretch ratios.

---

## `pitch_shift`

Pitch shifting is implemented as time stretching followed by resampling:

```text
ratio = 2^(semitones / 12)
stretched = time_stretch(signal, ratio)          # longer for pitch up
output    = resample(stretched, N)               # resample back to N samples
```

For pitch UP (ratio > 1): the signal is stretched longer by `ratio`, then resampled down to the original length. The resampling compresses the time axis, raising all frequencies by `ratio`.

For pitch DOWN (ratio < 1): the signal is compressed shorter, then resampled up.

**One octave up (12 semitones):** ratio = 2.0. The stretched signal is twice as long with the same pitch; resampling from 2N to N doubles all frequencies.

| semitones | ratio | effect |
| --- | --- | --- |
| +12 | 2.00 | one octave up |
| +7  | 1.50 | perfect fifth up |
| +3  | 1.19 | minor third up |
| 0   | 1.00 | no change |
| −12 | 0.50 | one octave down |
