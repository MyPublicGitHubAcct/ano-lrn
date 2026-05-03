# Warping

`src/python/warping.py` implements sample-rate conversion, time stretching, and pitch shifting.

## Summary

| Function | Key parameters | Returns | Description |
| --- | --- | --- | --- |
| `resample` | `signal`, `orig_fs`, `target_fs` | signal | Polyphase sample-rate conversion |
| `time_stretch` | `signal`, `rate`, `frame_size`, `hop_size` | signal | Phase-vocoder time stretch, pitch preserved |
| `pitch_shift` | `signal`, `semitones`, `fs`, `frame_size`, `hop_size` | signal | Phase-vocoder pitch shift, duration preserved |

---

### `resample`

Polyphase resampling converts between sample rates without aliasing by:

1. Finding the integer ratio `up / down = target_fs / orig_fs` (reduced by GCD).
2. Upsampling by `up` (inserting zeros).
3. Lowpass filtering at min(orig_fs, target_fs) / 2.
4. Downsampling by `down`.

The output length scales proportionally: `len(output) = len(input) · target_fs / orig_fs`.

**Quality:** `scipy.signal.resample_poly` uses a polyphase FIR filter with a Kaiser window, which gives excellent aliasing rejection. The transition band width depends on the oversampling ratio.

---

### `time_stretch`

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

### `pitch_shift`

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
