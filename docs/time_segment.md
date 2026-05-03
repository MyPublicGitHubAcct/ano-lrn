# Time Segment

`src/python/time_segment.py` provides windowing, framing, and overlap-add reconstruction — the fundamental building blocks for all block-processing DSP algorithms.

## Summary

| Function | Key parameters | Returns | Description |
| --- | --- | --- | --- |
| `apply_window` | `window_type` | `np.ndarray` | Multiply signal by a window function |
| `frame` | `frame_size`, `hop_size` | 2D `np.ndarray` | Segment signal into overlapping frames |
| `overlap_add` | `frames`, `hop_size`, `window_type` | `np.ndarray` | Reconstruct signal from frames via OLA |

---

### `apply_window`

Multiplies a signal element-wise by a smooth window function:

```text
y[n] = x[n] · w[n]
```

Supported window types: `'hann'`, `'hamming'`, `'blackman'`, `'rectangular'`.

Windows are used before FFT analysis to reduce spectral leakage. A rectangular window has maximum leakage (sharp edges cause sinc-shaped sidelobes); the Hann window reduces peak sidelobe levels to ~−32 dB; Blackman reduces them further to ~−58 dB at the cost of a wider main lobe.

**Hann window:** w[n] = 0.5 · (1 − cos(2π·n/(N−1)))

The Hann window is zero at both endpoints, making it ideal for overlap-add synthesis with 50% or 75% overlap.

---

### `frame`

Segments a 1D signal into a 2D array of overlapping frames:

```text
frames[i, :] = signal[i·hop : i·hop + frame_size]
```

Returns shape `(num_frames, frame_size)` where `num_frames = 1 + (N − frame_size) // hop_size`. Samples at the tail that don't fill a complete frame are zero-padded.

The `hop_size` controls overlap:
- `hop_size = frame_size` → no overlap (0%)
- `hop_size = frame_size // 2` → 50% overlap
- `hop_size = frame_size // 4` → 75% overlap (required for COLA with Hann window)

---

### `overlap_add`

Reconstructs a signal from a 2D array of frames using the overlap-add (OLA) method:

```text
output[i·hop : i·hop + frame_size] += frames[i, :] · w[:]
```

After accumulation, divides by the sum of squared window values to compensate for the analysis window. A window satisfying the constant overlap-add (COLA) condition produces perfect reconstruction:

**COLA condition:** Hann window with 50% or 75% overlap; Hamming window with 50% overlap.

The output length is `(num_frames − 1) · hop_size + frame_size`. OLA is the synthesis step in the phase vocoder, STFT processing, and audio effects that operate frame by frame.
