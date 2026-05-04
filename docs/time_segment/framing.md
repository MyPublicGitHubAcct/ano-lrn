# Framing

## `frame`

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

## `overlap_add`

Reconstructs a signal from a 2D array of frames using the overlap-add (OLA) method:

```text
output[i·hop : i·hop + frame_size] += frames[i, :] · w[:]
```

After accumulation, divides by the sum of squared window values to compensate for the analysis window. A window satisfying the constant overlap-add (COLA) condition produces perfect reconstruction:

**COLA condition:** Hann window with 50% or 75% overlap; Hamming window with 50% overlap.

The output length is `(num_frames − 1) · hop_size + frame_size`. OLA is the synthesis step in the phase vocoder, STFT processing, and audio effects that operate frame by frame.
