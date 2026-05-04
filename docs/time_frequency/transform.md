# STFT Transform

## `stft`

The STFT divides the signal into short overlapping frames, applies a window function, and takes the FFT of each frame:

```text
S[k, m] = Σ_n  x[n] · w[n − m·H] · e^(−j2πkn/N)
```

where N = `frame_size`, H = `hop_size`, and the sum runs over the frame indices.

Returns a complex array of shape `(N//2 + 1, num_frames)` (rfft output: DC to Nyquist). Each column is one time frame; each row is one frequency bin from 0 Hz to fs/2.

**Time-frequency resolution trade-off:** larger `frame_size` gives finer frequency resolution (bins spaced by fs/N Hz) but coarser time resolution (each frame covers N/fs seconds). Smaller `hop_size` gives denser temporal sampling but does not change the frequency resolution.

---

## `istft`

Inverts the STFT via overlap-add synthesis:

```text
x̂[n] = Σ_m  IFFT(S[:, m])[n − m·H] · w[n − m·H]
```

followed by normalisation by the sum-of-squares of the window. For perfect reconstruction the window must satisfy the COLA (constant overlap-add) condition — the Hann window with 75% overlap (hop = frame/4) is the standard choice.

**Application:** modify `S` in the frequency domain (e.g., apply a spectral mask) then call `istft` to synthesise the processed signal.
