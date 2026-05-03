# Time-Frequency

`src/python/time_frequency.py` provides the short-time Fourier transform (STFT), its inverse, and magnitude spectrogram — the core tools for frequency-domain audio processing.

## Summary

| Function | Key parameters | Returns | Description |
| --- | --- | --- | --- |
| `stft` | `frame_size`, `hop_size`, `window` | complex 2D array | Short-time Fourier transform |
| `istft` | `S`, `hop_size`, `window` | 1D array | Inverse STFT via overlap-add |
| `spectrogram` | `frame_size`, `hop_size`, `window` | real 2D array | Magnitude spectrogram in dB |

---

### `stft`

The STFT divides the signal into short overlapping frames, applies a window function, and takes the FFT of each frame:

```text
S[k, m] = Σ_n  x[n] · w[n − m·H] · e^(−j2πkn/N)
```

where N = `frame_size`, H = `hop_size`, and the sum runs over the frame indices.

Returns a complex array of shape `(N//2 + 1, num_frames)` (rfft output: DC to Nyquist). Each column is one time frame; each row is one frequency bin from 0 Hz to fs/2.

**Time-frequency resolution trade-off:** larger `frame_size` gives finer frequency resolution (bins spaced by fs/N Hz) but coarser time resolution (each frame covers N/fs seconds). Smaller `hop_size` gives denser temporal sampling but does not change the frequency resolution.

---

### `istft`

Inverts the STFT via overlap-add synthesis:

```text
x̂[n] = Σ_m  IFFT(S[:, m])[n − m·H] · w[n − m·H]
```

followed by normalisation by the sum-of-squares of the window. For perfect reconstruction the window must satisfy the COLA (constant overlap-add) condition — the Hann window with 75% overlap (hop = frame/4) is the standard choice.

**Application:** modify `S` in the frequency domain (e.g., apply a spectral mask) then call `istft` to synthesise the processed signal.

---

### `spectrogram`

Computes the log-magnitude STFT:

```text
S_dB[k, m] = 20 · log10(|S[k, m]| + ε)
```

The `ε = 1e-12` floor prevents log(0). Values well into the noise floor are typically around −240 dB (numerical precision limit). The output shape is the same as `stft`: `(N//2 + 1, num_frames)`.

**Note on absolute levels:** the raw FFT magnitude scales with `frame_size` and the window normalisation; a full-scale sine of amplitude 1 with frame_size=512 and a Hann window produces peak magnitudes around 42 dB, not 0 dB. Use this function for relative comparisons and visualisation, not for calibrated level measurements.
