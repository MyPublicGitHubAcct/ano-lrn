# Spectrogram

## `spectrogram`

Computes the log-magnitude STFT:

```text
S_dB[k, m] = 20 · log10(|S[k, m]| + ε)
```

The `ε = 1e-12` floor prevents log(0). Values well into the noise floor are typically around −240 dB (numerical precision limit). The output shape is the same as `stft`: `(N//2 + 1, num_frames)`.

**Note on absolute levels:** the raw FFT magnitude scales with `frame_size` and the window normalisation; a full-scale sine of amplitude 1 with frame_size=512 and a Hann window produces peak magnitudes around 42 dB, not 0 dB. Use this function for relative comparisons and visualisation, not for calibrated level measurements.
