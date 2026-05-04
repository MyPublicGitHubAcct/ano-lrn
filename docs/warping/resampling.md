# Resampling

## `resample`

Polyphase resampling converts between sample rates without aliasing by:

1. Finding the integer ratio `up / down = target_fs / orig_fs` (reduced by GCD).
2. Upsampling by `up` (inserting zeros).
3. Lowpass filtering at min(orig_fs, target_fs) / 2.
4. Downsampling by `down`.

The output length scales proportionally: `len(output) = len(input) · target_fs / orig_fs`.

**Quality:** `scipy.signal.resample_poly` uses a polyphase FIR filter with a Kaiser window, which gives excellent aliasing rejection. The transition band width depends on the oversampling ratio.
