# TODO

## Up next

### Complete the Audio EQ Cookbook filter set

Add notch (band-reject), all-pass, low-shelf, and high-shelf biquad filters to `src/ano_lrn/filters.py`, following the same pattern as the existing lowpass/highpass/bandpass:

- Private `_coeffs` helper per filter type
- Public function with `(signal, cutoff, fs, Q)` signature
- Tests in `tests/test_filters.py`
- Plots in `examples/plot_filters.py`
- DSP concepts documented in `docs/filters.md`
- README table updated

Notes:
- Shelf filters take a `gain_db` parameter instead of Q (gain at DC for low-shelf, gain at Nyquist for high-shelf)
- All-pass has unity magnitude response at all frequencies; only the phase changes — test via group delay rather than RMS attenuation
