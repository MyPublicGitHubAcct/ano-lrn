import numpy as np

from python.warping import resample

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_resample_output_length():
    """Downsampling 4410 samples from 44100 to 22050 Hz must produce exactly 2205 samples."""
    sig = _sine(n=4410)
    out = resample(sig, orig_fs=44100, target_fs=22050)
    assert len(out) == 2205


def test_resample_upsample_length():
    """Upsampling 2205 samples from 22050 to 44100 Hz must produce exactly 4410 samples."""
    sig = _sine(n=2205)
    out = resample(sig, orig_fs=22050, target_fs=44100)
    assert len(out) == 4410


def test_resample_preserves_frequency_content():
    """After downsampling, the dominant frequency of a 1 kHz sine must still be within 50 Hz of 1 kHz."""
    freq = 1000.0
    sig_44 = np.sin(2 * np.pi * freq * np.arange(4410) / 44100)
    sig_22 = resample(sig_44, 44100, 22050)
    spectrum = np.abs(np.fft.rfft(sig_22, n=len(sig_22)))
    freqs = np.fft.rfftfreq(len(sig_22), d=1.0 / 22050)
    dom = float(freqs[np.argmax(spectrum)])
    assert abs(dom - freq) < 50.0


def test_resample_identity_same_rate():
    """Resampling to the same rate must return the input unchanged."""
    sig = _sine(n=4096)
    out = resample(sig, 44100, 44100)
    np.testing.assert_allclose(out, sig, atol=1e-10)


def test_resample_single_sample_no_crash():
    """Resampling a 1-sample signal must not raise an error or produce NaN."""
    sig = np.array([0.5])
    out = resample(sig, orig_fs=44100, target_fs=22050)
    assert np.all(np.isfinite(out))


def test_resample_non_integer_ratio_length_and_frequency():
    """44100 → 48000 Hz: output length must equal round(N × 48000/44100) and dominant freq must be preserved."""
    orig_fs, target_fs = 44100, 48000
    freq = 1000.0
    n = 44100
    sig = np.sin(2 * np.pi * freq * np.arange(n) / orig_fs)
    out = resample(sig, orig_fs=orig_fs, target_fs=target_fs)
    expected_len = round(n * target_fs / orig_fs)
    assert abs(len(out) - expected_len) <= 1
    spectrum = np.abs(np.fft.rfft(out))
    freqs_ax = np.fft.rfftfreq(len(out), d=1.0 / target_fs)
    dominant = float(freqs_ax[np.argmax(spectrum)])
    assert abs(dominant - freq) < 50.0

