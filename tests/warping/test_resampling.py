import numpy as np

from python.warping import resample

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_resample_output_length():
    sig = _sine(n=4410)
    out = resample(sig, orig_fs=44100, target_fs=22050)
    assert len(out) == 2205


def test_resample_upsample_length():
    sig = _sine(n=2205)
    out = resample(sig, orig_fs=22050, target_fs=44100)
    assert len(out) == 4410


def test_resample_preserves_frequency_content():
    freq = 1000.0
    sig_44 = np.sin(2 * np.pi * freq * np.arange(4410) / 44100)
    sig_22 = resample(sig_44, 44100, 22050)
    spectrum = np.abs(np.fft.rfft(sig_22, n=len(sig_22)))
    freqs = np.fft.rfftfreq(len(sig_22), d=1.0 / 22050)
    dom = float(freqs[np.argmax(spectrum)])
    assert abs(dom - freq) < 50.0


def test_resample_identity_same_rate():
    sig = _sine(n=4096)
    out = resample(sig, 44100, 44100)
    np.testing.assert_allclose(out, sig, atol=1e-10)
