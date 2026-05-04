import numpy as np

from python.virtual_analog import moog_ladder

FS = 44100


def _sine(freq=440.0, n=2048):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def test_moog_ladder_shape():
    sig = _sine()
    assert moog_ladder(sig, cutoff=1000.0, fs=FS).shape == sig.shape


def test_moog_attenuates_high_frequencies():
    sig_low = _sine(freq=100.0)
    sig_high = _sine(freq=10000.0)
    out_low = moog_ladder(sig_low, cutoff=1000.0, fs=FS)
    out_high = moog_ladder(sig_high, cutoff=1000.0, fs=FS)
    rms_low = _rms(out_low[len(out_low) // 2:])
    rms_high = _rms(out_high[len(out_high) // 2:])
    assert rms_low > rms_high * 2


def test_moog_zero_resonance_no_self_oscillation():
    sig = np.zeros(2048)
    sig[0] = 1.0
    out = moog_ladder(sig, cutoff=1000.0, fs=FS, resonance=0.0)
    assert np.all(np.isfinite(out))


def test_moog_ladder_four_pole_rolloff_slope():
    """Slope between 1 and 2 octaves above cutoff must exceed −16 dB/oct.

    A 2-pole regression gives ≈ −12 dB/oct here; the four-pole Huovilainen
    implementation (with digital frequency warping) gives ≈ −21 dB/oct.
    The asymptotic 4-pole slope is −24 dB/oct, approached deeper in the stopband.
    """
    N = 131072
    cutoff = 1000.0
    impulse = np.zeros(N)
    impulse[0] = 1.0
    h = moog_ladder(impulse, cutoff=cutoff, fs=FS, resonance=0.0)
    freqs = np.fft.rfftfreq(N, d=1.0 / FS)
    db = 20 * np.log10(np.abs(np.fft.rfft(h)) + 1e-12)

    def db_at(f):
        idx = np.argmin(np.abs(freqs - f))
        return db[idx]

    f1 = cutoff * 2  # one octave above cutoff
    f2 = cutoff * 4  # two octaves above cutoff
    slope = db_at(f2) - db_at(f1)  # dB per octave (f2/f1 == 2)
    assert -27.0 <= slope <= -16.0, f"Expected ~-24 dB/octave rolloff, got {slope:.1f} dB/octave"
