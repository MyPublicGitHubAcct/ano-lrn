import numpy as np

from python.time_frequency import istft, stft


# ── Spectral ──────────────────────────────────────────────────────────────────

def spectral_centroid(
    signal: np.ndarray,
    fs: int,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> np.ndarray:
    """Frequency-weighted mean of the magnitude spectrum per frame.

    Returns an array of centroid values in Hz, one per STFT frame.
    High centroid = energy concentrated at high frequencies (e.g. cymbals).
    """
    S = stft(signal, frame_size, hop_size)
    mag = np.abs(S)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / fs)
    total = mag.sum(axis=0) + 1e-12
    return np.dot(freqs, mag) / total


def spectral_flux(
    signal: np.ndarray,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> np.ndarray:
    """L2 norm of the frame-to-frame magnitude difference.

    Returns an array of flux values, one per STFT frame (first frame is 0).
    Peaks in flux correspond to onsets or abrupt timbral changes.
    """
    S = stft(signal, frame_size, hop_size)
    mag = np.abs(S)
    diff = np.diff(mag, axis=1)
    flux = np.sqrt(np.sum(diff ** 2, axis=0))
    return np.concatenate([[0.0], flux])


def spectral_gate(
    signal: np.ndarray,
    threshold_db: float,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> np.ndarray:
    """Suppress spectral bins below threshold_db, pass those above.

    Operates in the STFT domain: bins with magnitude < threshold_db are zeroed.
    Returns a time-domain signal via overlap-add synthesis.
    """
    S = stft(signal, frame_size, hop_size)
    mag = np.abs(S)
    phase = np.angle(S)
    mag_db = 20.0 * np.log10(mag + 1e-12)
    mask = (mag_db >= threshold_db).astype(float)
    out = istft(mask * mag * np.exp(1j * phase), hop_size)
    n = len(signal)
    if len(out) >= n:
        return out[:n]
    return np.concatenate([out, np.zeros(n - len(out))])
