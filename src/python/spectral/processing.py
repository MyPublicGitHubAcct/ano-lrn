import numpy as np

from python.time_frequency import istft, stft


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
