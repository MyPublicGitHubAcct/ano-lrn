import numpy as np
from scipy.signal import resample as scipy_resample

from python.time_frequency import istft, stft


def time_stretch(
    signal: np.ndarray,
    rate: float,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> np.ndarray:
    """Phase-vocoder time stretching without changing pitch.

    rate > 1: output is longer (slower playback).
    rate < 1: output is shorter (faster playback).
    """
    if rate == 1.0:
        return signal.copy()
    S = stft(signal, frame_size, hop_size)
    num_frames = S.shape[1]
    out_frames = max(1, int(num_frames * rate))

    freq_bins = np.arange(frame_size // 2 + 1)
    omega = 2 * np.pi * freq_bins * hop_size / frame_size

    phase_acc = np.angle(S[:, 0])
    S_out = np.zeros((frame_size // 2 + 1, out_frames), dtype=complex)

    for i in range(out_frames):
        src = min(int(i / rate), num_frames - 1)
        mag = np.abs(S[:, src])
        if src < num_frames - 1:
            delta = np.angle(S[:, src + 1]) - np.angle(S[:, src]) - omega
            delta -= 2 * np.pi * np.round(delta / (2 * np.pi))
            true_freq = omega + delta
        else:
            true_freq = omega
        S_out[:, i] = mag * np.exp(1j * phase_acc)
        phase_acc += true_freq

    return istft(S_out, hop_size)


def pitch_shift(
    signal: np.ndarray,
    semitones: float,
    fs: int,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> np.ndarray:
    """Pitch shift by `semitones` without changing duration.

    Positive semitones = higher pitch; negative = lower pitch.
    Implemented as phase-vocoder time stretch followed by resampling.
    For pitch UP: stretch longer then resample down (higher freq per sample).
    For pitch DOWN: stretch shorter then resample up (lower freq per sample).
    """
    if semitones == 0.0:
        return signal.copy()
    ratio = 2 ** (semitones / 12.0)
    stretched = time_stretch(signal, ratio, frame_size, hop_size)
    return scipy_resample(stretched, len(signal)).astype(float)
