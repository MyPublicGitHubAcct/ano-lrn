import numpy as np


def _vactrol(control: np.ndarray, fs: int, attack_time: float, release_time: float) -> np.ndarray:
    """Asymmetric one-pole smoother modelling an LED/LDR optical coupler.

    Attack is fast (LED turns on quickly); release is slow (LDR resistance
    decays logarithmically after the LED dims). This asymmetry is the
    defining sonic characteristic of the Buchla 292 sound.
    """
    alpha_a = 1.0 - np.exp(-1.0 / (attack_time * fs))
    alpha_r = 1.0 - np.exp(-1.0 / (release_time * fs))
    brightness = np.zeros(len(control))
    b = 0.0
    for i, c in enumerate(control):
        alpha = alpha_a if c > b else alpha_r
        b += alpha * (c - b)
        brightness[i] = b
    return brightness


def lowpass_gate(
    signal: np.ndarray,
    control: np.ndarray,
    fs: int = 44100,
    mode: str = "both",
    attack_time: float = 0.010,
    release_time: float = 0.200,
    max_cutoff: float = 8000.0,
    min_cutoff: float = 20.0,
) -> np.ndarray:
    """Buchla 292c/292e lowpass gate emulation.

    Combines a vactrol-controlled 4-pole lowpass filter with a VCA. The
    vactrol (LED + photoresistor) model gives a fast attack and a slow,
    mushy release that is the signature of Buchla percussion and plucked
    sounds.

    Parameters
    ----------
    signal:
        Audio input, any amplitude.
    control:
        Control signal in [0, 1]. 1 = fully open, 0 = closed. Typically a
        gate or trigger (unit-step edges), but any envelope works.
    fs:
        Sample rate in Hz.
    mode:
        ``"both"`` (default) — lowpass filter + VCA together (most
        characteristic); ``"lowpass"`` — filter only, no amplitude
        control; ``"amplitude"`` — VCA only, no filtering.
    attack_time:
        Vactrol attack time constant in seconds (default 10 ms).
    release_time:
        Vactrol release time constant in seconds (default 200 ms).
    max_cutoff:
        Filter cutoff at control = 1 (Hz).
    min_cutoff:
        Filter cutoff at control = 0 (Hz). Must be > 0.

    Returns
    -------
    np.ndarray
        Processed audio, same length as ``signal``.
    """
    ctrl = np.clip(np.asarray(control, dtype=float), 0.0, 1.0)
    if len(ctrl) != len(signal):
        raise ValueError("control must be the same length as signal")

    brightness = _vactrol(ctrl, fs, attack_time, release_time)

    log_min = np.log(min_cutoff)
    log_max = np.log(max_cutoff)
    cutoff_hz = np.exp(log_min + brightness * (log_max - log_min))

    out = np.zeros(len(signal))

    if mode == "amplitude":
        return signal * brightness

    # Lowpass filter (4-pole one-pole cascade, no resonance)
    s = np.zeros(4)
    for i, x in enumerate(signal):
        fc = min(max(cutoff_hz[i] / fs, 1e-6), 0.499)
        g = 1.0 - np.exp(-2.0 * np.pi * fc)
        s[0] += g * (x - s[0])
        s[1] += g * (s[0] - s[1])
        s[2] += g * (s[1] - s[2])
        s[3] += g * (s[2] - s[3])
        out[i] = s[3] * brightness[i] if mode == "both" else s[3]

    return out
