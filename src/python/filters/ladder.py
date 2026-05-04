import numpy as np


def moog_ladder(
    signal: np.ndarray,
    cutoff: float,
    fs: int = 44100,
    resonance: float = 0.0,
) -> np.ndarray:
    """Zero-delay feedback 4-pole ladder filter (Moog/Minimoog emulation).

    Zavalishin TPT topology: four 1-pole ZDF lowpass stages in series with
    global resonance feedback. resonance in [0, 1] maps to feedback k in
    [0, 4]; k = 4 is the self-oscillation threshold. DC gain is 1/(1 + k).
    """
    g = np.tan(np.pi * cutoff / fs)
    k = resonance * 4.0
    G = g / (1.0 + g)
    inv_1pg = 1.0 / (1.0 + g)

    G2 = G * G
    G3 = G2 * G
    G4 = G3 * G
    denom = 1.0 + k * G4

    s1 = s2 = s3 = s4 = 0.0
    out = np.empty(len(signal))

    for n, x in enumerate(signal):
        S1 = s1 * inv_1pg
        S2 = s2 * inv_1pg
        S3 = s3 * inv_1pg
        S4 = s4 * inv_1pg

        y4 = (G4 * x + G3 * S1 + G2 * S2 + G * S3 + S4) / denom

        u1 = x - k * y4
        v1 = (u1 - s1) * G;  y1 = v1 + s1;  s1 = y1 + v1
        v2 = (y1 - s2) * G;  y2 = v2 + s2;  s2 = y2 + v2
        v3 = (y2 - s3) * G;  y3 = v3 + s3;  s3 = y3 + v3
        v4 = (y3 - s4) * G;  y4 = v4 + s4;  s4 = y4 + v4

        out[n] = y4

    return out
