import numpy as np

from python.generators._helpers import _time_axis


def generate_chirp(
    f_start: float = 20.0,
    f_end: float = 20000.0,
    fs: int = 44100,
    duration: float = 1.0,
    amplitude: float = 1.0,
    method: str = "logarithmic",
) -> tuple[np.ndarray, np.ndarray]:
    t = _time_axis(fs, duration)
    if method == "linear":
        phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration))
    else:
        # Exponential sweep: constant ratio per octave, preferred for audio testing
        k = np.log(f_end / f_start) / duration
        phase = 2 * np.pi * f_start * (np.exp(k * t) - 1) / k
    return t, amplitude * np.sin(phase)
