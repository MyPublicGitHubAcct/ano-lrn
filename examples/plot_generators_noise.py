"""Noise generators: white noise and pink noise — time domain and power spectrum."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_pink_noise, generate_white_noise

FS = 44100
DURATION = 0.5
SEED = 42

t_white, white = generate_white_noise(fs=FS, duration=DURATION, seed=SEED)
t_pink, pink = generate_pink_noise(fs=FS, duration=DURATION, seed=SEED)


def _power_spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs_hz, power_db) averaged into 1/3-octave-width bins."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    power = np.abs(np.fft.rfft(signal)) ** 2 / n
    db = 10.0 * np.log10(power + 1e-20)
    return freqs, db


fig, axes = plt.subplots(2, 2, figsize=(13, 7))

WINDOW_MS = 20
n_win = int(WINDOW_MS * FS / 1000)
t_ms = t_white[:n_win] * 1000

# Time domain
axes[0, 0].plot(t_ms, white[:n_win], linewidth=0.6, color="steelblue")
axes[0, 0].set_title("White Noise — time domain (first 20 ms)", fontsize=10, fontweight="bold")
axes[0, 0].set_xlabel("Time (ms)", fontsize=8)
axes[0, 0].set_ylabel("Amplitude", fontsize=8)

axes[0, 1].plot(t_ms, pink[:n_win], linewidth=0.6, color="darkorange")
axes[0, 1].set_title("Pink Noise — time domain (first 20 ms)", fontsize=10, fontweight="bold")
axes[0, 1].set_xlabel("Time (ms)", fontsize=8)
axes[0, 1].set_ylabel("Amplitude", fontsize=8)

# Power spectrum
freqs_w, db_w = _power_spectrum(white)
freqs_p, db_p = _power_spectrum(pink)

mask = freqs_w > 0

axes[1, 0].semilogx(freqs_w[mask], db_w[mask], linewidth=0.6, color="steelblue")
axes[1, 0].set_title("White Noise — power spectrum (flat)", fontsize=10, fontweight="bold")
axes[1, 0].set_xlabel("Frequency (Hz)", fontsize=8)
axes[1, 0].set_ylabel("Power (dB)", fontsize=8)
axes[1, 0].set_xlim(20, FS / 2)

axes[1, 1].semilogx(freqs_p[mask], db_p[mask], linewidth=0.6, color="darkorange")
axes[1, 1].set_title("Pink Noise — power spectrum (−3 dB/octave)", fontsize=10, fontweight="bold")
axes[1, 1].set_xlabel("Frequency (Hz)", fontsize=8)
axes[1, 1].set_ylabel("Power (dB)", fontsize=8)
axes[1, 1].set_xlim(20, FS / 2)

for ax in axes.flatten():
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)

fig.suptitle("Noise Generators", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
