"""apply_window: window shapes, spectral leakage comparison."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_sine
from python.time_segment import apply_window

FS = 44100
FREQ = 440.0
FRAME = 1024

# Frequency not aligned to a bin — maximises leakage
FREQ_MISALIGNED = 440.7
t = np.arange(FRAME) / FS
sig = np.sin(2 * np.pi * FREQ_MISALIGNED * t)

WINDOWS = ["rectangular", "hann", "hamming", "blackman"]
COLORS = ["steelblue", "darkorange", "mediumseagreen", "crimson"]

freqs = np.fft.rfftfreq(FRAME, d=1.0 / FS)

fig, axes = plt.subplots(2, len(WINDOWS), figsize=(16, 7))

for col, (wtype, color) in enumerate(zip(WINDOWS, COLORS)):
    windowed = apply_window(sig, window_type=wtype)
    t_ms = np.arange(FRAME) / FS * 1000

    axes[0, col].plot(t_ms, windowed, linewidth=0.7, color=color)
    axes[0, col].set_title(f"{wtype}", fontsize=10, fontweight="bold")
    axes[0, col].set_xlabel("Time (ms)", fontsize=7)
    axes[0, col].set_ylabel("Amplitude", fontsize=7)
    axes[0, col].tick_params(labelsize=7)
    axes[0, col].grid(True, linewidth=0.3, alpha=0.5)

    spectrum = np.abs(np.fft.rfft(windowed))
    db = 20 * np.log10(spectrum + 1e-12)
    db -= db.max()  # normalise to 0 dB peak
    mask = (freqs > 200) & (freqs < 1000)
    axes[1, col].plot(freqs[mask], db[mask], linewidth=0.8, color=color)
    axes[1, col].set_title(f"{wtype} — spectrum (normalised)", fontsize=9, fontweight="bold")
    axes[1, col].set_xlabel("Frequency (Hz)", fontsize=7)
    axes[1, col].set_ylabel("Magnitude (dB)", fontsize=7)
    axes[1, col].set_ylim(-80, 5)
    axes[1, col].tick_params(labelsize=7)
    axes[1, col].grid(True, linewidth=0.3, alpha=0.5)

fig.suptitle("apply_window() — Spectral Leakage Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
