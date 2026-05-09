"""Haas precedence effect: inter-channel delay creates spatial width."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_sine
from python.spatial import haas

FS = 44100
FREQ = 440.0
DURATION = 0.05

_, mono = generate_sine(freq=FREQ, fs=FS, duration=DURATION)
t_ms = np.arange(len(mono)) / FS * 1000

DELAYS = [0, 110, 441, 882, 1764]  # 0, 2.5, 10, 20, 40 ms
COLORS = ["steelblue", "darkorange", "mediumseagreen", "crimson", "mediumpurple"]

fig, axes = plt.subplots(2, len(DELAYS), figsize=(16, 6))

for col, (d, color) in enumerate(zip(DELAYS, COLORS)):
    L, R = haas(mono, delay_samples=d)
    ms = d / FS * 1000
    axes[0, col].plot(t_ms, L, linewidth=0.8, color="steelblue", label="L")
    axes[0, col].plot(t_ms, R, linewidth=0.8, color=color, alpha=0.7, label="R")
    axes[0, col].set_title(f"D={d} samp\n({ms:.1f} ms)", fontsize=9, fontweight="bold")
    axes[0, col].set_xlabel("Time (ms)", fontsize=7)
    axes[0, col].set_ylabel("Amplitude", fontsize=7)
    axes[0, col].tick_params(labelsize=6)
    axes[0, col].legend(fontsize=5)
    axes[0, col].grid(True, linewidth=0.3, alpha=0.5)

    # L-R difference (reveals the delay)
    diff = L - R
    axes[1, col].plot(t_ms, diff, linewidth=0.7, color=color)
    axes[1, col].set_title(f"L − R (D={d})", fontsize=9, fontweight="bold")
    axes[1, col].set_xlabel("Time (ms)", fontsize=7)
    axes[1, col].set_ylabel("L − R", fontsize=7)
    axes[1, col].tick_params(labelsize=6)
    axes[1, col].grid(True, linewidth=0.3, alpha=0.5)

fig.suptitle("Haas Precedence Effect — Inter-Channel Delay", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
