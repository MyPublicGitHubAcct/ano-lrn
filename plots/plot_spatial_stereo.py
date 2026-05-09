"""Pan and stereo widen: equal-power panning and M/S width control."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_sine
from python.spatial import pan, stereo_widen

FS = 44100
FREQ = 440.0
DURATION = 3 / FREQ

_, mono = generate_sine(freq=FREQ, fs=FS, duration=DURATION)
t_ms = np.arange(len(mono)) / FS * 1000

POSITIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]
COLORS = ["#1565c0", "#42a5f5", "#4caf50", "#ff9800", "#c62828"]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))

# pan: L/R amplitudes vs position
pos_range = np.linspace(-1, 1, 200)
theta = (pos_range + 1) * np.pi / 4
L_gain = np.cos(theta)
R_gain = np.sin(theta)
axes[0, 0].plot(pos_range, L_gain, color="steelblue", linewidth=1.2, label="Left")
axes[0, 0].plot(pos_range, R_gain, color="darkorange", linewidth=1.2, label="Right")
axes[0, 0].set_title("pan() — Equal-power gain law", fontweight="bold")
axes[0, 0].set_xlabel("Position")
axes[0, 0].set_ylabel("Gain")
axes[0, 0].legend(fontsize=8)

# pan: waveforms at each position
for pos, color in zip(POSITIONS, COLORS):
    L, R = pan(mono, position=pos)
    axes[0, 1].plot(t_ms, L, linewidth=0.8, color=color, label=f"L @ {pos:+.1f}", alpha=0.8)
axes[0, 1].set_title("pan() — Left channel", fontweight="bold")
axes[0, 1].set_xlabel("Time (ms)")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].legend(fontsize=6)

for pos, color in zip(POSITIONS, COLORS):
    L, R = pan(mono, position=pos)
    axes[0, 2].plot(t_ms, R, linewidth=0.8, color=color, label=f"R @ {pos:+.1f}", alpha=0.8)
axes[0, 2].set_title("pan() — Right channel", fontweight="bold")
axes[0, 2].set_xlabel("Time (ms)")
axes[0, 2].set_ylabel("Amplitude")
axes[0, 2].legend(fontsize=6)

# stereo_widen: start with a panned stereo pair
L_in, R_in = pan(mono, position=0.3)
WIDTHS = [0.0, 0.5, 1.0, 1.5, 2.0]
W_COLORS = ["#1565c0", "#42a5f5", "#4caf50", "#ff9800", "#c62828"]

for width, color in zip(WIDTHS, W_COLORS):
    L_w, R_w = stereo_widen(L_in, R_in, width=width)
    axes[1, 0].plot(t_ms, L_w, linewidth=0.8, color=color, label=f"w={width}", alpha=0.85)
axes[1, 0].set_title("stereo_widen() — Left output", fontweight="bold")
axes[1, 0].set_xlabel("Time (ms)")
axes[1, 0].set_ylabel("Amplitude")
axes[1, 0].legend(fontsize=6)

for width, color in zip(WIDTHS, W_COLORS):
    L_w, R_w = stereo_widen(L_in, R_in, width=width)
    axes[1, 1].plot(t_ms, R_w, linewidth=0.8, color=color, label=f"w={width}", alpha=0.85)
axes[1, 1].set_title("stereo_widen() — Right output", fontweight="bold")
axes[1, 1].set_xlabel("Time (ms)")
axes[1, 1].set_ylabel("Amplitude")
axes[1, 1].legend(fontsize=6)

# Side level vs width
side_rms = []
for w in np.linspace(0, 2, 100):
    _, R_w = stereo_widen(L_in, R_in, width=w)
    L_w, _ = stereo_widen(L_in, R_in, width=w)
    side = (L_w - R_w) * 0.5
    side_rms.append(np.sqrt(np.mean(side ** 2)))
axes[1, 2].plot(np.linspace(0, 2, 100), side_rms, color="steelblue", linewidth=1.2)
axes[1, 2].set_title("stereo_widen() — Side RMS vs width", fontweight="bold")
axes[1, 2].set_xlabel("Width")
axes[1, 2].set_ylabel("Side RMS")

for ax in axes.flatten():
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)

fig.suptitle("Spatial — Panning and Stereo Width", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
