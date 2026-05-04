"""Diode clip and analog saturate: virtual analog distortion models."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_sine
from python.virtual_analog import analog_saturate, diode_clip

FS = 44100
FREQ = 440.0
DURATION = 3 / FREQ

_, sig = generate_sine(freq=FREQ, fs=FS, duration=DURATION)
t_ms = np.arange(len(sig)) / FS * 1000
x_tc = np.linspace(-1.5, 1.5, 500)

COLORS = ["steelblue", "darkorange", "crimson"]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))

# Transfer curves
ax = axes[0, 0]
ax.plot(x_tc, x_tc, color="gray", linewidth=0.7, linestyle="--", label="Linear")
for thresh, color in zip([1.0, 0.7, 0.4], COLORS):
    ax.plot(x_tc, diode_clip(x_tc, threshold=thresh), color=color, linewidth=1.0, label=f"T={thresh}")
ax.set_title("diode_clip() — transfer curve", fontweight="bold")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.legend(fontsize=7)

ax = axes[1, 0]
ax.plot(x_tc, x_tc, color="gray", linewidth=0.7, linestyle="--", label="Linear")
for drive, color in zip([1.0, 3.0, 6.0], COLORS):
    ax.plot(x_tc, analog_saturate(x_tc * drive / drive, drive=drive), color=color, linewidth=1.0, label=f"drive={drive}")
ax.set_title("analog_saturate() — transfer curve", fontweight="bold")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.legend(fontsize=7)

# Waveforms
THRESHOLDS = [1.0, 0.7, 0.4]
DRIVES = [1.0, 3.0, 6.0]

for col, (thresh, color) in enumerate(zip(THRESHOLDS, COLORS), start=1):
    out = diode_clip(sig, threshold=thresh)
    axes[0, col].plot(t_ms, sig, color="gray", linewidth=0.5, alpha=0.4, label="Input")
    axes[0, col].plot(t_ms, out, color=color, linewidth=0.9, label=f"T={thresh}")
    axes[0, col].set_title(f"diode_clip T={thresh}", fontsize=9, fontweight="bold")
    axes[0, col].legend(fontsize=6)
    axes[0, col].set_xlabel("Time (ms)", fontsize=7)
    axes[0, col].set_ylabel("Amplitude", fontsize=7)

for col, (drive, color) in enumerate(zip(DRIVES, COLORS), start=1):
    out = analog_saturate(sig, drive=drive)
    axes[1, col].plot(t_ms, sig, color="gray", linewidth=0.5, alpha=0.4, label="Input")
    axes[1, col].plot(t_ms, out, color=color, linewidth=0.9, label=f"drive={drive}")
    axes[1, col].set_title(f"analog_saturate drive={drive}", fontsize=9, fontweight="bold")
    axes[1, col].legend(fontsize=6)
    axes[1, col].set_xlabel("Time (ms)", fontsize=7)
    axes[1, col].set_ylabel("Amplitude", fontsize=7)

for ax in axes.flatten():
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.5)

fig.suptitle("Virtual Analog Distortion: diode_clip and analog_saturate", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
