"""Multitap delay: rhythmic eight-tap stereo echo pattern on an impulse train."""

import matplotlib.pyplot as plt
import numpy as np

from python.delay import multitap_delay

FS = 44100
DURATION = 1.2  # seconds
N = int(DURATION * FS)

# Impulse train: one click every 0.3 s
impulse_train = np.zeros(N)
for k in range(4):
    impulse_train[round(k * 0.3 * FS)] = 1.0

t_ms = np.arange(N) / FS * 1000

# Eight taps: varied delays, gains, and pan positions spread across the stereo field
TAPS = [
    (0.04,  1.00, -0.9),   # 40 ms  — hard left
    (0.07,  0.80,  0.6),   # 70 ms  — right
    (0.11,  0.70, -0.5),   # 110 ms — left
    (0.15,  0.60,  0.9),   # 150 ms — hard right
    (0.20,  0.50, -0.3),   # 200 ms — slightly left
    (0.26,  0.40,  0.3),   # 260 ms — slightly right
    (0.33,  0.30, -0.8),   # 330 ms — left
    (0.42,  0.20,  0.7),   # 420 ms — right
]

_, L, R = multitap_delay(impulse_train, FS, TAPS)

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

# Input
axes[0].stem(t_ms, impulse_train, linefmt="gray", markerfmt="^", basefmt="gray",
             use_line_collection=True)
axes[0].set_ylabel("Amplitude")
axes[0].set_ylim(-0.1, 1.2)
axes[0].set_title("Input — Impulse Train (one click every 300 ms)", fontweight="bold")
axes[0].grid(True, linewidth=0.3, alpha=0.5)
axes[0].tick_params(labelsize=7)

# Left channel
axes[1].stem(t_ms, L, linefmt="steelblue", markerfmt="o", basefmt="gray",
             use_line_collection=True)
axes[1].set_ylabel("Amplitude")
axes[1].set_title("L channel output", fontweight="bold")
axes[1].grid(True, linewidth=0.3, alpha=0.5)
axes[1].tick_params(labelsize=7)

# Right channel
axes[2].stem(t_ms, R, linefmt="darkorange", markerfmt="o", basefmt="gray",
             use_line_collection=True)
axes[2].set_ylabel("Amplitude")
axes[2].set_xlabel("Time (ms)")
axes[2].set_title("R channel output", fontweight="bold")
axes[2].grid(True, linewidth=0.3, alpha=0.5)
axes[2].tick_params(labelsize=7)

# Tap annotation on L and R panels
for i, (dt, gain, pan) in enumerate(TAPS):
    dt_ms = dt * 1000
    axes[1].axvline(dt_ms, color="steelblue", linewidth=0.6, linestyle="--", alpha=0.4)
    axes[2].axvline(dt_ms, color="darkorange", linewidth=0.6, linestyle="--", alpha=0.4)
    axes[1].text(dt_ms + 1, 0.9, f"T{i+1}", fontsize=6, color="steelblue", alpha=0.7)
    axes[2].text(dt_ms + 1, 0.9, f"T{i+1}", fontsize=6, color="darkorange", alpha=0.7)

fig.suptitle(
    "Multitap Fractional Delay — Eight-Tap Stereo Echo Pattern\n"
    "(constant-power panning; each tap: varied delay, gain, and pan)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.show()
