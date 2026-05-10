"""Ping-pong delay: alternating stereo echoes from a single click."""

import matplotlib.pyplot as plt
import numpy as np

from python.delay import ping_pong_delay

FS = 44100
DURATION = 0.4  # seconds

# Single click (impulse) at t=0
N = int(DURATION * FS)
click = np.zeros(N)
click[0] = 1.0
t_ms = np.arange(N) / FS * 1000

COLORS = ["steelblue", "darkorange", "mediumseagreen", "crimson"]
FEEDBACKS = [0.0, 0.5, 0.7, 0.85]

fig, axes = plt.subplots(len(FEEDBACKS), 2, figsize=(14, 10), sharex=True)

for row, fb in enumerate(FEEDBACKS):
    _, L, R = ping_pong_delay(click, FS, delay_time=0.04, feedback=fb, mix=1.0)

    ax_l = axes[row, 0]
    ax_r = axes[row, 1]

    ax_l.stem(t_ms, L, linefmt="steelblue", markerfmt="o", basefmt="gray",
              label="L", use_line_collection=True)
    ax_r.stem(t_ms, R, linefmt="darkorange", markerfmt="o", basefmt="gray",
              label="R", use_line_collection=True)

    for ax, ch in ((ax_l, "L"), (ax_r, "R")):
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"feedback={fb:.2f} — {ch} channel", fontweight="bold")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.tick_params(labelsize=7)

for ax in axes[-1]:
    ax.set_xlabel("Time (ms)")

fig.suptitle(
    "Ping-Pong Delay — Alternating Stereo Echoes\n"
    "(click input, delay_time=40 ms, mix=1.0)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.show()
