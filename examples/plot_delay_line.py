"""Delay line: impulse response and time-alignment of a sine wave."""

import matplotlib.pyplot as plt
import numpy as np

from python.delay import delay_line
from python.generators import generate_impulse, generate_sine

FS = 44100
FREQ = 440.0
DURATION = 3 / FREQ  # 3 cycles

_, impulse = generate_impulse(fs=FS, duration=0.002)
_, sine = generate_sine(freq=FREQ, fs=FS, duration=DURATION)

DELAYS = [0, 22, 44, 88]
t_imp_ms = np.arange(len(impulse)) / FS * 1000
t_ms = np.arange(len(sine)) / FS * 1000

fig, axes = plt.subplots(2, len(DELAYS), figsize=(14, 6))

for col, d in enumerate(DELAYS):
    imp_delayed = delay_line(impulse, delay_samples=d)
    axes[0, col].stem(t_imp_ms, imp_delayed, linefmt="steelblue", markerfmt="C0o", basefmt="gray")
    axes[0, col].set_title(f"Impulse — D={d} samples\n({d/FS*1000:.2f} ms)", fontsize=9, fontweight="bold")
    axes[0, col].set_xlabel("Time (ms)", fontsize=7)
    axes[0, col].set_ylabel("Amplitude", fontsize=7)
    axes[0, col].tick_params(labelsize=7)
    axes[0, col].grid(True, linewidth=0.3, alpha=0.5)

    sine_delayed = delay_line(sine, delay_samples=d)
    axes[1, col].plot(t_ms, sine, linewidth=0.8, color="steelblue", alpha=0.5, label="Original")
    axes[1, col].plot(t_ms, sine_delayed, linewidth=0.8, color="darkorange", label=f"D={d}")
    axes[1, col].set_title(f"Sine — D={d} samples", fontsize=9, fontweight="bold")
    axes[1, col].set_xlabel("Time (ms)", fontsize=7)
    axes[1, col].set_ylabel("Amplitude", fontsize=7)
    axes[1, col].tick_params(labelsize=7)
    axes[1, col].grid(True, linewidth=0.3, alpha=0.5)
    axes[1, col].legend(fontsize=6)

fig.suptitle("Delay Line — Integer Sample Delay", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
