"""NLMS vs LMS: convergence speed comparison at two input levels."""

import matplotlib.pyplot as plt
import numpy as np

from python.adaptive import lms, nlms
from python.generators import generate_sine, generate_white_noise

FS = 44100
DURATION = 0.3
FREQ = 880.0

_, clean = generate_sine(freq=FREQ, fs=FS, duration=DURATION)
_, noise = generate_white_noise(fs=FS, duration=DURATION, seed=7)

NOISE_LEVEL = 0.5
noisy = clean + NOISE_LEVEL * noise
reference = NOISE_LEVEL * noise

lms_out, lms_err, _ = lms(desired=noisy, reference=reference, filter_order=32, mu=0.002)
nlms_out, nlms_err, _ = nlms(desired=noisy, reference=reference, filter_order=32, mu=0.5)

t_ms = np.arange(len(noisy)) / FS * 1000

fig, axes = plt.subplots(2, 2, figsize=(13, 7))

axes[0, 0].plot(t_ms, noisy, linewidth=0.5, color="darkorange", label="Noisy")
axes[0, 0].plot(t_ms, clean, linewidth=0.8, color="steelblue", alpha=0.7, label="Clean")
axes[0, 0].set_title("Input Signals", fontweight="bold")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].legend(fontsize=7)

axes[0, 1].plot(t_ms, lms_err, linewidth=0.6, color="steelblue", label="LMS (μ=0.002)")
axes[0, 1].plot(t_ms, nlms_err, linewidth=0.6, color="crimson", label="NLMS (μ=0.5)")
axes[0, 1].set_title("Error Comparison: LMS vs NLMS", fontweight="bold")
axes[0, 1].set_ylabel("Error")
axes[0, 1].legend(fontsize=7)

lms_pwr = 10 * np.log10(lms_err ** 2 + 1e-20)
nlms_pwr = 10 * np.log10(nlms_err ** 2 + 1e-20)

axes[1, 0].plot(t_ms, lms_pwr, linewidth=0.7, color="steelblue")
axes[1, 0].set_title("LMS Error Power (dB)", fontweight="bold")
axes[1, 0].set_xlabel("Time (ms)")
axes[1, 0].set_ylabel("Power (dB)")

axes[1, 1].plot(t_ms, nlms_pwr, linewidth=0.7, color="crimson")
axes[1, 1].set_title("NLMS Error Power (dB)", fontweight="bold")
axes[1, 1].set_xlabel("Time (ms)")
axes[1, 1].set_ylabel("Power (dB)")

for ax in axes.flatten():
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)

fig.suptitle("NLMS vs LMS Adaptive Filter", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
