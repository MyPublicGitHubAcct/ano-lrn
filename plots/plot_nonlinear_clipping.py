"""Hard clip and soft clip: transfer curves and harmonic spectra."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_sine
from python.nonlinear import hard_clip, soft_clip

FS = 44100
FREQ = 440.0
DURATION = 3 / FREQ

_, sig = generate_sine(freq=FREQ, fs=FS, duration=DURATION)
t_ms = np.arange(len(sig)) / FS * 1000
x_tc = np.linspace(-1.5, 1.5, 500)


def _spectrum_db(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(len(s), d=1.0 / FS)
    db = 20 * np.log10(np.abs(np.fft.rfft(s)) + 1e-12)
    return freqs, db


COLORS = ["steelblue", "darkorange", "crimson"]
CLIP_THRESHOLDS = [1.0, 0.7, 0.4]
DRIVES = [1.0, 3.0, 8.0]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))

# Transfer curves
ax = axes[0, 0]
ax.plot(x_tc, x_tc, color="gray", linewidth=0.8, linestyle="--", label="Linear")
for thresh, color in zip(CLIP_THRESHOLDS, COLORS):
    ax.plot(x_tc, hard_clip(x_tc, threshold=thresh), color=color, linewidth=1.0, label=f"T={thresh}")
ax.set_title("hard_clip() — transfer curve", fontweight="bold")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.legend(fontsize=7)

ax = axes[1, 0]
ax.plot(x_tc, x_tc, color="gray", linewidth=0.8, linestyle="--", label="Linear")
for drive, color in zip(DRIVES, COLORS):
    ax.plot(x_tc, soft_clip(x_tc, drive=drive), color=color, linewidth=1.0, label=f"drive={drive}")
ax.set_title("soft_clip() — transfer curve", fontweight="bold")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.legend(fontsize=7)

# Waveforms
for col, (thresh, color) in enumerate(zip(CLIP_THRESHOLDS, COLORS), start=1):
    out = hard_clip(sig, threshold=thresh)
    axes[0, col].plot(t_ms, sig, color="gray", linewidth=0.6, alpha=0.4, label="Input")
    axes[0, col].plot(t_ms, out, color=color, linewidth=0.9, label=f"T={thresh}")
    axes[0, col].axhline(thresh, color=color, linewidth=0.5, linestyle=":", alpha=0.6)
    axes[0, col].axhline(-thresh, color=color, linewidth=0.5, linestyle=":", alpha=0.6)
    axes[0, col].set_title(f"hard_clip T={thresh}", fontsize=9, fontweight="bold")
    axes[0, col].legend(fontsize=6)

for col, (drive, color) in enumerate(zip(DRIVES, COLORS), start=1):
    out = soft_clip(sig, drive=drive)
    axes[1, col].plot(t_ms, sig, color="gray", linewidth=0.6, alpha=0.4, label="Input")
    axes[1, col].plot(t_ms, out, color=color, linewidth=0.9, label=f"drive={drive}")
    axes[1, col].set_title(f"soft_clip drive={drive}", fontsize=9, fontweight="bold")
    axes[1, col].legend(fontsize=6)

for ax in axes.flatten():
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_xlabel("Time (ms)" if ax not in [axes[0, 0], axes[1, 0]] else ax.get_xlabel(), fontsize=7)

for ax in [axes[0, 0], axes[1, 0]]:
    ax.set_xlabel("Input")

fig.suptitle("Clipping: hard_clip and soft_clip", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
