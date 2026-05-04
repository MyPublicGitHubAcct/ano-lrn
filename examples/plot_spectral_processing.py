"""Spectral gate: frequency-domain noise suppression."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_sine, generate_white_noise
from python.spectral import spectral_gate
from python.time_frequency import spectrogram

FS = 44100
DURATION = 0.5
FRAME = 2048
HOP = 512

_, tone = generate_sine(freq=440.0, fs=FS, duration=DURATION)
_, noise = generate_white_noise(fs=FS, duration=DURATION, seed=7)

NOISE_LEVEL = 0.5
noisy = tone + NOISE_LEVEL * noise

THRESHOLDS_DB = [20, 35, 50]
COLORS = ["steelblue", "darkorange", "crimson"]

t_ms = np.arange(len(noisy)) / FS * 1000
freq_axis = np.fft.rfftfreq(FRAME, d=1.0 / FS)


def _spec(sig: np.ndarray) -> np.ndarray:
    return spectrogram(sig, frame_size=FRAME, hop_size=HOP)


frame_t = np.arange(_spec(noisy).shape[1]) * HOP / FS * 1000

fig, axes = plt.subplots(2, len(THRESHOLDS_DB) + 1, figsize=(16, 7))

# Original
axes[0, 0].plot(t_ms, noisy, linewidth=0.5, color="gray")
axes[0, 0].set_title("Noisy Input", fontweight="bold")
axes[0, 0].set_xlabel("Time (ms)", fontsize=7)
axes[0, 0].set_ylabel("Amplitude", fontsize=7)

S = _spec(noisy)
axes[1, 0].imshow(S, aspect="auto", origin="lower",
                  extent=[frame_t[0], frame_t[-1], 0, freq_axis[-1]],
                  vmin=-60, vmax=60, cmap="magma")
axes[1, 0].set_title("Spectrogram — noisy", fontweight="bold")
axes[1, 0].set_xlabel("Time (ms)", fontsize=7)
axes[1, 0].set_ylabel("Frequency (Hz)", fontsize=7)
axes[1, 0].set_ylim(0, 5000)

for col, (thresh, color) in enumerate(zip(THRESHOLDS_DB, COLORS), start=1):
    gated = spectral_gate(noisy, threshold_db=thresh, frame_size=FRAME, hop_size=HOP)
    n = min(len(t_ms), len(gated))
    axes[0, col].plot(t_ms[:n], gated[:n], linewidth=0.6, color=color)
    axes[0, col].plot(t_ms[:n], tone[:n], linewidth=0.7, color="steelblue", alpha=0.4, label="Target")
    axes[0, col].set_title(f"spectral_gate — {thresh} dB", fontsize=9, fontweight="bold")
    axes[0, col].set_xlabel("Time (ms)", fontsize=7)
    axes[0, col].set_ylabel("Amplitude", fontsize=7)
    axes[0, col].legend(fontsize=6)

    S_g = _spec(gated[:n])
    axes[1, col].imshow(S_g, aspect="auto", origin="lower",
                        extent=[frame_t[0], frame_t[-1], 0, freq_axis[-1]],
                        vmin=-60, vmax=60, cmap="magma")
    axes[1, col].set_title(f"Spectrogram — gated {thresh} dB", fontsize=9, fontweight="bold")
    axes[1, col].set_xlabel("Time (ms)", fontsize=7)
    axes[1, col].set_ylabel("Frequency (Hz)", fontsize=7)
    axes[1, col].set_ylim(0, 5000)

for ax in axes.flatten():
    ax.tick_params(labelsize=7)

fig.suptitle("Spectral Gate — Noise Suppression", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
