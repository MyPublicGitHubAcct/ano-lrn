"""HPSS: harmonic-percussive source separation via median filtering."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_sine, generate_white_noise
from python.source_separation import hpss
from python.time_frequency import spectrogram

FS = 44100
DURATION = 0.5
FRAME = 2048
HOP = 512

# Synthetic mixture: sustained tone (harmonic) + noise bursts (percussive)
_, tone = generate_sine(freq=440.0, fs=FS, duration=DURATION)
_, noise_raw = generate_white_noise(fs=FS, duration=DURATION, seed=3)

# Create percussive bursts
perc = np.zeros_like(noise_raw)
for onset in [0.1, 0.25, 0.4]:
    idx = int(onset * FS)
    burst_len = int(0.01 * FS)
    perc[idx : idx + burst_len] = noise_raw[idx : idx + burst_len] * 3.0

mixture = tone + perc
harmonic, percussive = hpss(mixture, fs=FS, frame_size=FRAME, hop_size=HOP, kernel_size=17)

t = np.linspace(0, DURATION, len(mixture))
t_ms = t * 1000

fig, axes = plt.subplots(2, 3, figsize=(14, 7))


def _spec(sig: np.ndarray) -> np.ndarray:
    return spectrogram(sig, frame_size=FRAME, hop_size=HOP)


times = np.arange(_spec(mixture).shape[1]) * HOP / FS * 1000
freq_bins = np.fft.rfftfreq(FRAME, d=1.0 / FS)

for col, (label, sig, color) in enumerate([
    ("Mixture", mixture, "steelblue"),
    ("Harmonic", harmonic, "darkorange"),
    ("Percussive", percussive, "crimson"),
]):
    axes[0, col].plot(t_ms, sig, linewidth=0.5, color=color)
    axes[0, col].set_title(f"{label} — time domain", fontsize=9, fontweight="bold")
    axes[0, col].set_xlabel("Time (ms)", fontsize=7)
    axes[0, col].set_ylabel("Amplitude", fontsize=7)
    axes[0, col].tick_params(labelsize=7)
    axes[0, col].grid(True, linewidth=0.3, alpha=0.5)

    S = _spec(sig)
    axes[1, col].imshow(
        S,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 0, freq_bins[-1]],
        vmin=-60, vmax=0,
        cmap="magma",
    )
    axes[1, col].set_title(f"{label} — spectrogram", fontsize=9, fontweight="bold")
    axes[1, col].set_xlabel("Time (ms)", fontsize=7)
    axes[1, col].set_ylabel("Frequency (Hz)", fontsize=7)
    axes[1, col].set_ylim(0, 4000)
    axes[1, col].tick_params(labelsize=7)

fig.suptitle("HPSS — Harmonic-Percussive Source Separation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
