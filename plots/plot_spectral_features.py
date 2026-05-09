"""Spectral centroid and flux: frequency-domain feature extraction."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_chirp, generate_white_noise
from python.spectral import spectral_centroid, spectral_flux

FS = 44100
DURATION = 1.0
FRAME = 2048
HOP = 512

_, chirp = generate_chirp(start_freq=100.0, end_freq=8000.0, fs=FS, duration=DURATION, method="logarithmic")
_, noise = generate_white_noise(fs=FS, duration=DURATION, seed=0)

chirp_centroid = spectral_centroid(chirp, fs=FS, frame_size=FRAME, hop_size=HOP)
noise_centroid = spectral_centroid(noise, fs=FS, frame_size=FRAME, hop_size=HOP)
chirp_flux = spectral_flux(chirp, frame_size=FRAME, hop_size=HOP)
noise_flux = spectral_flux(noise, frame_size=FRAME, hop_size=HOP)

frame_times = np.arange(len(chirp_centroid)) * HOP / FS * 1000
t_ms = np.arange(len(chirp)) / FS * 1000

fig, axes = plt.subplots(2, 3, figsize=(14, 7))

axes[0, 0].plot(t_ms, chirp, linewidth=0.5, color="steelblue")
axes[0, 0].set_title("Log Chirp (100–8000 Hz)", fontweight="bold")
axes[0, 0].set_xlabel("Time (ms)")
axes[0, 0].set_ylabel("Amplitude")

axes[0, 1].plot(frame_times, chirp_centroid / 1000, linewidth=0.9, color="steelblue")
axes[0, 1].set_title("Centroid — Chirp (rises with frequency)", fontweight="bold")
axes[0, 1].set_xlabel("Time (ms)")
axes[0, 1].set_ylabel("Centroid (kHz)")

axes[0, 2].plot(frame_times, chirp_flux, linewidth=0.8, color="crimson")
axes[0, 2].set_title("Spectral Flux — Chirp (high = rapid change)", fontweight="bold")
axes[0, 2].set_xlabel("Time (ms)")
axes[0, 2].set_ylabel("Flux")

axes[1, 0].plot(t_ms, noise, linewidth=0.5, color="darkorange")
axes[1, 0].set_title("White Noise", fontweight="bold")
axes[1, 0].set_xlabel("Time (ms)")
axes[1, 0].set_ylabel("Amplitude")

axes[1, 1].plot(frame_times, noise_centroid / 1000, linewidth=0.8, color="darkorange")
axes[1, 1].set_title("Centroid — White Noise (≈ flat at FS/4)", fontweight="bold")
axes[1, 1].set_xlabel("Time (ms)")
axes[1, 1].set_ylabel("Centroid (kHz)")

axes[1, 2].plot(frame_times, noise_flux, linewidth=0.7, color="mediumseagreen")
axes[1, 2].set_title("Spectral Flux — White Noise (random each frame)", fontweight="bold")
axes[1, 2].set_xlabel("Time (ms)")
axes[1, 2].set_ylabel("Flux")

for ax in axes.flatten():
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)

fig.suptitle("Spectral Features — Centroid and Flux", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
