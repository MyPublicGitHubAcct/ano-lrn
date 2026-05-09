"""Wiener filter: STFT-domain source extraction using a spectral mask."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_sine, generate_white_noise
from python.source_separation import wiener_filter

FS = 44100
DURATION = 0.3

_, target = generate_sine(freq=440.0, fs=FS, duration=DURATION)
_, interferer = generate_sine(freq=1000.0, fs=FS, duration=DURATION)
_, noise = generate_white_noise(fs=FS, duration=DURATION, seed=5)

NOISE_LEVEL = 0.3
mixture = target + NOISE_LEVEL * interferer + 0.1 * noise

# Estimate: use the clean target as oracle estimate (best case)
extracted = wiener_filter(mixture, source_estimate=target)

t_ms = np.arange(len(mixture)) / FS * 1000


def _spectrum(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(len(s), d=1.0 / FS)
    db = 20 * np.log10(np.abs(np.fft.rfft(s)) + 1e-12)
    return freqs, db


fig, axes = plt.subplots(2, 2, figsize=(13, 7))

axes[0, 0].plot(t_ms, mixture, linewidth=0.6, color="darkorange", label="Mixture")
axes[0, 0].plot(t_ms, target, linewidth=0.6, color="steelblue", alpha=0.6, label="Target")
axes[0, 0].set_title("Input: Mixture vs Target", fontweight="bold")
axes[0, 0].set_xlabel("Time (ms)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].legend(fontsize=7)

n = min(len(mixture), len(extracted))
axes[0, 1].plot(t_ms[:n], target[:n], linewidth=0.7, color="steelblue", alpha=0.6, label="Target")
axes[0, 1].plot(t_ms[:n], extracted[:n], linewidth=0.8, color="mediumseagreen", label="Extracted")
axes[0, 1].set_title("Wiener Filter Output vs Target", fontweight="bold")
axes[0, 1].set_xlabel("Time (ms)")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].legend(fontsize=7)

freqs, db_mix = _spectrum(mixture)
freqs, db_tgt = _spectrum(target)
freqs, db_ext = _spectrum(extracted[:n])
mask = (freqs > 0) & (freqs < 3000)
axes[1, 0].plot(freqs[mask], db_mix[mask], linewidth=0.8, color="darkorange", label="Mixture")
axes[1, 0].plot(freqs[mask], db_tgt[mask], linewidth=0.8, color="steelblue", alpha=0.7, label="Target")
axes[1, 0].set_title("Spectrum: Mixture vs Target", fontweight="bold")
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("Magnitude (dB)")
axes[1, 0].legend(fontsize=7)

axes[1, 1].plot(freqs[mask], db_tgt[mask], linewidth=0.8, color="steelblue", alpha=0.7, label="Target")
axes[1, 1].plot(freqs[mask], db_ext[mask], linewidth=0.8, color="mediumseagreen", label="Extracted")
axes[1, 1].set_title("Spectrum: Extracted vs Target", fontweight="bold")
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_ylabel("Magnitude (dB)")
axes[1, 1].legend(fontsize=7)

for ax in axes.flatten():
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)

fig.suptitle("Wiener Filter — Source Extraction", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
