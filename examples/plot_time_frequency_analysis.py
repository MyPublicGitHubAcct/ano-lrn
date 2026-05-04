"""Spectrogram: log-magnitude STFT for visualisation and analysis."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_chirp, generate_sine, generate_white_noise
from python.time_frequency import spectrogram

FS = 44100
DURATION = 0.5
FRAME = 2048
HOP = 512

_, chirp = generate_chirp(start_freq=100.0, end_freq=8000.0, fs=FS, duration=DURATION, method="logarithmic")
_, sine = generate_sine(freq=1000.0, fs=FS, duration=DURATION)
_, noise = generate_white_noise(fs=FS, duration=DURATION, seed=0)
mixture = chirp + 0.3 * sine + 0.2 * noise

SIGNALS = [
    ("Log Chirp", chirp, "inferno"),
    ("Sine 1 kHz", sine, "magma"),
    ("Mixture", mixture, "plasma"),
]

freq_axis = np.fft.rfftfreq(FRAME, d=1.0 / FS)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (label, sig, cmap) in zip(axes, SIGNALS):
    S = spectrogram(sig, frame_size=FRAME, hop_size=HOP)
    frame_t = np.arange(S.shape[1]) * HOP / FS * 1000
    im = ax.imshow(
        S,
        aspect="auto",
        origin="lower",
        extent=[frame_t[0], frame_t[-1], 0, freq_axis[-1]],
        vmin=S.max() - 60,
        vmax=S.max(),
        cmap=cmap,
    )
    ax.set_title(f"Spectrogram — {label}", fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 8000)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, label="dB", shrink=0.8)

fig.suptitle("Spectrogram — Log-Magnitude STFT", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
