"""Resample: polyphase sample-rate conversion."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_sine
from python.warping import resample

FS_ORIG = 44100
FREQ = 440.0
DURATION = 0.05

_, sig = generate_sine(freq=FREQ, fs=FS_ORIG, duration=DURATION)
t_orig = np.arange(len(sig)) / FS_ORIG * 1000

TARGETS = [22050, 48000, 96000, 8000]
COLORS = ["steelblue", "darkorange", "mediumseagreen", "crimson"]

fig, axes = plt.subplots(2, len(TARGETS), figsize=(16, 7))

for col, (fs_target, color) in enumerate(zip(TARGETS, COLORS)):
    resampled = resample(sig, orig_fs=FS_ORIG, target_fs=fs_target)
    t_res = np.arange(len(resampled)) / fs_target * 1000
    ratio = fs_target / FS_ORIG

    axes[0, col].plot(t_orig, sig, linewidth=0.8, color="gray", alpha=0.5, label=f"Original ({FS_ORIG})")
    axes[0, col].plot(t_res, resampled, linewidth=0.8, color=color, label=f"Resampled ({fs_target})")
    axes[0, col].set_title(f"{FS_ORIG} → {fs_target} Hz\n({len(sig)} → {len(resampled)} samples)",
                           fontsize=9, fontweight="bold")
    axes[0, col].set_xlabel("Time (ms)", fontsize=7)
    axes[0, col].set_ylabel("Amplitude", fontsize=7)
    axes[0, col].legend(fontsize=6)
    axes[0, col].tick_params(labelsize=7)
    axes[0, col].grid(True, linewidth=0.3, alpha=0.5)

    # Spectra comparison
    freqs_orig = np.fft.rfftfreq(len(sig), d=1.0 / FS_ORIG)
    db_orig = 20 * np.log10(np.abs(np.fft.rfft(sig)) + 1e-12)
    freqs_res = np.fft.rfftfreq(len(resampled), d=1.0 / fs_target)
    db_res = 20 * np.log10(np.abs(np.fft.rfft(resampled)) + 1e-12)
    f_max = min(FS_ORIG, fs_target) / 2
    mask_o = freqs_orig <= f_max
    mask_r = freqs_res <= f_max
    axes[1, col].plot(freqs_orig[mask_o], db_orig[mask_o], color="gray", linewidth=0.8, alpha=0.5, label="Original")
    axes[1, col].plot(freqs_res[mask_r], db_res[mask_r], color=color, linewidth=0.8, label="Resampled")
    axes[1, col].set_title(f"Spectrum (0–{f_max/1000:.0f} kHz)", fontsize=9, fontweight="bold")
    axes[1, col].set_xlabel("Frequency (Hz)", fontsize=7)
    axes[1, col].set_ylabel("Magnitude (dB)", fontsize=7)
    axes[1, col].legend(fontsize=6)
    axes[1, col].tick_params(labelsize=7)
    axes[1, col].grid(True, linewidth=0.3, alpha=0.5)

fig.suptitle("Polyphase Resampling", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
