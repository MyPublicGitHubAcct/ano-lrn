"""Time stretch and pitch shift: phase-vocoder processing."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_sine
from python.warping import pitch_shift, time_stretch

FS = 44100
FREQ = 440.0
DURATION = 0.3

_, sig = generate_sine(freq=FREQ, fs=FS, duration=DURATION)
t_orig = np.arange(len(sig)) / FS * 1000

RATES = [0.5, 1.0, 2.0]
SEMITONES = [-12, 0, 7, 12]
COLORS = ["steelblue", "mediumseagreen", "crimson"]
PITCH_COLORS = ["#1565c0", "steelblue", "darkorange", "crimson"]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))

# time_stretch
for col, (rate, color) in enumerate(zip(RATES, COLORS)):
    stretched = time_stretch(sig, rate=rate, frame_size=1024, hop_size=256)
    t_s = np.arange(len(stretched)) / FS * 1000
    axes[0, col].plot(t_s, stretched, linewidth=0.7, color=color)
    axes[0, col].set_title(f"time_stretch — rate={rate}\n({len(stretched)} samples, {len(stretched)/FS*1000:.0f} ms)",
                           fontsize=9, fontweight="bold")
    axes[0, col].set_xlabel("Time (ms)", fontsize=7)
    axes[0, col].set_ylabel("Amplitude", fontsize=7)
    axes[0, col].tick_params(labelsize=7)
    axes[0, col].grid(True, linewidth=0.3, alpha=0.5)

# stretch spectrum: pitch should be unchanged
axes[0, 3].set_title("time_stretch() spectra\n(all at 440 Hz despite different lengths)", fontsize=9, fontweight="bold")
for rate, color in zip(RATES, COLORS):
    stretched = time_stretch(sig, rate=rate, frame_size=1024, hop_size=256)
    freqs = np.fft.rfftfreq(len(stretched), d=1.0 / FS)
    db = 20 * np.log10(np.abs(np.fft.rfft(stretched)) + 1e-12)
    mask = (freqs > 200) & (freqs < 1000)
    axes[0, 3].plot(freqs[mask], db[mask], linewidth=0.8, color=color, label=f"rate={rate}")
axes[0, 3].axvline(FREQ, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
axes[0, 3].legend(fontsize=7)
axes[0, 3].set_xlabel("Frequency (Hz)", fontsize=7)
axes[0, 3].set_ylabel("Magnitude (dB)", fontsize=7)
axes[0, 3].tick_params(labelsize=7)
axes[0, 3].grid(True, linewidth=0.3, alpha=0.5)

# pitch_shift
for col, (semitones, color) in enumerate(zip(SEMITONES, PITCH_COLORS)):
    shifted = pitch_shift(sig, semitones=semitones, fs=FS, frame_size=1024, hop_size=256)
    t_sh = np.arange(len(shifted)) / FS * 1000
    expected_freq = FREQ * (2 ** (semitones / 12))
    axes[1, col].plot(t_sh[:len(t_orig)], shifted[:len(t_orig)], linewidth=0.7, color=color,
                      label=f"{semitones:+d} st → {expected_freq:.0f} Hz")
    axes[1, col].set_title(f"pitch_shift {semitones:+d} semitones\n(expected: {expected_freq:.0f} Hz)",
                           fontsize=9, fontweight="bold")
    axes[1, col].set_xlabel("Time (ms)", fontsize=7)
    axes[1, col].set_ylabel("Amplitude", fontsize=7)
    axes[1, col].tick_params(labelsize=7)
    axes[1, col].grid(True, linewidth=0.3, alpha=0.5)

fig.suptitle("Phase-Vocoder: Time Stretch and Pitch Shift", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
