"""Tremolo and ring modulation: amplitude modulation effects."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_sine
from python.modulator import ring_modulate, tremolo

FS = 44100
FREQ = 440.0
DURATION = 0.1

_, carrier = generate_sine(freq=FREQ, fs=FS, duration=DURATION)
t_ms = np.arange(len(carrier)) / FS * 1000


def _spectrum(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / FS)
    db = 20 * np.log10(np.abs(np.fft.rfft(sig)) + 1e-12)
    return freqs, db


fig, axes = plt.subplots(2, 3, figsize=(14, 7))

# tremolo
DEPTHS = [0.0, 0.5, 1.0]
RATE = 5.0
COLORS = ["steelblue", "darkorange", "crimson"]

for depth, color in zip(DEPTHS, COLORS):
    trem = tremolo(carrier, rate=RATE, depth=depth, fs=FS)
    axes[0, 0].plot(t_ms, trem, linewidth=0.8, color=color, label=f"depth={depth}", alpha=0.85)
axes[0, 0].set_title(f"tremolo() — rate={RATE} Hz", fontweight="bold")
axes[0, 0].set_xlabel("Time (ms)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].legend(fontsize=7)

trem_full = tremolo(carrier, rate=RATE, depth=1.0, fs=FS)
freqs, db = _spectrum(trem_full)
mask = (freqs > 0) & (freqs < 2000)
axes[0, 1].plot(freqs[mask], db[mask], linewidth=0.8, color="steelblue")
axes[0, 1].axvline(FREQ, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
axes[0, 1].axvline(FREQ - RATE, color="darkorange", linewidth=0.5, linestyle="--", alpha=0.6, label=f"f±{RATE:.0f}")
axes[0, 1].axvline(FREQ + RATE, color="darkorange", linewidth=0.5, linestyle="--", alpha=0.6)
axes[0, 1].set_title("tremolo() spectrum — sidebands at f±rate", fontweight="bold")
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].set_ylabel("Magnitude (dB)")
axes[0, 1].legend(fontsize=7)

axes[0, 2].set_visible(False)

# ring modulation
CARRIERS = [50.0, 100.0, 200.0]
for c_freq, color in zip(CARRIERS, COLORS):
    rm = ring_modulate(carrier, carrier_freq=c_freq, fs=FS)
    axes[1, 0].plot(t_ms, rm, linewidth=0.8, color=color, label=f"carrier={c_freq:.0f} Hz", alpha=0.85)
axes[1, 0].set_title("ring_modulate()", fontweight="bold")
axes[1, 0].set_xlabel("Time (ms)")
axes[1, 0].set_ylabel("Amplitude")
axes[1, 0].legend(fontsize=7)

C_FREQ = 100.0
rm = ring_modulate(carrier, carrier_freq=C_FREQ, fs=FS)
freqs, db = _spectrum(rm)
mask = (freqs > 0) & (freqs < 2000)
axes[1, 1].plot(freqs[mask], db[mask], linewidth=0.8, color="steelblue")
for f, label, color in [(FREQ - C_FREQ, f"f−{C_FREQ:.0f}", "darkorange"), (FREQ + C_FREQ, f"f+{C_FREQ:.0f}", "crimson")]:
    axes[1, 1].axvline(f, color=color, linewidth=0.5, linestyle="--", alpha=0.8, label=label)
axes[1, 1].set_title(f"ring_modulate() spectrum — sidebands, no carrier", fontweight="bold")
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_ylabel("Magnitude (dB)")
axes[1, 1].legend(fontsize=7)

axes[1, 2].set_visible(False)

for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)

fig.suptitle("Amplitude Modulation: Tremolo and Ring Modulation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
