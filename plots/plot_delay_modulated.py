"""Chorus and flanger: waveform and frequency-response visualisations."""

import matplotlib.pyplot as plt
import numpy as np

from python.delay import chorus, flanger
from python.generators import generate_impulse, generate_sine

FS = 44100
DURATION = 0.5

_, impulse = generate_impulse(fs=FS, duration=DURATION)
_, sine = generate_sine(freq=440.0, fs=FS, duration=0.05)
t_ms = np.arange(len(sine)) / FS * 1000


def _freq_response(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(np.fft.rfft(h))
    freqs = np.fft.rfftfreq(len(h), d=1.0 / FS)
    db = 20.0 * np.log10(spectrum + 1e-12)
    return freqs, db


COLORS = ["steelblue", "darkorange", "mediumseagreen", "crimson"]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# ── chorus: waveform ──
ax = axes[0, 0]
ax.plot(t_ms, sine, color="gray", linewidth=0.8, label="dry", alpha=0.6)
for mix, color in zip([0.3, 0.5, 0.8], COLORS):
    ch = chorus(sine, rate=1.5, depth_ms=3.0, delay_ms=25.0, mix=mix, fs=FS)
    ax.plot(t_ms, ch, linewidth=0.8, color=color, label=f"mix={mix}", alpha=0.85)
ax.set_title("Chorus — waveform (440 Hz sine, rate=1.5 Hz, depth=3 ms, delay=25 ms)",
             fontweight="bold")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude")
ax.legend(fontsize=7)
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.tick_params(labelsize=7)

# ── chorus: frequency response (impulse response)  ──
ax = axes[0, 1]
CHORUS_DELAYS = [15.0, 25.0, 35.0]
for d_ms, color in zip(CHORUS_DELAYS, COLORS):
    h = chorus(impulse, rate=0.0, depth_ms=0.0, delay_ms=d_ms, mix=0.5, fs=FS)
    freqs, db = _freq_response(h)
    mask = (freqs > 10) & (freqs < 2000)
    ax.plot(freqs[mask], db[mask], linewidth=0.8, color=color,
            label=f"delay={d_ms:.0f} ms")
ax.set_title("Chorus — static frequency response (depth=0, mix=0.5)\nDense comb peaks below 100 Hz",
             fontweight="bold")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_xlim(10, 2000)
ax.set_ylim(-20, 5)
ax.legend(fontsize=7)
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.tick_params(labelsize=7)

# ── flanger: waveform ──
ax = axes[1, 0]
ax.plot(t_ms, sine, color="gray", linewidth=0.8, label="dry", alpha=0.6)
for fb, color in zip([0.0, 0.5, 0.8], COLORS):
    fl = flanger(sine, rate=0.5, depth_ms=1.5, delay_ms=2.5, feedback=fb, mix=0.5, fs=FS)
    ax.plot(t_ms, fl, linewidth=0.8, color=color, label=f"fb={fb}", alpha=0.85)
ax.set_title("Flanger — waveform (440 Hz sine, rate=0.5 Hz, delay=2.5 ms)",
             fontweight="bold")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude")
ax.legend(fontsize=7)
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.tick_params(labelsize=7)

# ── flanger: frequency response at static delay (depth=0) — sweeping notches ──
ax = axes[1, 1]
FLANGER_DELAYS = [1.0, 2.5, 5.0]
for d_ms, color in zip(FLANGER_DELAYS, COLORS):
    h = flanger(impulse, rate=0.0, depth_ms=0.0, delay_ms=d_ms, feedback=0.5, mix=0.5, fs=FS)
    freqs, db = _freq_response(h)
    d_s = d_ms * FS / 1000.0
    first_notch = FS / (2 * d_s)
    mask = (freqs > 10) & (freqs < 5000)
    ax.plot(freqs[mask], db[mask], linewidth=0.8, color=color,
            label=f"delay={d_ms} ms  (notch@{first_notch:.0f} Hz)")
ax.set_title("Flanger — static comb response (depth=0, feedback=0.5, mix=0.5)\nNotch at fs/(2·D)",
             fontweight="bold")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_xlim(10, 5000)
ax.set_ylim(-25, 15)
ax.legend(fontsize=7)
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.tick_params(labelsize=7)

fig.suptitle("Modulated Delay Effects — Chorus and Flanger", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
