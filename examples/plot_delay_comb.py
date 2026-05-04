"""Comb filters: FIR feedforward and IIR feedback frequency responses."""

import matplotlib.pyplot as plt
import numpy as np

from python.delay import comb_filter, feedback_delay
from python.generators import generate_impulse

FS = 44100
DURATION = 0.5
DELAY_SAMPLES = 441  # ~10 ms → peaks at multiples of 100 Hz

_, impulse = generate_impulse(fs=FS, duration=DURATION)


def _freq_response(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(np.fft.rfft(h))
    freqs = np.fft.rfftfreq(len(h), d=1.0 / FS)
    db = 20.0 * np.log10(spectrum + 1e-12)
    return freqs, db


GAINS = [0.25, 0.5, 0.75]
COLORS = ["steelblue", "darkorange", "mediumseagreen"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# FIR feedforward comb
ax = axes[0]
for g, color in zip(GAINS, COLORS):
    h = comb_filter(impulse, delay_samples=DELAY_SAMPLES, gain=g)
    freqs, db = _freq_response(h)
    mask = freqs > 0
    ax.plot(freqs[mask], db[mask], label=f"gain={g}", color=color, linewidth=0.9)
ax.set_title(f"FIR Feedforward Comb (D={DELAY_SAMPLES})\nNotches at odd multiples of fs/(2D)", fontweight="bold")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_xlim(0, 2000)
ax.set_ylim(-30, 10)
ax.legend(fontsize=8)
ax.grid(True, linewidth=0.3, alpha=0.5)

# IIR feedback comb
ax = axes[1]
for g, color in zip(GAINS, COLORS):
    h = feedback_delay(impulse, delay_samples=DELAY_SAMPLES, feedback=g)
    freqs, db = _freq_response(h)
    mask = freqs > 0
    ax.plot(freqs[mask], db[mask], label=f"feedback={g}", color=color, linewidth=0.9)
ax.set_title(f"IIR Feedback Comb (D={DELAY_SAMPLES})\nResonant peaks at k·fs/D", fontweight="bold")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_xlim(0, 2000)
ax.set_ylim(-10, 20)
ax.legend(fontsize=8)
ax.grid(True, linewidth=0.3, alpha=0.5)

for ax in axes:
    ax.tick_params(labelsize=7)

fig.suptitle("Comb Filters — Frequency Response", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
