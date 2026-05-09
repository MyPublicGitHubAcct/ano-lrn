"""Formant filter: cascade of resonators for vowel synthesis."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_impulse
from python.source_filter import formant_filter

FS = 44100
DURATION = 0.5

_, impulse = generate_impulse(fs=FS, duration=DURATION)

VOWELS = {
    "/a/": {"freqs": [800.0, 1200.0, 2500.0], "bws": [100.0, 100.0, 120.0]},
    "/i/": {"freqs": [300.0, 2200.0, 3000.0], "bws": [80.0, 90.0, 120.0]},
    "/u/": {"freqs": [300.0, 800.0, 2300.0], "bws": [80.0, 100.0, 120.0]},
}
COLORS = ["steelblue", "darkorange", "mediumseagreen"]

freqs_axis = np.fft.rfftfreq(len(impulse), d=1.0 / FS)


def _db(s: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.abs(np.fft.rfft(s)) + 1e-12)


fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, (vowel, params), color in zip(axes, VOWELS.items(), COLORS):
    h = formant_filter(impulse, formant_freqs=params["freqs"], bandwidths=params["bws"], fs=FS)
    db = _db(h)
    mask = freqs_axis > 0
    ax.plot(freqs_axis[mask], db[mask], linewidth=1.0, color=color)
    for f in params["freqs"]:
        ax.axvline(f, color=color, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_title(f"Vowel {vowel}\nF1={params['freqs'][0]:.0f}, F2={params['freqs'][1]:.0f}, F3={params['freqs'][2]:.0f} Hz",
                 fontweight="bold", fontsize=9)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(100, 5000)
    ax.set_ylim(-40, 10)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)

fig.suptitle("Formant Filter — Vowel Synthesis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
