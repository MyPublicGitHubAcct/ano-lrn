"""Periodic waveform generators: sine, square, sawtooth, triangle, multi-tone."""

import matplotlib.pyplot as plt

from python.generators import (
    generate_multi_tone,
    generate_sawtooth,
    generate_sine,
    generate_square,
    generate_triangle,
)

FS = 44100
FREQ = 440.0


def _cycles(freq: float, n: int = 2) -> float:
    return n / freq


PLOTS = [
    {
        "title": "Sine",
        "fn": lambda: generate_sine(freq=FREQ, fs=FS, duration=_cycles(FREQ)),
    },
    {
        "title": "Square (duty=0.5)",
        "fn": lambda: generate_square(freq=FREQ, fs=FS, duration=_cycles(FREQ)),
    },
    {
        "title": "Sawtooth",
        "fn": lambda: generate_sawtooth(freq=FREQ, fs=FS, duration=_cycles(FREQ)),
    },
    {
        "title": "Triangle",
        "fn": lambda: generate_triangle(freq=FREQ, fs=FS, duration=_cycles(FREQ)),
    },
    {
        "title": "Multi-tone (440 + 880 + 1320 Hz)",
        "fn": lambda: generate_multi_tone(freqs=[440.0, 880.0, 1320.0], fs=FS, duration=_cycles(FREQ)),
    },
]

COLS = 3
ROWS = -(-len(PLOTS) // COLS)

fig, axes = plt.subplots(ROWS, COLS, figsize=(14, ROWS * 2.8))
axes = axes.flatten()

for ax, spec in zip(axes, PLOTS):
    t, wave = spec["fn"]()
    ax.plot(t * 1000, wave, linewidth=0.8, color="steelblue")
    ax.set_title(spec["title"], fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)

for ax in axes[len(PLOTS):]:
    ax.set_visible(False)

fig.suptitle("Periodic Waveform Generators", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
