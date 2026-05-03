"""Plot one cycle (or a representative window) of every test signal generator."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import (
    generate_chirp,
    generate_dc,
    generate_impulse,
    generate_pink_noise,
    generate_sawtooth,
    generate_sine,
    generate_square,
    generate_step,
    generate_triangle,
    generate_white_noise,
)

FS = 44100
FREQ = 440.0
# Two full cycles at 440 Hz
CYCLE_DURATION = 2 / FREQ
# Longer window for signals that need context
WINDOW_DURATION = 0.05


def _cycles(freq: float, n: int = 2) -> float:
    return n / freq


def _ms(ms: float) -> float:
    return ms / 1000.0


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
        "title": "White Noise",
        "fn": lambda: generate_white_noise(fs=FS, duration=_ms(10), seed=42),
    },
    {
        "title": "Pink Noise",
        "fn": lambda: generate_pink_noise(fs=FS, duration=_ms(10), seed=42),
    },
    {
        "title": "Impulse (delay=0)",
        "fn": lambda: generate_impulse(fs=FS, duration=_ms(5), delay=0.0),
    },
    {
        "title": "Step (onset=2 ms)",
        "fn": lambda: generate_step(fs=FS, duration=_ms(10), onset=_ms(2)),
    },
    {
        "title": "Chirp – Logarithmic",
        "fn": lambda: generate_chirp(f_start=20, f_end=20000, fs=FS, duration=1.0, method="logarithmic"),
    },
    {
        "title": "Chirp – Linear",
        "fn": lambda: generate_chirp(f_start=20, f_end=20000, fs=FS, duration=1.0, method="linear"),
    },
    {
        "title": "DC (amplitude=1)",
        "fn": lambda: generate_dc(fs=FS, duration=_ms(10), amplitude=1.0),
    },
]

COLS = 3
ROWS = -(-len(PLOTS) // COLS)  # ceiling division

fig, axes = plt.subplots(ROWS, COLS, figsize=(14, ROWS * 2.8))
axes = axes.flatten()

for ax, spec in zip(axes, PLOTS):
    t, wave = spec["fn"]()
    # Convert time axis to milliseconds for readability
    ax.plot(t * 1000, wave, linewidth=0.8, color="steelblue")
    ax.set_title(spec["title"], fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    # Add a faint zero line for reference
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)

# Hide any unused axes
for ax in axes[len(PLOTS):]:
    ax.set_visible(False)

fig.suptitle("Test Signal Generators", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()
