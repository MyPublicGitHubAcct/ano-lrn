"""Reference generators: DC, Nyquist (fs/2), half-Nyquist (fs/4), quarter-Nyquist (fs/8)."""

import matplotlib.pyplot as plt

from python.generators import (
    generate_dc,
    generate_half_nyquist,
    generate_nyquist,
    generate_quarter_nyquist,
)


def _ms(ms: float) -> float:
    return ms / 1000.0


FS = 44100

PLOTS = [
    {
        "title": "DC (amplitude = 1)",
        "fn": lambda: generate_dc(fs=FS, duration=_ms(5), amplitude=1.0),
        "color": "steelblue",
    },
    {
        "title": "Nyquist (fs/2) — 4 samples shown",
        "fn": lambda: generate_nyquist(fs=FS, duration=_ms(0.1)),
        "color": "darkorange",
    },
    {
        "title": "Half-Nyquist (fs/4) — 8 samples shown",
        "fn": lambda: generate_half_nyquist(fs=FS, duration=_ms(0.2)),
        "color": "mediumseagreen",
    },
    {
        "title": "Quarter-Nyquist (fs/8) — 8 samples shown",
        "fn": lambda: generate_quarter_nyquist(fs=FS, duration=_ms(0.2)),
        "color": "mediumpurple",
    },
]

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

for ax, spec in zip(axes, PLOTS):
    t, wave = spec["fn"]()
    ax.plot(t * 1000, wave, linewidth=0.8, color=spec["color"])
    # Show individual samples as dots for the Nyquist-family signals
    if len(wave) <= 12:
        ax.scatter(t * 1000, wave, s=20, color=spec["color"], zorder=5)
    ax.set_title(spec["title"], fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)

fig.suptitle("Reference Generators", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
