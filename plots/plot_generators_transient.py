"""Transient generators: impulse and step."""

import matplotlib.pyplot as plt

from python.generators import generate_impulse, generate_step


def _ms(ms: float) -> float:
    return ms / 1000.0


FS = 44100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

t, wave = generate_impulse(fs=FS, duration=_ms(5), delay=0.0)
axes[0].plot(t * 1000, wave, linewidth=0.8, color="steelblue")
axes[0].set_title("Impulse (delay = 0)", fontsize=10, fontweight="bold")
axes[0].set_xlabel("Time (ms)", fontsize=8)
axes[0].set_ylabel("Amplitude", fontsize=8)
axes[0].set_ylim(-0.2, 1.3)

t, wave = generate_step(fs=FS, duration=_ms(10), onset=_ms(2))
axes[1].plot(t * 1000, wave, linewidth=0.8, color="darkorange")
axes[1].set_title("Step (onset = 2 ms)", fontsize=10, fontweight="bold")
axes[1].set_xlabel("Time (ms)", fontsize=8)
axes[1].set_ylabel("Amplitude", fontsize=8)

for ax in axes:
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)

fig.suptitle("Transient Generators", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
