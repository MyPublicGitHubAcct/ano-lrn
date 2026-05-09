"""Waveshaping and bitcrush: polynomial distortion and quantisation."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_sine
from python.nonlinear import bitcrush, waveshape

FS = 44100
FREQ = 440.0
DURATION = 3 / FREQ

_, sig = generate_sine(freq=FREQ, fs=FS, duration=DURATION)
t_ms = np.arange(len(sig)) / FS * 1000

COLORS = ["steelblue", "darkorange", "mediumseagreen", "crimson"]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))

# waveshape: transfer curves and waveforms
x_tc = np.linspace(-1, 1, 400)
SHAPES = [
    ("Identity", [0, 1]),
    ("3rd harmonic (T3)", [-3, 0, 4]),
    ("Odd saturation", [0, 1, 0, -0.3]),
    ("Chebyshev T5", [0, 5, 0, -20, 0, 16]),
]

ax_tc = axes[0, 0]
ax_tc.set_title("waveshape() — transfer curves", fontweight="bold")
for (label, coeffs), color in zip(SHAPES, COLORS):
    ax_tc.plot(x_tc, np.polyval(list(reversed(coeffs)) if len(coeffs) > 1 else coeffs, x_tc), color=color, linewidth=0.9, label=label)
    # use np.polynomial.polynomial.polyval which uses ascending order
    ax_tc.clear()

# Redo with correct polyval
ax_tc = axes[0, 0]
ax_tc.set_title("waveshape() — transfer curves", fontweight="bold")
for (label, coeffs), color in zip(SHAPES, COLORS):
    y_tc = np.polynomial.polynomial.polyval(x_tc, coeffs)
    ax_tc.plot(x_tc, y_tc, color=color, linewidth=0.9, label=label)
ax_tc.legend(fontsize=6)
ax_tc.set_xlabel("Input")
ax_tc.set_ylabel("Output")

for col, ((label, coeffs), color) in enumerate(zip(SHAPES[1:], COLORS[1:]), start=1):
    out = waveshape(sig, coeffs)
    axes[0, col].plot(t_ms, sig, color="gray", linewidth=0.5, alpha=0.4, label="Input")
    axes[0, col].plot(t_ms, out, color=color, linewidth=0.9, label=label)
    axes[0, col].set_title(f"waveshape: {label}", fontsize=9, fontweight="bold")
    axes[0, col].legend(fontsize=6)
    axes[0, col].set_xlabel("Time (ms)", fontsize=7)

# bitcrush
BITS = [16, 8, 4, 2]
for col, (bits, color) in enumerate(zip(BITS, COLORS)):
    out = bitcrush(sig, bits=bits)
    axes[1, col].plot(t_ms, out, color=color, linewidth=0.9, label=f"{bits} bits ({2**bits} levels)")
    axes[1, col].set_title(f"bitcrush — {bits} bits", fontsize=9, fontweight="bold")
    axes[1, col].legend(fontsize=6)
    axes[1, col].set_xlabel("Time (ms)", fontsize=7)
    axes[1, col].set_ylabel("Amplitude", fontsize=7)

for ax in axes.flatten():
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.5)

fig.suptitle("Waveshaping and Bitcrush", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
