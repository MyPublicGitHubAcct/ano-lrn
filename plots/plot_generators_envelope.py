"""ADSR envelope generator: shapes and gated sine application."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_adsr, generate_sine

FS = 44100

# --- four ADSR shapes ---

SHAPES = [
    dict(attack=0.01, decay=0.05, sustain=0.8, release=0.1,  label="Fast attack / high sustain"),
    dict(attack=0.15, decay=0.10, sustain=0.5, release=0.25, label="Slow attack / mid sustain"),
    dict(attack=0.05, decay=0.30, sustain=0.2, release=0.05, label="Long decay / low sustain"),
    dict(attack=0.10, decay=0.05, sustain=0.0, release=0.20, label="Pluck (sustain = 0)"),
]

DURATION = 0.6

fig, axes = plt.subplots(len(SHAPES), 1, figsize=(12, 8), sharex=True)
colors = ["steelblue", "darkorange", "mediumseagreen", "crimson"]

for ax, shape, color in zip(axes, SHAPES, colors):
    t, env = generate_adsr(fs=FS, duration=DURATION, **{k: v for k, v in shape.items() if k != "label"})
    ax.plot(t * 1000, env, color=color, linewidth=1.2)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.set_title(shape["label"], fontsize=9, fontweight="bold")
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)

axes[-1].set_xlabel("Time (ms)", fontsize=9)
fig.suptitle("ADSR Envelope Shapes", fontsize=13, fontweight="bold")
plt.tight_layout()

# --- linear vs exponential comparison ---

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

kw = dict(attack=0.05, decay=0.15, sustain=0.5, release=0.2, fs=FS, duration=0.55)
t_lin, env_lin = generate_adsr(**kw, curve="linear")
t_exp, env_exp = generate_adsr(**kw, curve="exponential")

axes2[0].plot(t_lin * 1000, env_lin, color="steelblue", linewidth=1.2, label="linear")
axes2[0].plot(t_exp * 1000, env_exp, color="darkorange", linewidth=1.2, linestyle="--", label="exponential")
axes2[0].set_title("Linear vs Exponential Curve", fontsize=10, fontweight="bold")
axes2[0].set_xlabel("Time (ms)", fontsize=9)
axes2[0].set_ylabel("Amplitude", fontsize=9)
axes2[0].legend(fontsize=8)
axes2[0].grid(True, linewidth=0.3, alpha=0.5)
axes2[0].set_ylim(-0.05, 1.1)

# --- gated sine ---

FREQ = 440.0
GATE_DURATION = 0.5
t_sine, sine = generate_sine(freq=FREQ, fs=FS, duration=GATE_DURATION)
_, gate = generate_adsr(attack=0.02, decay=0.05, sustain=0.75, release=0.15,
                        fs=FS, duration=GATE_DURATION, curve="linear")
shaped = sine * gate

axes2[1].plot(t_sine * 1000, sine, color="lightgray", linewidth=0.6, label="Dry sine")
axes2[1].plot(t_sine * 1000, shaped, color="steelblue", linewidth=0.9, label="Gated sine")
axes2[1].plot(t_sine * 1000, gate, color="crimson", linewidth=1.0, linestyle="--",
              alpha=0.8, label="ADSR envelope")
axes2[1].set_title("Gated Sine (440 Hz)", fontsize=10, fontweight="bold")
axes2[1].set_xlabel("Time (ms)", fontsize=9)
axes2[1].set_ylabel("Amplitude", fontsize=9)
axes2[1].legend(fontsize=8)
axes2[1].grid(True, linewidth=0.3, alpha=0.5)

fig2.suptitle("ADSR Envelope — Curve Comparison and Gated Sine", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()
