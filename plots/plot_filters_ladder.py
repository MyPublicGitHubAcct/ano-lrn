"""
ZDF Moog ladder filter — frequency and time-domain responses.

Top row — frequency response (magnitude in dB, log-frequency x-axis):
  Left  : five resonance values at cutoff = 1000 Hz
            shows the resonant peak growing and DC gain dropping with k
  Right : three cutoff frequencies at resonance = 0.7
            shows the 24 dB/oct slope tracking the cutoff

Bottom row — time domain:
  Left  : impulse response at five resonance values (first 80 ms)
            shows ringing increasing and decay slowing as resonance → 1
  Right : step response at five resonance values (first 40 ms)
            shows the characteristic Moog "boing"; steady-state level is
            1/(1 + k) so higher resonance settles to a lower DC value
"""

import matplotlib.pyplot as plt
import numpy as np

from python.filters import moog_ladder
from python.generators import generate_impulse, generate_step

FS = 44100
DURATION = 1.0
CUTOFF = 1000.0

RESONANCES = [0.0, 0.25, 0.5, 0.75, 0.95]
RES_LABELS = [f"res = {r}" for r in RESONANCES]
RES_COLORS = ["steelblue", "mediumseagreen", "darkorange", "mediumpurple", "crimson"]

CUTOFFS = [250.0, 1000.0, 4000.0]
CUTOFF_COLORS = ["steelblue", "darkorange", "mediumseagreen"]


def _impulse() -> np.ndarray:
    _, imp = generate_impulse(fs=FS, duration=DURATION)
    return imp


def _step() -> np.ndarray:
    _, st = generate_step(fs=FS, duration=DURATION)
    return st


def _freq_response(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(np.fft.rfft(h))
    freqs = np.fft.rfftfreq(len(h), d=1.0 / FS)
    db = 20.0 * np.log10(spectrum + 1e-12)
    return freqs, db


def _plot_freq(ax, freqs, db, label, color, cutoff=None):
    mask = freqs > 0
    ax.semilogx(freqs[mask], db[mask], label=label, color=color, linewidth=1.2)
    if cutoff is not None:
        ax.axvline(cutoff, color=color, linewidth=0.6, linestyle="--", alpha=0.4)


fig, axes = plt.subplots(2, 2, figsize=(14, 9))

imp = _impulse()
step = _step()

# ── [0,0] Frequency response: resonance sweep ─────────────────────────────────

ax = axes[0, 0]
for res, label, color in zip(RESONANCES, RES_LABELS, RES_COLORS):
    h = moog_ladder(imp, cutoff=CUTOFF, fs=FS, resonance=res)
    _plot_freq(ax, *_freq_response(h), label=label, color=color)
ax.axvline(CUTOFF, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
ax.axhline(-3, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
ax.set_title(f"Frequency response — resonance sweep\n(cutoff = {int(CUTOFF)} Hz)", fontweight="bold")
ax.set_ylim(-80, 35)

# ── [0,1] Frequency response: cutoff sweep ────────────────────────────────────

ax = axes[0, 1]
for fc, color in zip(CUTOFFS, CUTOFF_COLORS):
    h = moog_ladder(imp, cutoff=fc, fs=FS, resonance=0.7)
    _plot_freq(ax, *_freq_response(h), label=f"{int(fc)} Hz", color=color, cutoff=fc)
ax.axhline(-3, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
ax.set_title("Frequency response — cutoff sweep\n(resonance = 0.7)", fontweight="bold")
ax.set_ylim(-80, 15)

for ax in axes[0]:
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("Magnitude (dB)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=7)
    ax.set_xlim(20, FS / 2)

# ── [1,0] Impulse response: resonance sweep ───────────────────────────────────

ax = axes[1, 0]
window_ms = 80
n = int(window_ms * FS / 1000)
t_ms = np.arange(n) / FS * 1000

for res, label, color in zip(RESONANCES, RES_LABELS, RES_COLORS):
    h = moog_ladder(imp, cutoff=CUTOFF, fs=FS, resonance=res)
    ax.plot(t_ms, h[:n], label=label, color=color, linewidth=0.9)

ax.set_title(f"Impulse response — resonance sweep\n(cutoff = {int(CUTOFF)} Hz)", fontweight="bold")
ax.set_xlabel("Time (ms)", fontsize=8)
ax.set_ylabel("Amplitude", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
ax.legend(fontsize=7)

# ── [1,1] Step response: resonance sweep ──────────────────────────────────────

ax = axes[1, 1]
window_ms = 40
n = int(window_ms * FS / 1000)
t_ms = np.arange(n) / FS * 1000

for res, label, color in zip(RESONANCES, RES_LABELS, RES_COLORS):
    h = moog_ladder(step, cutoff=CUTOFF, fs=FS, resonance=res)
    k = res * 4.0
    dc = 1.0 / (1.0 + k)
    ax.plot(t_ms, h[:n], label=f"{label}  (DC→{dc:.2f})", color=color, linewidth=0.9)

ax.axhline(1.0, color="gray", linewidth=0.7, linestyle=":", alpha=0.6, label="input")
ax.set_title(f"Step response — resonance sweep\n(cutoff = {int(CUTOFF)} Hz)", fontweight="bold")
ax.set_xlabel("Time (ms)", fontsize=8)
ax.set_ylabel("Amplitude", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.legend(fontsize=7)

fig.suptitle("ZDF Moog Ladder Filter", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
