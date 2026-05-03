"""
EQ / parametric filter frequency responses and impulse responses.

Top row — frequency response (magnitude in dB, log-frequency x-axis):
  Low-pass  : three cutoff frequencies at Q = 0.707
  High-pass : three cutoff frequencies at Q = 0.707
  Band-pass : three Q values at cutoff = 1000 Hz
  Notch     : three Q values at cutoff = 1000 Hz
  All-pass  : three Q values at cutoff = 1000 Hz (magnitude flat at 0 dB)

Bottom row — impulse response (time domain, first 30 ms):
  Low-pass, High-pass, Band-pass, Notch, All-pass at representative settings
"""

import matplotlib.pyplot as plt
import numpy as np

from python.filters import allpass, bandpass, highpass, lowpass, notch
from python.generators import generate_impulse

FS = 44100
DURATION = 1.0
WINDOW_MS = 30

CUTOFFS = [500.0, 1000.0, 4000.0]
QS = [0.5, 1.0, 4.0]
AP_QS = [0.5, 0.707, 2.0]
COLORS = ["steelblue", "darkorange", "mediumseagreen"]


def _impulse() -> np.ndarray:
    _, imp = generate_impulse(fs=FS, duration=DURATION)
    return imp


def _freq_response(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(np.fft.rfft(h))
    freqs = np.fft.rfftfreq(len(h), d=1.0 / FS)
    db = 20.0 * np.log10(spectrum + 1e-12)
    return freqs, db


def _plot_freq(ax, freqs, db, label, color, cutoff=None):
    mask = freqs > 0
    ax.semilogx(freqs[mask], db[mask], label=label, color=color, linewidth=1.2)
    if cutoff is not None:
        ax.axvline(cutoff, color=color, linewidth=0.6, linestyle="--", alpha=0.5)


def _plot_ir(ax, h, label, color):
    n = int(WINDOW_MS * FS / 1000)
    t_ms = np.arange(n) / FS * 1000
    ax.plot(t_ms, h[:n], color=color, linewidth=0.9, label=label)


fig, axes = plt.subplots(2, 5, figsize=(20, 8))

imp = _impulse()

# ── Row 0: frequency responses ────────────────────────────────────────────────

ax = axes[0, 0]
for cutoff, color in zip(CUTOFFS, COLORS):
    h = lowpass(imp, cutoff=cutoff, fs=FS, Q=0.707)
    _plot_freq(ax, *_freq_response(h), label=f"{int(cutoff)} Hz", color=color, cutoff=cutoff)
ax.set_title("Low-pass (Q = 0.707)", fontweight="bold")
ax.set_ylim(-70, 5)

ax = axes[0, 1]
for cutoff, color in zip(CUTOFFS, COLORS):
    h = highpass(imp, cutoff=cutoff, fs=FS, Q=0.707)
    _plot_freq(ax, *_freq_response(h), label=f"{int(cutoff)} Hz", color=color, cutoff=cutoff)
ax.set_title("High-pass (Q = 0.707)", fontweight="bold")
ax.set_ylim(-70, 5)

ax = axes[0, 2]
for Q, color in zip(QS, COLORS):
    h = bandpass(imp, cutoff=1000.0, fs=FS, Q=Q)
    _plot_freq(ax, *_freq_response(h), label=f"Q = {Q}", color=color, cutoff=1000.0)
ax.set_title("Band-pass (cutoff = 1000 Hz)", fontweight="bold")
ax.set_ylim(-70, 5)

ax = axes[0, 3]
for Q, color in zip(QS, COLORS):
    h = notch(imp, cutoff=1000.0, fs=FS, Q=Q)
    _plot_freq(ax, *_freq_response(h), label=f"Q = {Q}", color=color, cutoff=1000.0)
ax.set_title("Notch (cutoff = 1000 Hz)", fontweight="bold")
ax.set_ylim(-70, 5)

ax = axes[0, 4]
for Q, color in zip(AP_QS, COLORS):
    h = allpass(imp, cutoff=1000.0, fs=FS, Q=Q)
    _plot_freq(ax, *_freq_response(h), label=f"Q = {Q}", color=color, cutoff=1000.0)
ax.set_title("All-pass (cutoff = 1000 Hz)\n|H| = 0 dB at all frequencies", fontweight="bold")
ax.set_ylim(-2, 2)

for ax in axes[0]:
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("Magnitude (dB)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax.axhline(-3, color="gray", linewidth=0.5, linestyle=":", alpha=0.7)
    ax.legend(fontsize=7)
    ax.set_xlim(20, FS / 2)

# ── Row 1: impulse responses ──────────────────────────────────────────────────

IR_SPECS = [
    ("Low-pass\n1000 Hz, Q = 0.707",  lowpass,  dict(cutoff=1000.0, Q=0.707)),
    ("High-pass\n1000 Hz, Q = 0.707", highpass, dict(cutoff=1000.0, Q=0.707)),
    ("Band-pass\n1000 Hz, Q = 4.0",   bandpass, dict(cutoff=1000.0, Q=4.0)),
    ("Notch\n1000 Hz, Q = 4.0",       notch,    dict(cutoff=1000.0, Q=4.0)),
    ("All-pass\n1000 Hz, Q = 0.707",  allpass,  dict(cutoff=1000.0, Q=0.707)),
]

for (title, fn, kwargs), ax, color in zip(IR_SPECS, axes[1], COLORS + ["mediumpurple", "crimson"]):
    h = fn(imp, fs=FS, **kwargs)
    _plot_ir(ax, h, label=title.split("\n")[0], color=color)
    ax.set_title(title, fontweight="bold", fontsize=9)
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)

fig.suptitle("EQ / Parametric Filter Responses", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
