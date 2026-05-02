"""
Two-row grid of filter plots.

Top row — frequency response (magnitude in dB, log-frequency x-axis):
  Low-pass : three cutoff frequencies overlaid at fixed Q = 0.707
  High-pass: three cutoff frequencies overlaid at fixed Q = 0.707
  Band-pass: three Q values overlaid at fixed cutoff = 1000 Hz

Bottom row — impulse response (time domain, first 30 ms):
  Low-pass, High-pass, Band-pass at representative settings

Impulse responses are derived from generate_impulse(), demonstrating how
the test-signal generators and filters work together.
"""

import matplotlib.pyplot as plt
import numpy as np

from ano_lrn.filters import bandpass, highpass, lowpass
from ano_lrn.generators import generate_impulse

FS = 44100
DURATION = 1.0  # 1 s → 0.5 Hz frequency resolution from FFT
WINDOW_MS = 30  # impulse-response display window in milliseconds

CUTOFFS = [500.0, 1000.0, 4000.0]
BP_QS = [0.5, 1.0, 4.0]
COLORS = ["steelblue", "darkorange", "mediumseagreen"]


def _impulse() -> np.ndarray:
    _, imp = generate_impulse(fs=FS, duration=DURATION)
    return imp


def _freq_response(h: np.ndarray):
    """Return (freqs_hz, magnitude_db) for a filter impulse response."""
    spectrum = np.abs(np.fft.rfft(h))
    freqs = np.fft.rfftfreq(len(h), d=1.0 / FS)
    db = 20.0 * np.log10(spectrum + 1e-12)
    return freqs, db


def _plot_freq(ax, freqs, db, label, color, cutoff=None):
    mask = freqs > 0  # skip DC for log scale
    ax.semilogx(freqs[mask], db[mask], label=label, color=color, linewidth=1.2)
    if cutoff is not None:
        ax.axvline(cutoff, color=color, linewidth=0.6, linestyle="--", alpha=0.5)


def _plot_impulse_response(ax, h, label, color):
    n_samples = int(WINDOW_MS * FS / 1000)
    t_ms = np.arange(n_samples) / FS * 1000
    ax.plot(t_ms, h[:n_samples], color=color, linewidth=0.9, label=label)


fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# ── Row 0: frequency responses ────────────────────────────────────────────────

imp = _impulse()

# Low-pass: vary cutoff
ax = axes[0, 0]
for cutoff, color in zip(CUTOFFS, COLORS):
    h = lowpass(imp, cutoff=cutoff, fs=FS, Q=0.707)
    freqs, db = _freq_response(h)
    _plot_freq(ax, freqs, db, label=f"{int(cutoff)} Hz", color=color, cutoff=cutoff)
ax.set_title("Low-pass (Q = 0.707)", fontweight="bold")
ax.set_ylim(-70, 5)

# High-pass: vary cutoff
ax = axes[0, 1]
for cutoff, color in zip(CUTOFFS, COLORS):
    h = highpass(imp, cutoff=cutoff, fs=FS, Q=0.707)
    freqs, db = _freq_response(h)
    _plot_freq(ax, freqs, db, label=f"{int(cutoff)} Hz", color=color, cutoff=cutoff)
ax.set_title("High-pass (Q = 0.707)", fontweight="bold")
ax.set_ylim(-70, 5)

# Band-pass: vary Q
ax = axes[0, 2]
for Q, color in zip(BP_QS, COLORS):
    h = bandpass(imp, cutoff=1000.0, fs=FS, Q=Q)
    freqs, db = _freq_response(h)
    _plot_freq(ax, freqs, db, label=f"Q = {Q}", color=color, cutoff=1000.0)
ax.set_title("Band-pass (cutoff = 1000 Hz)", fontweight="bold")
ax.set_ylim(-70, 5)

for ax in axes[0]:
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("Magnitude (dB)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax.axhline(-3, color="gray", linewidth=0.5, linestyle=":", alpha=0.7)
    ax.legend(fontsize=7)
    ax.set_xlim(20, FS / 2)

# ── Row 1: impulse responses ──────────────────────────────────────────────────

IMPULSE_SPECS = [
    ("Low-pass\ncutoff = 1000 Hz, Q = 0.707",  lowpass,   dict(cutoff=1000.0, Q=0.707)),
    ("High-pass\ncutoff = 1000 Hz, Q = 0.707", highpass,  dict(cutoff=1000.0, Q=0.707)),
    ("Band-pass\ncutoff = 1000 Hz, Q = 4.0",   bandpass,  dict(cutoff=1000.0, Q=4.0)),
]

for (title, fn, kwargs), ax, color in zip(IMPULSE_SPECS, axes[1], COLORS):
    h = fn(imp, fs=FS, **kwargs)
    _plot_impulse_response(ax, h, label=title.split("\n")[0], color=color)
    ax.set_title(title, fontweight="bold", fontsize=9)
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)

fig.suptitle("Biquad Filter Responses", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
