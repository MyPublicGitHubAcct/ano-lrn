"""
ZDF state-variable filter — accuracy and frequency response.

Top-left  : LP/BP/HP/notch responses at cutoff = 1000 Hz, Q = 1/√2 (Butterworth)
Top-right : ZDF vs Chamberlin SVF at fc/fs = 0.4 (17640 Hz)
              ZDF hits −3 dB exactly; Chamberlin drifts due to sin vs tan warp
Bottom-left : LP frequency response — Q sweep (Butterworth through high-Q resonance)
Bottom-right: KVL identity check: LP + k·BP + HP = input at each sample
"""

import matplotlib.pyplot as plt
import numpy as np

from python.filters import svf, zdf_svf
from python.generators import generate_impulse, generate_sine

FS = 44100
DURATION = 4.0      # long impulse for fine frequency resolution
CUTOFF = 1000.0
BUTTERWORTH_Q = 1.0 / np.sqrt(2)

Q_VALUES = [BUTTERWORTH_Q, 1.0, 2.0, 5.0, 10.0]
Q_COLORS = ["steelblue", "mediumseagreen", "darkorange", "mediumpurple", "crimson"]


def _impulse(duration: float = DURATION) -> np.ndarray:
    _, imp = generate_impulse(fs=FS, duration=duration)
    return imp


def _freq_db(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(len(h), d=1.0 / FS)
    db = 20.0 * np.log10(np.abs(np.fft.rfft(h)) + 1e-12)
    return freqs, db


def _plot_h(ax, h, label, color, cutoff=None, lw=1.2):
    freqs, db = _freq_db(h)
    mask = freqs > 0
    ax.semilogx(freqs[mask], db[mask], label=label, color=color, linewidth=lw)
    if cutoff is not None:
        ax.axvline(cutoff, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)


def _style(ax, xlim=(20, FS / 2), ylim=(-60, 15)):
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("Magnitude (dB)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


fig, axes = plt.subplots(2, 2, figsize=(14, 9))
imp = _impulse()

# ── [0,0] LP/BP/HP/notch at Butterworth Q ─────────────────────────────────────

ax = axes[0, 0]
for mode, color, label in [("lp", "steelblue", "LP"), ("bp", "darkorange", "BP"),
                             ("hp", "mediumseagreen", "HP"), ("notch", "crimson", "Notch")]:
    h = zdf_svf(imp, cutoff=CUTOFF, fs=FS, resonance=BUTTERWORTH_Q, mode=mode)
    _plot_h(ax, h, label=label, color=color, cutoff=CUTOFF)
ax.axhline(-3, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
ax.set_title(f"ZDF SVF — LP/BP/HP/Notch\n(cutoff = {int(CUTOFF)} Hz, Q = 1/√2 Butterworth)",
             fontweight="bold")
_style(ax)

# ── [0,1] ZDF vs Chamberlin at fc/fs = 0.4 ────────────────────────────────────

ax = axes[0, 1]
HIGH_FC = 0.4 * FS  # 17640 Hz — where Chamberlin accuracy degrades
lp_zdf = zdf_svf(imp, cutoff=HIGH_FC, fs=FS, resonance=BUTTERWORTH_Q, mode="lp")
lp_cham = svf(imp, cutoff=HIGH_FC, fs=FS, resonance=0.0, mode="lp")

_plot_h(ax, lp_zdf, label=f"ZDF LP (fc = {int(HIGH_FC)} Hz)", color="steelblue", cutoff=HIGH_FC)
_plot_h(ax, lp_cham, label=f"Chamberlin LP (fc = {int(HIGH_FC)} Hz)", color="crimson",
        cutoff=HIGH_FC, lw=1.0)
ax.axhline(-3, color="black", linewidth=0.8, linestyle=":", alpha=0.7, label="−3 dB")
ax.set_title("ZDF vs Chamberlin SVF at fc/fs = 0.4\n(ZDF hits −3 dB exactly; Chamberlin drifts)",
             fontweight="bold")
_style(ax, xlim=(100, FS / 2), ylim=(-60, 15))

# ── [1,0] LP Q sweep ──────────────────────────────────────────────────────────

ax = axes[1, 0]
for Q, color in zip(Q_VALUES, Q_COLORS):
    h = zdf_svf(imp, cutoff=CUTOFF, fs=FS, resonance=Q, mode="lp")
    _plot_h(ax, h, label=f"Q = {Q:.2g}", color=color, cutoff=CUTOFF)
ax.axhline(-3, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
ax.set_title(f"LP frequency response — Q sweep\n(cutoff = {int(CUTOFF)} Hz)", fontweight="bold")
_style(ax, ylim=(-60, 30))

# ── [1,1] KVL identity: LP + k·BP + HP = x ───────────────────────────────────

ax = axes[1, 1]
_, sig = generate_sine(freq=330.0, fs=FS, duration=0.005)
t_ms = np.arange(len(sig)) / FS * 1000

Q = 2.0
k = 1.0 / Q
lp, bp, hp = zdf_svf(sig, cutoff=CUTOFF, fs=FS, resonance=Q, mode="all")
reconstructed = lp + k * bp + hp
error = reconstructed - sig

ax.plot(t_ms, sig, label="input x", color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
ax.plot(t_ms, lp, label="LP", color="steelblue", linewidth=0.9)
ax.plot(t_ms, k * bp, label=f"k·BP (k={k:.2f})", color="darkorange", linewidth=0.9)
ax.plot(t_ms, hp, label="HP", color="mediumseagreen", linewidth=0.9)
ax.plot(t_ms, error * 1e11, label=f"error ×10¹¹ (max {np.max(np.abs(error)):.1e})",
        color="crimson", linewidth=0.7, linestyle=":")
ax.set_title(f"KVL identity: LP + k·BP + HP = x\n(cutoff = {int(CUTOFF)} Hz, Q = {Q})",
             fontweight="bold")
ax.set_xlabel("Time (ms)", fontsize=8)
ax.set_ylabel("Amplitude", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.legend(fontsize=6.5)

fig.suptitle("Zero-Delay-Feedback State-Variable Filter (ZDF SVF)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
