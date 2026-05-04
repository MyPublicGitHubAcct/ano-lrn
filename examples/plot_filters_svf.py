"""
Chamberlin state-variable filter — frequency and time-domain responses.

Top-left  : LP/BP/HP/notch frequency response at resonance = 0.5
              shows all four output ports from a single filter pass
Top-right : LP frequency response at five resonance values
              shows the resonant peak growing with resonance
Bottom-left : Impulse response — LP, BP, HP at resonance = 0.5
Bottom-right: Per-sample cutoff modulation (sine LFO sweeping 200–2000 Hz)
              spectrogram of LP output — demonstrates the key SVF advantage
"""

import matplotlib.pyplot as plt
import numpy as np

from python.filters import svf
from python.generators import generate_impulse, generate_white_noise

FS = 44100
DURATION = 1.0
CUTOFF = 1000.0

RESONANCES = [0.0, 0.3, 0.5, 0.7, 0.9]
RES_LABELS = [f"res = {r}" for r in RESONANCES]
RES_COLORS = ["steelblue", "mediumseagreen", "darkorange", "mediumpurple", "crimson"]


def _impulse() -> np.ndarray:
    _, imp = generate_impulse(fs=FS, duration=DURATION)
    return imp


def _freq_response(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(np.fft.rfft(h))
    freqs = np.fft.rfftfreq(len(h), d=1.0 / FS)
    db = 20.0 * np.log10(spectrum + 1e-12)
    return freqs, db


def _plot_freq(ax, h, label, color, cutoff=None):
    freqs, db = _freq_response(h)
    mask = freqs > 0
    ax.semilogx(freqs[mask], db[mask], label=label, color=color, linewidth=1.2)
    if cutoff is not None:
        ax.axvline(cutoff, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)


def _style_freq_ax(ax):
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("Magnitude (dB)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=7)
    ax.set_xlim(20, FS / 2)


fig, axes = plt.subplots(2, 2, figsize=(14, 9))

imp = _impulse()

# ── [0,0] All four output modes at resonance = 0.5 ────────────────────────────

ax = axes[0, 0]
modes = [("lp", "steelblue", "LP"), ("bp", "darkorange", "BP"),
         ("hp", "mediumseagreen", "HP"), ("notch", "crimson", "Notch")]
for mode, color, label in modes:
    h = svf(imp, cutoff=CUTOFF, fs=FS, resonance=0.5, mode=mode)
    _plot_freq(ax, h, label=label, color=color, cutoff=CUTOFF)
ax.axhline(-3, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
ax.set_title(f"LP / BP / HP / Notch — all four modes\n(cutoff = {int(CUTOFF)} Hz, resonance = 0.5)",
             fontweight="bold")
ax.set_ylim(-60, 25)
_style_freq_ax(ax)

# ── [0,1] LP frequency response: resonance sweep ──────────────────────────────

ax = axes[0, 1]
for res, label, color in zip(RESONANCES, RES_LABELS, RES_COLORS):
    h = svf(imp, cutoff=CUTOFF, fs=FS, resonance=res, mode="lp")
    _plot_freq(ax, h, label=label, color=color, cutoff=CUTOFF)
ax.axhline(-3, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
ax.set_title(f"LP frequency response — resonance sweep\n(cutoff = {int(CUTOFF)} Hz)",
             fontweight="bold")
ax.set_ylim(-60, 25)
_style_freq_ax(ax)

# ── [1,0] Impulse response — LP, BP, HP ───────────────────────────────────────

ax = axes[1, 0]
window_ms = 20
n_win = int(window_ms * FS / 1000)
t_ms = np.arange(n_win) / FS * 1000

lp, bp, hp = svf(imp, cutoff=CUTOFF, fs=FS, resonance=0.5, mode="all")
ax.plot(t_ms, lp[:n_win], label="LP", color="steelblue", linewidth=0.9)
ax.plot(t_ms, bp[:n_win], label="BP", color="darkorange", linewidth=0.9)
ax.plot(t_ms, hp[:n_win], label="HP", color="mediumseagreen", linewidth=0.9)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
ax.set_title(f"Impulse responses — LP, BP, HP\n(cutoff = {int(CUTOFF)} Hz, resonance = 0.5)",
             fontweight="bold")
ax.set_xlabel("Time (ms)", fontsize=8)
ax.set_ylabel("Amplitude", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.legend(fontsize=7)

# ── [1,1] Per-sample cutoff modulation via manual loop ────────────────────────

ax = axes[1, 1]
rng = np.random.default_rng(42)
_, noise = generate_white_noise(fs=FS, duration=DURATION)

# Modulate cutoff with a 2 Hz LFO sweeping 200–2000 Hz, sample-by-sample
lfo_rate = 2.0
t = np.arange(len(noise)) / FS
cutoff_mod = 200.0 + 900.0 * (0.5 + 0.5 * np.sin(2.0 * np.pi * lfo_rate * t))

# Run SVF sample-by-sample with time-varying f
lp_mod = np.empty(len(noise))
bp_s = lp_s = 0.0
for i, x in enumerate(noise):
    fc = cutoff_mod[i]
    res = 0.6
    q = np.sqrt(2.0) * (1.0 - res)
    f_max = np.sqrt(q * q + 4.0) - q
    f = min(2.0 * np.sin(np.pi * fc / FS), f_max * (1.0 - 1e-9))
    lp_s = lp_s + f * bp_s
    hp_s = x - lp_s - q * bp_s
    bp_s = f * hp_s + bp_s
    lp_mod[i] = lp_s

ax.specgram(lp_mod, NFFT=512, Fs=FS, noverlap=256, cmap="inferno", vmin=-80)
ax.plot(t, cutoff_mod, color="cyan", linewidth=0.8, label="cutoff LFO")
ax.set_title("LP output — per-sample cutoff modulation\n(2 Hz LFO, 200–1100 Hz, resonance = 0.6)",
             fontweight="bold")
ax.set_xlabel("Time (s)", fontsize=8)
ax.set_ylabel("Frequency (Hz)", fontsize=8)
ax.tick_params(labelsize=7)
ax.set_ylim(0, 4000)
ax.legend(fontsize=7)

fig.suptitle("Chamberlin State-Variable Filter (SVF)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
