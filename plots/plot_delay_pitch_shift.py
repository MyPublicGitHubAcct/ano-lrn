"""Pitch shifter: 440 Hz sine shifted up and down by 7 semitones."""

import matplotlib.pyplot as plt
import numpy as np

from python.delay import pitch_shift

FS = 44100
DURATION = 1.0
N = int(DURATION * FS)
t = np.arange(N) / FS

freq = 440.0
signal = np.sin(2 * np.pi * freq * t)

out_up = pitch_shift(signal, FS, semitones=7)
out_dn = pitch_shift(signal, FS, semitones=-7)

# Buffer size (must match implementation)
B = 1
while B < int(FS * 0.05):
    B <<= 1

expected_up = freq * 2 ** (7 / 12)   # ≈ 659.3 Hz
expected_dn = freq * 2 ** (-7 / 12)  # ≈ 293.7 Hz

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# ── Waveform panels (zoom to 20 ms after startup) ──────────────────────────
zoom_start = B + int(0.002 * FS)
zoom_end = zoom_start + int(0.020 * FS)
t_zoom = t[zoom_start:zoom_end] * 1000

for col, (label, out, exp) in enumerate([
    ("+7 st → {:.1f} Hz".format(expected_up), out_up, expected_up),
    ("−7 st → {:.1f} Hz".format(expected_dn), out_dn, expected_dn),
]):
    ax = axes[0, col]
    ax.plot(t_zoom, signal[zoom_start:zoom_end], color="gray",
            linewidth=0.8, alpha=0.6, label=f"Input {freq:.0f} Hz")
    ax.plot(t_zoom, out[zoom_start:zoom_end], color="steelblue" if col == 0 else "tomato",
            linewidth=1.0, label=label)
    ax.set_title(f"Waveform — {label}", fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.tick_params(labelsize=8)

# ── Spectrum panels ─────────────────────────────────────────────────────────
# Use steady-state region (skip first B samples)
steady = slice(B, N)
nfft = (N - B) * 4  # zero-padding for smoother display

for col, (label, out, exp, color) in enumerate([
    ("+7 semitones", out_up, expected_up, "steelblue"),
    ("−7 semitones", out_dn, expected_dn, "tomato"),
]):
    ax = axes[1, col]
    spec_in = 20 * np.log10(np.abs(np.fft.rfft(signal[steady], n=nfft)) + 1e-9)
    spec_out = 20 * np.log10(np.abs(np.fft.rfft(out[steady], n=nfft)) + 1e-9)
    freqs = np.fft.rfftfreq(nfft, 1.0 / FS)

    ax.plot(freqs, spec_in, color="gray", linewidth=0.7, alpha=0.6, label=f"Input {freq:.0f} Hz")
    ax.plot(freqs, spec_out, color=color, linewidth=0.9, label=f"Output {exp:.1f} Hz")
    ax.axvline(freq, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(exp, color=color, linewidth=0.9, linestyle="--",
               label=f"Expected {exp:.1f} Hz")
    ax.set_xlim(0, max(freq, exp) * 3)
    ax.set_ylim(-80, 10)
    ax.set_title(f"Spectrum — {label}", fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.tick_params(labelsize=8)

# ── Grain pointer phase panels ───────────────────────────────────────────────
n_disp = np.arange(min(3 * B, N), dtype=float)
t_disp = n_disp / FS * 1000

for col, semitones in enumerate([7, -7]):
    r = 2.0 ** (semitones / 12.0)
    p0 = (B / 2.0 + (1.0 - r) * n_disp) % B
    p1 = (p0 + B / 2.0) % B
    win0 = 0.5 * (1.0 - np.cos(2.0 * np.pi * p0 / B))
    win1 = 0.5 * (1.0 - np.cos(2.0 * np.pi * p1 / B))

    ax = axes[2, col]
    ax.plot(t_disp, win0, color="steelblue", linewidth=0.9, label="Head 0 Hann")
    ax.plot(t_disp, win1, color="tomato", linewidth=0.9, linestyle="--", label="Head 1 Hann")
    ax.plot(t_disp, win0 + win1, color="black", linewidth=0.6, linestyle=":",
            alpha=0.5, label="Sum (= 1)")
    ax.set_title(
        f"Hann crossfade envelopes — {'+' if semitones > 0 else ''}{semitones} st  "
        f"(r = {r:.4f}, grain period ≈ {B / abs(1 - r):.0f} smp)",
        fontweight="bold",
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Window weight")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.tick_params(labelsize=8)

fig.suptitle(
    f"Time-Domain Pitch Shifter — 440 Hz sine, ±7 semitones  "
    f"(buffer B = {B} smp = {B / FS * 1000:.1f} ms @ {FS} Hz)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.show()
