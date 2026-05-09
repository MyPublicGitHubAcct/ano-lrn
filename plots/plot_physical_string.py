"""Karplus-Strong plucked string: waveform, decay envelope, and harmonic spectrum."""

import matplotlib.pyplot as plt
import numpy as np

from python.physical import pluck

FS = 44100
DURATION = 2.0

# --- Waveform at two pickup positions ---
_, sig_bridge = pluck(freq=220.0, fs=FS, duration=DURATION, pickup=0.1, seed=0)
_, sig_neck = pluck(freq=220.0, fs=FS, duration=DURATION, pickup=0.5, seed=0)

t = np.arange(len(sig_bridge)) / FS

# --- Decay comparison at different damping values ---
_, sig_bright = pluck(freq=440.0, fs=FS, duration=DURATION, damping=0.9995, seed=1)
_, sig_muted = pluck(freq=440.0, fs=FS, duration=DURATION, damping=0.990, seed=1)

# --- Harmonic spectrum (first 0.5 s to capture harmonics before decay) ---
n_fft = int(0.5 * FS)
freqs = np.fft.rfftfreq(n_fft, 1 / FS)
spec_bridge = np.abs(np.fft.rfft(sig_bridge[:n_fft]))
spec_neck = np.abs(np.fft.rfft(sig_neck[:n_fft]))

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# Top-left: waveform, first 30 ms
ms30 = int(0.03 * FS)
t_ms = t[:ms30] * 1000
axes[0, 0].plot(t_ms, sig_bridge[:ms30], color="steelblue", lw=0.8, label="pickup=0.1 (bridge)")
axes[0, 0].plot(t_ms, sig_neck[:ms30], color="darkorange", lw=0.8, alpha=0.8, label="pickup=0.5 (neck)")
axes[0, 0].set_title("Waveform — first 30 ms (220 Hz)", fontweight="bold")
axes[0, 0].set_xlabel("Time (ms)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, lw=0.3, alpha=0.5)

# Top-right: full decay envelope
envelope = lambda s: np.abs(s)  # noqa: E731
axes[0, 1].plot(t, 20 * np.log10(np.maximum(envelope(sig_bright), 1e-9)),
                color="steelblue", lw=0.7, label="damping=0.9995")
axes[0, 1].plot(t, 20 * np.log10(np.maximum(envelope(sig_muted), 1e-9)),
                color="darkorange", lw=0.7, label="damping=0.990")
axes[0, 1].axhline(-40, color="crimson", lw=0.8, ls="--", label="−40 dB")
axes[0, 1].set_title("Amplitude Decay — 440 Hz", fontweight="bold")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Amplitude (dB)")
axes[0, 1].set_ylim(-80, 5)
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, lw=0.3, alpha=0.5)

# Bottom-left: harmonic spectrum, bridge pickup
axes[1, 0].plot(freqs[freqs < 3000], 20 * np.log10(spec_bridge[freqs < 3000] + 1e-9),
                color="steelblue", lw=0.8)
axes[1, 0].set_title("Spectrum — pickup=0.1 (bridge-like, all harmonics)", fontweight="bold")
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("Magnitude (dB)")
axes[1, 0].grid(True, lw=0.3, alpha=0.5)

# Bottom-right: harmonic spectrum, neck pickup — even harmonics suppressed
axes[1, 1].plot(freqs[freqs < 3000], 20 * np.log10(spec_neck[freqs < 3000] + 1e-9),
                color="darkorange", lw=0.8)
axes[1, 1].set_title("Spectrum — pickup=0.5 (neck-like, even harmonics suppressed)", fontweight="bold")
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_ylabel("Magnitude (dB)")
axes[1, 1].grid(True, lw=0.3, alpha=0.5)

fig.suptitle("Karplus-Strong Plucked String Synthesis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
