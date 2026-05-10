"""Tape echo: impulse echo trail with wow/flutter smearing shown in a spectrogram."""

import matplotlib.pyplot as plt
import numpy as np

from python.delay import tape_delay

FS = 44100
DURATION = 1.2  # seconds
N = int(DURATION * FS)

# Short noise burst as input — broadband so the spectrogram shows smearing clearly
rng = np.random.default_rng(42)
burst_len = int(0.015 * FS)  # 15 ms burst
signal = np.zeros(N)
signal[:burst_len] = rng.standard_normal(burst_len) * np.hanning(burst_len)

t_s = np.arange(N) / FS
t_ms = t_s * 1000

# Tape delay with pronounced wow and flutter for visible spectrogram smearing
DELAY_TIME = 0.18  # 180 ms tape loop
FEEDBACK = 0.65
FLUTTER_RATE = 10.0
FLUTTER_DEPTH = 0.002   # 2 ms peak deviation
WOW_RATE = 0.6
WOW_DEPTH = 0.005       # 5 ms peak deviation
SATURATION = 0.3

_, echo = tape_delay(
    signal, FS,
    delay_time=DELAY_TIME,
    feedback=FEEDBACK,
    flutter_rate=FLUTTER_RATE,
    flutter_depth=FLUTTER_DEPTH,
    wow_rate=WOW_RATE,
    wow_depth=WOW_DEPTH,
    saturation=SATURATION,
)

# Mix for display: dry + echo
mixed = signal + echo

# Delay modulation curve for the bottom panel
n_axis = np.arange(N)
wow_curve = WOW_DEPTH * 1000 * np.sin(2 * np.pi * WOW_RATE * n_axis / FS)
flutter_curve = FLUTTER_DEPTH * 1000 * np.sin(2 * np.pi * FLUTTER_RATE * n_axis / FS)
delay_curve_ms = DELAY_TIME * 1000 + wow_curve + flutter_curve

fig, axes = plt.subplots(3, 1, figsize=(14, 11))

# --- Panel 1: waveform ---
ax = axes[0]
ax.plot(t_ms, signal, color="gray", linewidth=0.6, alpha=0.7, label="Dry")
ax.plot(t_ms, echo, color="steelblue", linewidth=0.6, alpha=0.85, label="Echo trail")
ax.set_ylabel("Amplitude")
ax.set_xlim(0, DURATION * 1000)
ax.set_title(
    f"Tape Echo — dry burst + echo trail  "
    f"(delay={DELAY_TIME*1000:.0f} ms, feedback={FEEDBACK}, saturation={SATURATION})",
    fontweight="bold",
)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, linewidth=0.3, alpha=0.4)
ax.tick_params(labelsize=8)

# --- Panel 2: spectrogram of mixed signal ---
ax = axes[1]
WIN = 1024
HOP = 256
window = np.hanning(WIN)
n_frames = (N - WIN) // HOP
spec = np.zeros((WIN // 2 + 1, n_frames))
for i in range(n_frames):
    frame = mixed[i * HOP: i * HOP + WIN] * window
    spec[:, i] = np.abs(np.fft.rfft(frame))

spec_db = 20 * np.log10(spec + 1e-9)
spec_db -= spec_db.max()

frame_times_ms = (np.arange(n_frames) * HOP + WIN // 2) / FS * 1000
freqs = np.fft.rfftfreq(WIN, 1.0 / FS)

ax.pcolormesh(
    frame_times_ms, freqs / 1000, spec_db,
    vmin=-60, vmax=0, cmap="inferno", shading="auto",
)
ax.set_ylabel("Frequency (kHz)")
ax.set_ylim(0, 8)
ax.set_xlim(0, DURATION * 1000)
ax.set_title(
    f"Spectrogram — wow/flutter smearing visible on each echo "
    f"(flutter {FLUTTER_RATE} Hz ±{FLUTTER_DEPTH*1000:.0f} ms, "
    f"wow {WOW_RATE} Hz ±{WOW_DEPTH*1000:.0f} ms)",
    fontweight="bold",
)
ax.grid(True, linewidth=0.3, alpha=0.2, color="white")
ax.tick_params(labelsize=8)

# Annotate expected echo arrival times
for k in range(1, 7):
    arr_ms = DELAY_TIME * 1000 * k
    if arr_ms < DURATION * 1000:
        ax.axvline(arr_ms, color="cyan", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.text(arr_ms + 2, 7.5, f"echo {k}", fontsize=6, color="cyan", alpha=0.7)

# --- Panel 3: delay modulation curve ---
ax = axes[2]
ax.plot(t_ms, delay_curve_ms, color="darkorange", linewidth=0.8, label="Total delay")
ax.plot(t_ms, DELAY_TIME * 1000 + wow_curve, color="steelblue", linewidth=0.8,
        linestyle="--", alpha=0.7, label=f"Wow only ({WOW_RATE} Hz)")
ax.plot(t_ms, DELAY_TIME * 1000 + flutter_curve, color="mediumseagreen", linewidth=0.6,
        linestyle=":", alpha=0.7, label=f"Flutter only ({FLUTTER_RATE} Hz)")
ax.axhline(DELAY_TIME * 1000, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Delay (ms)")
ax.set_xlim(0, 300)  # zoom to first 300 ms to see the modulation clearly
ax.set_title("Tape delay modulation — wow (slow) and flutter (fast) perturbations",
             fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, linewidth=0.3, alpha=0.4)
ax.tick_params(labelsize=8)

fig.suptitle(
    "Tape Echo Emulator — Wow, Flutter, Feedback, and Soft-Clipping Saturation",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.show()
