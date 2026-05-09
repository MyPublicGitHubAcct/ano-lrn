"""frame and overlap_add: segmentation and reconstruction."""

import matplotlib.pyplot as plt
import numpy as np

from python.generators import generate_chirp
from python.time_segment import apply_window, frame, overlap_add

FS = 44100
DURATION = 0.05
FRAME_SIZE = 512
COLORS = ["steelblue", "darkorange", "mediumseagreen", "crimson", "mediumpurple"]

_, sig = generate_chirp(start_freq=200.0, end_freq=2000.0, fs=FS, duration=DURATION)
t_ms = np.arange(len(sig)) / FS * 1000

fig, axes = plt.subplots(2, 3, figsize=(14, 7))

# Original signal
axes[0, 0].plot(t_ms, sig, linewidth=0.8, color="steelblue")
axes[0, 0].set_title("Input Signal", fontweight="bold")
axes[0, 0].set_xlabel("Time (ms)")
axes[0, 0].set_ylabel("Amplitude")

# Frames at different hop sizes
HOP_SIZES = [FRAME_SIZE, FRAME_SIZE // 2, FRAME_SIZE // 4]
HOP_LABELS = ["No overlap (hop=N)", "50% overlap (hop=N/2)", "75% overlap (hop=N/4)"]

for col, (hop, label) in enumerate(zip(HOP_SIZES, HOP_LABELS), start=1):
    frames_arr = frame(sig, frame_size=FRAME_SIZE, hop_size=hop)
    ax = axes[0, col] if col < 3 else axes[1, 0]
    # Show first 5 frames
    for i, (f_data, color) in enumerate(zip(frames_arr[:5], COLORS)):
        t_frame = (np.arange(FRAME_SIZE) + i * hop) / FS * 1000
        ax.plot(t_frame, f_data, linewidth=0.7, color=color, alpha=0.7, label=f"Frame {i}")
    ax.set_title(f"frame() — {label}\n({frames_arr.shape[0]} frames)", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=7)
    ax.set_ylabel("Amplitude", fontsize=7)
    ax.legend(fontsize=5, loc="upper right")
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.5)

# overlap_add reconstruction — perfect reconstruction check
HOP_OLA = FRAME_SIZE // 2
frames_arr = frame(sig, frame_size=FRAME_SIZE, hop_size=HOP_OLA)
recon = overlap_add(frames_arr, hop_size=HOP_OLA, window_type="hann")
n = min(len(t_ms), len(recon))
axes[1, 1].plot(t_ms[:n], sig[:n], linewidth=1.0, color="steelblue", alpha=0.5, label="Original")
axes[1, 1].plot(t_ms[:n], recon[:n], linewidth=0.8, color="crimson", linestyle="--", label="OLA reconstruct")
axes[1, 1].set_title("overlap_add() — Perfect Reconstruction (50% hop, Hann)", fontweight="bold")
axes[1, 1].set_xlabel("Time (ms)")
axes[1, 1].set_ylabel("Amplitude")
axes[1, 1].legend(fontsize=7)

# Reconstruction error
error = sig[:n] - recon[:n]
axes[1, 2].plot(t_ms[:n], error, linewidth=0.7, color="mediumseagreen")
axes[1, 2].set_title("Reconstruction Error (should be ≈ 0)", fontweight="bold")
axes[1, 2].set_xlabel("Time (ms)")
axes[1, 2].set_ylabel("Error")

for ax in axes.flatten():
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)

fig.suptitle("Framing and Overlap-Add Reconstruction", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
