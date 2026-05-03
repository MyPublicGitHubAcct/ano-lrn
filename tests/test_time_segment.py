import numpy as np
import pytest

from python.time_segment import apply_window, frame, overlap_add

FS = 44100


def _sine(n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * 440.0 * t)


# ── Shape ─────────────────────────────────────────────────────────────────────

def test_apply_window_shape():
    sig = _sine()
    assert apply_window(sig).shape == sig.shape


def test_frame_shape():
    sig = _sine(4096)
    frames = frame(sig, frame_size=512, hop_size=256)
    expected_frames = 1 + (4096 - 512) // 256
    assert frames.shape == (expected_frames, 512)


def test_overlap_add_shape():
    frames_arr = np.random.randn(8, 512)
    out = overlap_add(frames_arr, hop_size=256)
    expected_len = (8 - 1) * 256 + 512
    assert len(out) == expected_len


# ── apply_window ──────────────────────────────────────────────────────────────

def test_apply_window_hann_zeros_endpoints():
    sig = np.ones(1024)
    out = apply_window(sig, "hann")
    # Hann window is 0 at first and last sample.
    assert out[0] == pytest.approx(0.0, abs=1e-10)
    assert out[-1] == pytest.approx(0.0, abs=1e-10)


def test_apply_window_rectangular_is_identity():
    sig = _sine()
    np.testing.assert_array_equal(apply_window(sig, "rectangular"), sig)


def test_apply_window_reduces_amplitude():
    sig = np.ones(1024)
    out = apply_window(sig, "hann")
    assert np.max(out) < 1.0


def test_apply_window_unknown_raises():
    with pytest.raises(ValueError):
        apply_window(np.ones(64), "unknown_window")


# ── frame ─────────────────────────────────────────────────────────────────────

def test_frame_first_frame_matches_signal():
    sig = np.arange(1024.0)
    frames = frame(sig, frame_size=256, hop_size=128)
    np.testing.assert_array_equal(frames[0], sig[:256])


def test_frame_second_frame_at_hop_offset():
    sig = np.arange(1024.0)
    frames = frame(sig, frame_size=256, hop_size=128)
    np.testing.assert_array_equal(frames[1], sig[128:384])


def test_frame_no_hop_overlap():
    sig = np.arange(512.0)
    frames = frame(sig, frame_size=256, hop_size=256)
    assert frames.shape[0] == 2
    np.testing.assert_array_equal(frames[0], sig[:256])
    np.testing.assert_array_equal(frames[1], sig[256:])


# ── overlap_add ───────────────────────────────────────────────────────────────

def test_overlap_add_rectangular_reconstructs():
    # With rectangular window and 50% overlap, OLA should reconstruct a flat signal.
    frame_size = 512
    hop_size = 256
    sig = np.ones(frame_size + hop_size * 4)
    frames = frame(sig, frame_size, hop_size)
    out = overlap_add(frames, hop_size, window_type="rectangular")
    # Interior samples (clear of edge effects) should be 1.0.
    interior = out[frame_size: -frame_size]
    np.testing.assert_allclose(interior, np.ones_like(interior), atol=0.05)
