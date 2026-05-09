import numpy as np

from python.time_segment import frame, overlap_add


def test_frame_shape():
    """frame must return a 2-D array with the correct number of frames and the requested frame size."""
    sig = np.sin(2 * np.pi * 440.0 * np.arange(4096) / 44100)
    frames = frame(sig, frame_size=512, hop_size=256)
    expected_frames = 1 + (4096 - 512) // 256
    assert frames.shape == (expected_frames, 512)


def test_overlap_add_shape():
    """overlap_add must return a 1-D array whose length equals (n_frames − 1) × hop + frame_size."""
    frames_arr = np.random.randn(8, 512)
    out = overlap_add(frames_arr, hop_size=256)
    expected_len = (8 - 1) * 256 + 512
    assert len(out) == expected_len


def test_frame_first_frame_matches_signal():
    """The first frame must contain the first frame_size samples of the signal (no implicit padding)."""
    sig = np.arange(1024.0)
    frames = frame(sig, frame_size=256, hop_size=128)
    np.testing.assert_array_equal(frames[0], sig[:256])


def test_frame_second_frame_at_hop_offset():
    """The second frame must start at index hop_size (correct stride through the signal)."""
    sig = np.arange(1024.0)
    frames = frame(sig, frame_size=256, hop_size=128)
    np.testing.assert_array_equal(frames[1], sig[128:384])


def test_frame_no_hop_overlap():
    """With hop_size = frame_size there is no overlap; the two frames must partition the signal exactly."""
    sig = np.arange(512.0)
    frames = frame(sig, frame_size=256, hop_size=256)
    assert frames.shape[0] == 2
    np.testing.assert_array_equal(frames[0], sig[:256])
    np.testing.assert_array_equal(frames[1], sig[256:])


def test_overlap_add_rectangular_reconstructs():
    """Rectangular OLA with 50% overlap must reconstruct the interior of a constant signal to within 5%."""
    frame_size = 512
    hop_size = 256
    sig = np.ones(frame_size + hop_size * 4)
    frames = frame(sig, frame_size, hop_size)
    out = overlap_add(frames, hop_size, window_type="rectangular")
    interior = out[frame_size:-frame_size]
    np.testing.assert_allclose(interior, np.ones_like(interior), atol=0.05)
