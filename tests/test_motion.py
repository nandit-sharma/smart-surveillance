"""
tests/test_motion.py
─────────────────────
Tests for MotionAnalyzer — uses synthetic frames (no camera needed).
"""
import numpy as np
import pytest

from core.motion.motion_analyzer import MotionAnalyzer


W, H = 640, 480


@pytest.fixture
def analyzer():
    return MotionAnalyzer(W, H)


def blank_frame(val=0):
    """Solid-color BGR frame."""
    return np.full((H, W, 3), val, dtype=np.uint8)


class TestMotionAnalyzer:

    def test_process_returns_tuple(self, analyzer):
        frame = blank_frame(100)
        result = analyzer.process(frame)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_motion_level_in_range(self, analyzer):
        frame = blank_frame(80)
        level, _ = analyzer.process(frame)
        assert 0.0 <= level <= 1.0

    def test_no_motion_on_static_scene(self, analyzer):
        frame = blank_frame(100)
        # Warm up background model
        for _ in range(10):
            analyzer.process(frame)
        level, _ = analyzer.process(frame)
        # Static scene should report near-zero motion
        assert level < 0.15, f"Expected low motion, got {level:.3f}"

    def test_high_motion_on_changed_frame(self, analyzer):
        # Warm up with black frames
        black = blank_frame(0)
        for _ in range(5):
            analyzer.process(black)

        # Suddenly switch to white — huge change
        white = blank_frame(255)
        level, _ = analyzer.process(white)
        assert level > 0.05, f"Expected high motion, got {level:.3f}"

    def test_heatmap_overlay_shape(self, analyzer):
        frame = blank_frame(100)
        analyzer.process(frame)
        overlay = analyzer.get_heatmap_overlay(frame)
        assert overlay.shape == frame.shape

    def test_sudden_motion_false_on_static(self, analyzer):
        frame = blank_frame(100)
        # First call sets prev_gray
        analyzer.detect_sudden_motion(frame)
        detected, frac = analyzer.detect_sudden_motion(frame)
        assert detected is False
        assert frac < 0.12

    def test_sudden_motion_true_on_change(self, analyzer):
        black = blank_frame(0)
        white = blank_frame(255)
        analyzer.detect_sudden_motion(black)
        detected, frac = analyzer.detect_sudden_motion(white)
        assert detected is True
        assert frac > 0.12

    def test_flow_vis_same_size_as_input(self, analyzer):
        frame = blank_frame(128)
        _, vis = analyzer.process(frame)
        assert vis.shape == frame.shape

    def test_heatmap_accumulates(self, analyzer):
        # Motion should increase heatmap values over time
        black = blank_frame(0)
        white = blank_frame(255)
        initial_sum = analyzer._heatmap.sum()

        for i in range(20):
            analyzer.process(black if i % 2 == 0 else white)

        final_sum = analyzer._heatmap.sum()
        assert final_sum > initial_sum, "Heatmap should accumulate motion energy"
