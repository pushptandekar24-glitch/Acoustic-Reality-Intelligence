"""
Tests for Segment Analysis and Audio Comparison modules.
"""
import pytest
import numpy as np

from src.segment_analysis import analyze_audio_segments
from src.recording_comparison import compare_recordings
from src.synthetic_detection import SyntheticAudioDetector


def test_analyze_audio_segments():
    sr = 16000
    # 4 seconds signal with sudden amplitude jump at 2s
    t = np.linspace(0, 4.0, sr * 4, endpoint=False)
    sig1 = 0.2 * np.sin(2 * np.pi * 300 * t[:sr * 2])
    sig2 = 0.9 * np.sin(2 * np.pi * 300 * t[sr * 2:])
    signal = np.concatenate([sig1, sig2]).astype(np.float32)

    detector = SyntheticAudioDetector()
    res = analyze_audio_segments(
        signal,
        sample_rate=sr,
        segment_duration_sec=1.0,
        overlap_ratio=0.5,
        classifier_model=detector.model
    )

    assert res["total_segments"] > 3
    assert len(res["segments"]) == res["total_segments"]
    assert "anomaly_count" in res
    assert "timeline_data" in res


def test_compare_recordings():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig_a = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sig_b = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    meta = {"duration_seconds": 1.0, "sample_rate": sr, "channels": 1}

    res = compare_recordings(sig_a, sr, meta, sig_b, sr, meta)

    assert "similarity_score_percent" in res
    # Identical signals should have very high similarity (~100%)
    assert res["similarity_score_percent"] > 98.0
    assert res["cosine_distance"] < 0.05

