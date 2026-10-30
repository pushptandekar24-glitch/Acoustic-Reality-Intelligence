"""
Tests for Acoustic Features and Laboratory DSP module.
"""
import pytest
import numpy as np

import config
from src.acoustic_features import (
    extract_model_features,
    compute_audio_laboratory_features
)


def test_extract_model_features_dimensions():
    # 2 seconds of synthetic test signal
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

    features = extract_model_features(signal, sr)

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, 26)
    assert np.all(np.isfinite(features))


def test_extract_model_features_short_signal_padding():
    # Signal shorter than n_fft window (e.g. 500 samples)
    sr = 16000
    signal = np.sin(np.linspace(0, 10, 500)).astype(np.float32)

    features = extract_model_features(signal, sr)
    assert features.shape == (1, 26)
    assert np.all(np.isfinite(features))


def test_extract_model_features_empty_signal():
    with pytest.raises(ValueError, match="empty audio"):
        extract_model_features(np.array([], dtype=np.float32))


def test_compute_audio_laboratory_features():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    lab = compute_audio_laboratory_features(signal, sr)

    required_keys = [
        "stft_db",
        "mel_spec_db",
        "mfcc_matrix",
        "rms_curve",
        "zcr_curve",
        "centroid_curve",
        "bandwidth_curve",
        "rolloff_curve",
        "flatness_curve",
        "summary_stats",
        "disclaimer"
    ]
    for key in required_keys:
        assert key in lab, f"Missing key '{key}' in lab features"

    stats = lab["summary_stats"]
    assert "rms_energy" in stats
    assert "spectral_centroid" in stats
    assert "zero_crossing_rate" in stats
    assert stats["rms_energy"]["mean"] > 0
    assert np.all(np.isfinite(lab["stft_db"]))

