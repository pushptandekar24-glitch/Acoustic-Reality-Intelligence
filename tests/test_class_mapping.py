"""
Tests for Class Mapping, Checksum Validation, and Headline Generation.
Verifies that:
- Checksum matches b6d965c31ceba6a6be2483af2e378dead36f59a44ea97f5c3b3cf6f5a8160f91
- Class 0 produces 'Your audio appears to be REAL'
- Class 1 produces 'Your audio appears to be AI-GENERATED'
- Checksum mismatch or invalid metadata safely prevents incorrect prediction labeling
"""
import pytest
import numpy as np
from pathlib import Path
import json

import config
from src.synthetic_detection import (
    compute_file_sha256,
    verify_and_load_model,
    SyntheticAudioDetector,
    ModelVerificationError,
    find_model_and_metadata_paths
)


def test_model_file_sha256_checksum():
    model_path, _ = find_model_and_metadata_paths()
    actual_hash = compute_file_sha256(model_path)
    assert actual_hash == config.EXPECTED_MODEL_SHA256, (
        f"Model SHA256 checksum mismatch! Expected: {config.EXPECTED_MODEL_SHA256}, got: {actual_hash}"
    )


def test_verified_metadata_content():
    _, metadata_path = find_model_and_metadata_paths()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata.get("metadata_version") == 1
    assert metadata.get("model_sha256") == config.EXPECTED_MODEL_SHA256
    class_names = metadata.get("class_names")
    assert class_names.get("0") == "Real"
    assert class_names.get("1") == "AI-generated"

    headlines = metadata.get("headlines")
    assert headlines.get("0") == "Your audio appears to be REAL"
    assert headlines.get("1") == "Your audio appears to be AI-GENERATED"


def test_headlines_both_classes_explicitly():
    """Explicitly verify headline mapping for Class 0 and Class 1."""
    detector = SyntheticAudioDetector()

    # Verify Class 0 headline
    headline_0 = detector.headlines[0]
    assert headline_0 == "Your audio appears to be REAL"

    # Verify Class 1 headline
    headline_1 = detector.headlines[1]
    assert headline_1 == "Your audio appears to be AI-GENERATED"


def test_prediction_pipeline_end_to_end():
    detector = SyntheticAudioDetector()
    sr = 16000
    t = np.linspace(0, 1.5, sr * 2, endpoint=False)
    dummy_signal = 0.3 * np.sin(2 * np.pi * 400 * t).astype(np.float32)

    res = detector.predict_audio(dummy_signal, sr)

    assert "prediction_class" in res
    assert res["prediction_class"] in [0, 1]
    assert "headline" in res
    if res["prediction_class"] == 0:
        assert res["headline"] == "Your audio appears to be REAL"
        assert res["is_ai_generated"] is False
    else:
        assert res["headline"] == "Your audio appears to be AI-GENERATED"
        assert res["is_ai_generated"] is True

    assert "real_probability" in res
    assert "ai_probability" in res
    assert 0.0 <= res["real_probability"] <= 1.0
    assert 0.0 <= res["ai_probability"] <= 1.0


def test_checksum_mismatch_prevents_labeling(monkeypatch):
    """Verifies that an altered expected SHA-256 causes a fail-safe exception."""
    monkeypatch.setattr(config, "EXPECTED_MODEL_SHA256", "0000000000000000000000000000000000000000000000000000000000000000")
    with pytest.raises(ModelVerificationError, match="SHA-256 mismatch"):
        verify_and_load_model()

