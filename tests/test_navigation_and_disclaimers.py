"""
Regression tests for three-option navigation, class mapping, model checksum,
voice-specific limitation notice, and prediction headline consistency.
"""
import pytest
import config
from src.synthetic_detection import (
    find_model_and_metadata_paths,
    compute_file_sha256,
    SyntheticAudioDetector
)
from src.ui.detection import VOICE_LIMITATION_NOTICE
from src.ui.overview import render_overview_page
from src.ui.detection import render_detection_page
from src.ui.about import render_about_page
import app


def test_three_option_navigation_options():
    """Verify that the navigation menu strictly contains the three requested options."""
    expected_options = [
        "Overview",
        "Voice / Audio Detection",
        "About",
    ]
    assert app.NAV_OPTIONS == expected_options, (
        f"Navigation options mismatch. Expected {expected_options}, got {app.NAV_OPTIONS}"
    )
    assert len(app.NAV_OPTIONS) == 3


def test_model_checksum_integrity():
    """Verify that the model file hash matches the expected SHA-256 fingerprint."""
    model_path, _ = find_model_and_metadata_paths()
    actual_sha = compute_file_sha256(model_path)
    assert actual_sha == config.EXPECTED_MODEL_SHA256, (
        f"SHA-256 mismatch! Expected {config.EXPECTED_MODEL_SHA256}, got {actual_sha}"
    )


def test_verified_class_mapping_and_headlines():
    """Verify class 0 is Real and class 1 is AI-generated with exact headlines."""
    detector = SyntheticAudioDetector()
    assert detector.class_names[0] == "Real"
    assert detector.class_names[1] == "AI-generated"

    assert detector.headlines[0] == "Your audio appears to be REAL"
    assert detector.headlines[1] == "Your audio appears to be AI-GENERATED"


def test_voice_detection_limitation_notice():
    """Verify the exact mandatory voice limitation notice is present."""
    expected_phrases = [
        "experimental general-audio authenticity assessment",
        "existing SVM model",
        "not an independently validated human-voice or AI voice-clone detector",
        "Results may be incorrect, particularly for unfamiliar sound types, speech, cloned voices",
    ]
    for phrase in expected_phrases:
        assert phrase in VOICE_LIMITATION_NOTICE, f"Missing required phrasing: '{phrase}'"


def test_view_renderers_importable():
    """Verify all 3 view renderers are callable functions."""
    assert callable(render_overview_page)
    assert callable(render_detection_page)
    assert callable(render_about_page)

