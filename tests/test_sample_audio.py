"""
Tests for Sample Audio Library, Ground-Truth Verification, Pipeline Predictions,
Class-to-Headline Mapping, Two-Sample Comparison, and Missing File Handling.
"""
import json
import pytest
from pathlib import Path
import numpy as np

import config
from src.sample_library import (
    load_sample_manifest,
    resolve_sample_path,
    verify_ground_truth,
    analyze_sample_audio,
    compare_two_samples,
    get_comparison_sample_lists,
)
from src.synthetic_detection import SyntheticAudioDetector, find_model_and_metadata_paths


def test_sample_manifest_loading():
    """Verify that the sample manifest exists, is valid JSON, and contains the required 3 categories."""
    manifest = load_sample_manifest()
    assert "categories" in manifest
    assert "dataset_reference" in manifest

    categories = manifest["categories"]
    expected_categories = ["rain_and_thunder", "birds_and_insects", "siren_and_alarm"]
    for cat in expected_categories:
        assert cat in categories, f"Missing required category: {cat}"
        cat_info = categories[cat]
        assert "name" in cat_info
        assert "samples" in cat_info
        samples = cat_info["samples"]
        assert "sample_a_real" in samples
        assert "sample_b_ai" in samples


def test_sample_manifest_missing_or_corrupt(tmp_path):
    """Verify that manifest loading fails gracefully on missing or invalid files."""
    missing_file = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_sample_manifest(missing_file)

    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{invalid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse"):
        load_sample_manifest(corrupt_file)

    missing_keys = tmp_path / "incomplete.json"
    missing_keys.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="'categories' key missing"):
        load_sample_manifest(missing_keys)


def test_ground_truth_label_verification():
    """Verify ground truth validation logic for real, AI, mismatched, and unverified audio."""
    # 1. Verified Real Audio
    real_sample_info = {
        "is_verified": True,
        "ground_truth_class": 0,
        "ground_truth_label": "Real Audio",
    }
    match_real = verify_ground_truth(real_sample_info, predicted_class=0)
    assert match_real["is_verified"] is True
    assert match_real["prediction_matches_ground_truth"] is True
    assert match_real["status_code"] == "MATCH"
    assert "matches verified ground truth" in match_real["status_text"]

    mismatch_real = verify_ground_truth(real_sample_info, predicted_class=1)
    assert mismatch_real["prediction_matches_ground_truth"] is False
    assert mismatch_real["status_code"] == "MISMATCH"
    assert "does not match verified ground truth" in mismatch_real["status_text"]

    # 2. Verified AI-Generated Audio
    ai_sample_info = {
        "is_verified": True,
        "ground_truth_class": 1,
        "ground_truth_label": "AI-Generated Audio",
    }
    match_ai = verify_ground_truth(ai_sample_info, predicted_class=1)
    assert match_ai["prediction_matches_ground_truth"] is True
    assert match_ai["status_code"] == "MATCH"

    mismatch_ai = verify_ground_truth(ai_sample_info, predicted_class=0)
    assert mismatch_ai["prediction_matches_ground_truth"] is False
    assert mismatch_ai["status_code"] == "MISMATCH"

    # 3. Unverified Demonstration Audio (Must NOT assume or guess truth)
    unverified_info = {
        "is_verified": False,
        "ground_truth_class": None,
        "ground_truth_label": "Demo audio — authenticity label unverified",
    }
    unverified_res = verify_ground_truth(unverified_info, predicted_class=0)
    assert unverified_res["is_verified"] is False
    assert unverified_res["prediction_matches_ground_truth"] is None
    assert unverified_res["status_code"] == "UNVERIFIED"
    assert "unverified" in unverified_res["status_text"].lower()


def test_missing_sample_handling(tmp_path):
    """Verify clean handling of missing sample entries and missing audio files on disk."""
    # A. Sample with filename None
    missing_entry = {
        "sample_id": "missing_test",
        "filename": None,
        "notes": "No sample file available.",
    }
    res_none = analyze_sample_audio(missing_entry, sample_dir=tmp_path)
    assert res_none["status"] == "missing"
    assert "No audio file associated" in res_none["error_message"] or "No sample file" in res_none["error_message"]

    # B. Sample with non-existent file path
    non_existent_entry = {
        "sample_id": "missing_file_test",
        "filename": "file_that_does_not_exist_404.wav",
    }
    res_missing = analyze_sample_audio(non_existent_entry, sample_dir=tmp_path)
    assert res_missing["status"] == "missing"
    assert "not found on disk" in res_missing["error_message"]

    # C. Corrupted audio file
    corrupt_file = tmp_path / "corrupt_audio.wav"
    corrupt_file.write_bytes(b"NOT A REAL RIFF AUDIO HEADER")
    corrupt_entry = {
        "sample_id": "corrupt_test",
        "filename": "corrupt_audio.wav",
    }
    res_corrupt = analyze_sample_audio(corrupt_entry, sample_dir=tmp_path)
    assert res_corrupt["status"] == "error"
    assert "error" in res_corrupt["error_message"].lower()


def test_sample_prediction_using_existing_pipeline():
    """Verify inference pipeline execution on available audio files using the existing model."""
    manifest = load_sample_manifest()
    detector = SyntheticAudioDetector()

    # Check verified birds sample
    birds_real = manifest["categories"]["birds_and_insects"]["samples"]["sample_a_real"]
    res_birds_real = analyze_sample_audio(birds_real, detector=detector)
    assert res_birds_real["status"] == "ready"
    assert res_birds_real["prediction_class"] in [0, 1]
    assert res_birds_real["duration_seconds"] > 0
    assert res_birds_real["sample_rate"] == 16000
    assert res_birds_real["model_verified"] is True
    assert res_birds_real["model_sha256"] == config.EXPECTED_MODEL_SHA256

    # Check verified siren sample
    siren_real = manifest["categories"]["siren_and_alarm"]["samples"]["sample_a_real"]
    res_siren_real = analyze_sample_audio(siren_real, detector=detector)
    assert res_siren_real["status"] == "ready"
    assert res_siren_real["prediction_class"] in [0, 1]
    assert res_siren_real["model_verified"] is True


def test_correct_class_to_headline_mapping_for_samples():
    """Verify that predictions map strictly to the verified headlines without alterations."""
    manifest = load_sample_manifest()
    detector = SyntheticAudioDetector()

    for cat_id, cat_info in manifest["categories"].items():
        for role_key, s in cat_info["samples"].items():
            if not s.get("filename"):
                continue
            res = analyze_sample_audio(s, detector=detector)
            if res.get("status") == "ready":
                p_class = res["prediction_class"]
                headline = res["headline"]
                if p_class == 0:
                    assert headline == "Your audio appears to be REAL"
                    assert res["class_name"] == "Real"
                    assert res["is_ai_generated"] is False
                elif p_class == 1:
                    assert headline == "Your audio appears to be AI-GENERATED"
                    assert res["class_name"] == "AI-generated"
                    assert res["is_ai_generated"] is True


def test_two_sample_comparison_feature():
    """Verify that comparative analysis between real and AI samples produces structured metrics and table rows."""
    manifest = load_sample_manifest()
    sample_a = manifest["categories"]["birds_and_insects"]["samples"]["sample_a_real"]
    sample_b = manifest["categories"]["birds_and_insects"]["samples"]["sample_b_ai"]

    comp = compare_two_samples(sample_a, sample_b)
    assert comp["both_ready"] is True
    assert "sample_a" in comp
    assert "sample_b" in comp
    assert len(comp["comparison_rows"]) > 5

    # Check critical comparison rows
    attributes = [r["Attribute"] for r in comp["comparison_rows"]]
    assert "File Name" in attributes
    assert "Duration" in attributes
    assert "Sample Rate & Channels" in attributes
    assert "Verified Ground Truth" in attributes
    assert "Model Prediction" in attributes
    assert "Headline" in attributes
    assert "Model P(AI-generated)" in attributes
    assert "Ground-Truth Verification" in attributes

    # Verify disclaimer is included
    assert "disclaimer" in comp
    assert "does not guarantee correct" in comp["disclaimer"]


def test_comparison_with_missing_sample():
    """Verify that comparison gracefully handles missing files without crashing."""
    manifest = load_sample_manifest()
    real_sample = manifest["categories"]["rain_and_thunder"]["samples"]["sample_a_real"]
    missing_sample = manifest["categories"]["rain_and_thunder"]["samples"]["sample_b_ai"]

    comp = compare_two_samples(real_sample, missing_sample)
    assert comp["both_ready"] is False
    assert comp["sample_b"]["status"] == "missing"
    assert len(comp["comparison_rows"]) == 0


def test_comparison_sample_lists_populated():
    """Verify that dropdown options are populated with valid on-disk samples."""
    manifest = load_sample_manifest()
    real_list, ai_list = get_comparison_sample_lists(manifest)
    assert len(real_list) >= 2, f"Expected at least 2 real samples on disk, found {len(real_list)}"
    assert len(ai_list) >= 2, f"Expected at least 2 AI samples on disk, found {len(ai_list)}"
    for item in real_list:
        assert Path(config.SAMPLE_AUDIO_DIR / item["filename"]).exists()
    for item in ai_list:
        assert Path(config.SAMPLE_AUDIO_DIR / item["filename"]).exists()

