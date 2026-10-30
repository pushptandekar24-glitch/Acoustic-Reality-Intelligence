"""
Tests for Report Generation (PDF and CSV exports).
"""
import pytest
import numpy as np

from src.report_generator import generate_pdf_report, generate_features_csv


def test_generate_pdf_report_valid_bytes():
    metadata = {
        "duration_seconds": 2.5,
        "sample_rate": 16000,
        "channels": 1,
        "mono": True,
        "file_size_bytes": 80000,
    }
    auth_result = {
        "is_ai_generated": False,
        "headline": "Your audio appears to be REAL",
        "class_name": "Real",
        "real_probability": 0.88,
        "ai_probability": 0.12,
        "model_sha256": "b6d965c31ceba6a6be2483af2e378dead36f59a44ea97f5c3b3cf6f5a8160f91",
    }
    sound_result = {
        "top_event": "Rain & Thunder",
        "top_score": 0.65,
        "top_predictions": [{"event": "Rain & Thunder", "score": 0.65}],
        "backend": "DSP Profiler",
    }
    signal = np.sin(np.linspace(0, 10, 16000)).astype(np.float32)

    pdf_bytes = generate_pdf_report(
        filename="test_audio.wav",
        metadata=metadata,
        authenticity_result=auth_result,
        sound_recognition_result=sound_result,
        signal=signal,
        sample_rate=16000
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # PDF starts with %PDF- header magic bytes
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_features_csv():
    feats = np.arange(26).astype(float).reshape(1, -1)
    meta = {
        "duration_seconds": 1.0,
        "sample_rate": 16000,
        "channels": 1,
    }
    csv_str = generate_features_csv(feats, meta, "test.wav")
    assert "mfcc_1_mean" in csv_str
    assert "mfcc_13_std" in csv_str
    assert "test.wav" in csv_str

