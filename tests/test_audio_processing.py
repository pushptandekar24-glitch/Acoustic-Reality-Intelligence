"""
Tests for Audio Processing module (decoding, resampling, validation, error handling).
"""
import io
import pytest
import numpy as np
import soundfile as sf

from src.audio_processing import (
    validate_audio_file,
    decode_audio_bytes,
    AudioProcessingError
)
import config


def create_test_wav_bytes(duration_sec=1.0, sr=16000, freq=440.0) -> bytes:
    """Utility helper: generates synthetic test WAV audio bytes."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Sine wave
    signal = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV")
    buf.seek(0)
    return buf.getvalue()


def test_validate_audio_empty_file():
    with pytest.raises(AudioProcessingError, match="empty"):
        validate_audio_file(b"", "test.wav")


def test_validate_audio_unsupported_extension():
    with pytest.raises(AudioProcessingError, match="Unsupported file extension"):
        validate_audio_file(b"dummy content", "test.exe")


def test_validate_audio_exceeds_size():
    with pytest.raises(AudioProcessingError, match="exceeds maximum allowed"):
        validate_audio_file(b"x" * 100, "test.wav", max_size_bytes=50)


def test_decode_valid_wav_audio():
    wav_bytes = create_test_wav_bytes(duration_sec=1.5, sr=16000, freq=440.0)
    signal, sr, meta = decode_audio_bytes(wav_bytes)

    assert isinstance(signal, np.ndarray)
    assert sr == 16000
    assert meta["sample_rate"] == 16000
    assert meta["mono"] is True
    assert meta["duration_seconds"] == pytest.approx(1.5, rel=1e-2)
    assert meta["total_samples"] == int(1.5 * 16000)
    assert meta["is_silent"] is False
    assert np.all(np.isfinite(signal))


def test_decode_corrupted_audio():
    corrupted_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00invalid data payload"
    with pytest.raises(AudioProcessingError):
        decode_audio_bytes(corrupted_bytes)


def test_decode_silence_audio():
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    signal = np.zeros_like(t, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, 16000, format="WAV")
    buf.seek(0)
    wav_bytes = buf.getvalue()

    signal, sr, meta = decode_audio_bytes(wav_bytes)
    assert meta["is_silent"] is True

