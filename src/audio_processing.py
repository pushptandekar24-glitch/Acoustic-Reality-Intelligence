"""
Audio processing and validation module for Acoustic Reality Intelligence.
Handles secure audio loading, decoding, resampling, validation, and metadata extraction.
"""
import io
import math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import librosa
import soundfile as sf

import config


class AudioProcessingError(Exception):
    """Custom exception raised for invalid, corrupted, or unsupported audio files."""
    pass


def validate_audio_file(
    file_bytes: bytes,
    filename: str,
    max_size_bytes: int = config.MAX_FILE_SIZE_BYTES
) -> None:
    """
    Validates uploaded file against security and format constraints.
    """
    if not file_bytes:
        raise AudioProcessingError("The uploaded file is empty (0 bytes). Please upload a valid audio recording.")

    if len(file_bytes) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        actual_mb = len(file_bytes) / (1024 * 1024)
        raise AudioProcessingError(
            f"File size ({actual_mb:.1f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB."
        )

    extension = Path(filename).suffix.lower()
    if extension not in config.SUPPORTED_EXTENSIONS:
        supported = ", ".join(config.SUPPORTED_EXTENSIONS).upper()
        raise AudioProcessingError(
            f"Unsupported file extension '{extension}'. Supported audio formats: {supported}."
        )


def decode_audio_bytes(
    file_bytes: bytes,
    target_sr: Optional[int] = config.TARGET_SAMPLE_RATE,
    mono: bool = config.IS_MONO
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Decodes audio bytes into a float32 numpy waveform and extracts metadata.

    Returns:
        (signal, sample_rate, metadata_dict)
    """
    if not file_bytes:
        raise AudioProcessingError("Cannot decode empty audio bytes.")

    bio = io.BytesIO(file_bytes)
    
    # First, probe metadata using soundfile if possible
    native_sr = None
    native_channels = 1
    native_format = "Unknown"
    native_subtype = "Unknown"
    
    try:
        bio.seek(0)
        with sf.SoundFile(bio) as sfile:
            native_sr = sfile.samplerate
            native_channels = sfile.channels
            native_format = sfile.format
            native_subtype = sfile.subtype
    except Exception:
        # Fall back gracefully if soundfile header reading fails on non-WAV/FLAC
        pass

    # Decode audio using librosa
    try:
        bio.seek(0)
        signal, sr = librosa.load(
            bio,
            sr=target_sr,
            mono=mono
        )
    except Exception as exc:
        raise AudioProcessingError(f"Failed to decode audio recording: {str(exc)}") from exc

    if signal is None or signal.size == 0:
        raise AudioProcessingError("The audio file contains no usable audio samples.")

    # Guard against non-finite values (NaN / Inf)
    if not np.all(np.isfinite(signal)):
        signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)

    duration = float(librosa.get_duration(y=signal, sr=sr))
    peak_val = float(np.max(np.abs(signal)))
    rms_val = float(np.sqrt(np.mean(signal**2)))
    is_silent = bool(peak_val < 1e-4 or rms_val < 1e-5)
    is_clipped = bool(peak_val >= 0.999)

    metadata: Dict[str, Any] = {
        "duration_seconds": round(duration, 3),
        "sample_rate": sr,
        "native_sample_rate": native_sr or sr,
        "channels": native_channels,
        "mono": mono,
        "total_samples": int(signal.size),
        "peak_amplitude": round(peak_val, 4),
        "rms_energy": round(rms_val, 5),
        "is_silent": is_silent,
        "is_clipped": is_clipped,
        "format": native_format,
        "subtype": native_subtype,
        "file_size_bytes": len(file_bytes),
    }

    return signal, sr, metadata


def decode_audio_native(
    file_bytes: bytes
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Decodes audio at its native sample rate without resampling, useful for detailed Laboratory inspection.
    """
    return decode_audio_bytes(file_bytes, target_sr=None, mono=False)

