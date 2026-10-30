"""
Acoustic feature extraction and laboratory DSP analysis.
Reproduces the verified 26-MFCC feature pipeline and provides comprehensive laboratory DSP metrics.
"""
from typing import Dict, Any, Tuple
import numpy as np
import librosa
from scipy import stats

import config


def extract_model_features(
    signal: np.ndarray,
    sample_rate: int = config.TARGET_SAMPLE_RATE,
    n_mfcc: int = config.N_MFCC
) -> np.ndarray:
    """
    Extracts the exact 26-dimensional MFCC feature vector matching the original model pipeline:
    - 16 kHz mono audio input
    - 13 MFCC coefficients
    - Mean and Standard Deviation across frames for each coefficient
    - Returns shape (1, 26)

    Feature ordering: [mean_0..mean_12, std_0..std_12]
    """
    if signal is None or signal.size == 0:
        raise ValueError("Cannot extract features from empty audio signal.")

    # Guard against NaN/Inf
    if not np.all(np.isfinite(signal)):
        signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)

    # If clip is extremely short (< n_fft), pad symmetrically to allow STFT
    min_samples = 2048
    if signal.size < min_samples:
        pad_width = min_samples - signal.size
        signal = np.pad(signal, (0, pad_width), mode='constant')

    # Compute 13 MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)
    
    # Calculate mean and standard deviation along time frames (axis=1)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)
    
    feature_vector = np.concatenate((mfcc_means, mfcc_stds))
    
    if feature_vector.shape != (config.FEATURE_DIMENSION,):
        raise ValueError(
            f"Expected {config.FEATURE_DIMENSION} features, got {feature_vector.shape[0]}."
        )

    return feature_vector.reshape(1, -1)


def compute_audio_laboratory_features(
    signal: np.ndarray,
    sample_rate: int = config.TARGET_SAMPLE_RATE
) -> Dict[str, Any]:
    """
    Computes comprehensive DSP and acoustic laboratory features:
    - Waveform summary
    - STFT Spectrogram (dB)
    - Mel-frequency Spectrogram (dB)
    - MFCC matrix (13 coefficients)
    - RMS Energy trajectory
    - Zero-crossing rate (ZCR) trajectory
    - Spectral Centroid trajectory
    - Spectral Bandwidth trajectory
    - Spectral Rolloff & Spectral Flatness
    - Comprehensive summary statistics (mean, std, min, max, median)
    """
    if signal is None or signal.size == 0:
        raise ValueError("Cannot analyze empty signal.")

    # Clean signal
    signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)
    if signal.size < 2048:
        signal = np.pad(signal, (0, 2048 - signal.size), mode='constant')

    # 1. STFT Spectrogram
    stft = np.abs(librosa.stft(signal, n_fft=2048, hop_length=512))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)

    # 2. Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 3. MFCC Matrix
    mfcc_matrix = librosa.feature.mfcc(
        y=signal, sr=sample_rate, n_mfcc=config.N_MFCC, hop_length=512
    )

    # 4. Spectral & Temporal Curves
    rms_curve = librosa.feature.rms(y=signal, hop_length=512)[0]
    zcr_curve = librosa.feature.zero_crossing_rate(y=signal, hop_length=512)[0]
    centroid_curve = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, hop_length=512)[0]
    bandwidth_curve = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, hop_length=512)[0]
    rolloff_curve = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, hop_length=512, roll_percent=0.85)[0]
    flatness_curve = librosa.feature.spectral_flatness(y=signal, hop_length=512)[0]

    # Time axes
    frames = np.arange(len(rms_curve))
    times = librosa.frames_to_time(frames, sr=sample_rate, hop_length=512)

    def stats_dict(arr: np.ndarray) -> Dict[str, float]:
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    summary_stats = {
        "rms_energy": stats_dict(rms_curve),
        "zero_crossing_rate": stats_dict(zcr_curve),
        "spectral_centroid": stats_dict(centroid_curve),
        "spectral_bandwidth": stats_dict(bandwidth_curve),
        "spectral_rolloff": stats_dict(rolloff_curve),
        "spectral_flatness": stats_dict(flatness_curve),
    }

    # Dynamic range & Crest factor
    peak_amp = float(np.max(np.abs(signal)))
    rms_overall = float(np.sqrt(np.mean(signal**2)))
    crest_factor = float(peak_amp / (rms_overall + 1e-9))
    dynamic_range_db = float(20 * np.log10(peak_amp / (rms_overall + 1e-9))) if rms_overall > 0 else 0.0

    summary_stats["crest_factor"] = round(crest_factor, 3)
    summary_stats["dynamic_range_db"] = round(dynamic_range_db, 2)

    return {
        "stft_db": stft_db,
        "mel_spec_db": mel_spec_db,
        "mfcc_matrix": mfcc_matrix,
        "rms_curve": rms_curve,
        "zcr_curve": zcr_curve,
        "centroid_curve": centroid_curve,
        "bandwidth_curve": bandwidth_curve,
        "rolloff_curve": rolloff_curve,
        "flatness_curve": flatness_curve,
        "times": times,
        "summary_stats": summary_stats,
        "sample_rate": sample_rate,
        "disclaimer": (
            "Acoustic features describe measurable physical and spectral characteristics of sound. "
            "They do not constitute forensic proof of authenticity on their own."
        ),
    }

