"""
Comparative acoustic analysis module for Acoustic Reality Intelligence.
Compares two recordings side-by-side across temporal, spectral, and MFCC feature domains,
computing documented similarity and distance metrics.
"""
from typing import Dict, Any, Tuple
import numpy as np
import librosa
from scipy.spatial.distance import cosine, euclidean

import config
from src.acoustic_features import extract_model_features, compute_audio_laboratory_features


def compare_recordings(
    signal_a: np.ndarray,
    sr_a: int,
    meta_a: Dict[str, Any],
    signal_b: np.ndarray,
    sr_b: int,
    meta_b: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs comparative analysis between Recording A and Recording B.
    Computes:
    - Metadata comparison
    - Acoustic summary statistics comparison
    - 26-MFCC feature distance & Cosine similarity
    - Dynamic Time Warping (DTW) distance on MFCC sequences
    - Documented interpretations and limitations
    """
    # 1. Ensure resampled signals for unified DSP feature extraction
    if sr_a != config.TARGET_SAMPLE_RATE:
        sig_a_16k = librosa.resample(signal_a, orig_sr=sr_a, target_sr=config.TARGET_SAMPLE_RATE)
    else:
        sig_a_16k = signal_a

    if sr_b != config.TARGET_SAMPLE_RATE:
        sig_b_16k = librosa.resample(signal_b, orig_sr=sr_b, target_sr=config.TARGET_SAMPLE_RATE)
    else:
        sig_b_16k = signal_b

    # 2. Extract 26-MFCC vectors
    feats_a = extract_model_features(sig_a_16k, config.TARGET_SAMPLE_RATE).flatten()
    feats_b = extract_model_features(sig_b_16k, config.TARGET_SAMPLE_RATE).flatten()

    # 3. Compute Metrics
    # Cosine distance: 1 - cos_sim. Range: [0, 2] -> convert to similarity percentage
    cos_dist = float(cosine(feats_a, feats_b))
    cos_similarity = max(0.0, min(1.0, 1.0 - cos_dist))
    similarity_percentage = round(cos_similarity * 100, 1)

    # Normalized Euclidean distance
    norm_a = feats_a / (np.linalg.norm(feats_a) + 1e-9)
    norm_b = feats_b / (np.linalg.norm(feats_b) + 1e-9)
    norm_euclidean_dist = float(euclidean(norm_a, norm_b))

    # Dynamic Time Warping (DTW) on raw MFCC frame sequences
    mfcc_seq_a = librosa.feature.mfcc(y=sig_a_16k, sr=config.TARGET_SAMPLE_RATE, n_mfcc=config.N_MFCC)
    mfcc_seq_b = librosa.feature.mfcc(y=sig_b_16k, sr=config.TARGET_SAMPLE_RATE, n_mfcc=config.N_MFCC)

    try:
        # Compute DTW distance between the two time series
        dtw_cost, _ = librosa.sequence.dtw(X=mfcc_seq_a, Y=mfcc_seq_b, metric='cosine')
        normalized_dtw = float(dtw_cost[-1, -1] / (mfcc_seq_a.shape[1] + mfcc_seq_b.shape[1]))
    except Exception:
        normalized_dtw = float(cos_dist)

    # 4. Extract Laboratory Features for both
    lab_a = compute_audio_laboratory_features(sig_a_16k, config.TARGET_SAMPLE_RATE)
    lab_b = compute_audio_laboratory_features(sig_b_16k, config.TARGET_SAMPLE_RATE)

    # Summary Delta Comparison
    metrics_delta = {
        "duration_sec": {
            "file_a": meta_a.get("duration_seconds", 0.0),
            "file_b": meta_b.get("duration_seconds", 0.0),
            "difference": round(abs(meta_a.get("duration_seconds", 0.0) - meta_b.get("duration_seconds", 0.0)), 2),
        },
        "rms_energy_mean": {
            "file_a": round(lab_a["summary_stats"]["rms_energy"]["mean"], 5),
            "file_b": round(lab_b["summary_stats"]["rms_energy"]["mean"], 5),
            "difference": round(abs(lab_a["summary_stats"]["rms_energy"]["mean"] - lab_b["summary_stats"]["rms_energy"]["mean"]), 5),
        },
        "spectral_centroid_mean": {
            "file_a": round(lab_a["summary_stats"]["spectral_centroid"]["mean"], 1),
            "file_b": round(lab_b["summary_stats"]["spectral_centroid"]["mean"], 1),
            "difference": round(abs(lab_a["summary_stats"]["spectral_centroid"]["mean"] - lab_b["summary_stats"]["spectral_centroid"]["mean"]), 1),
        },
        "zero_crossing_rate_mean": {
            "file_a": round(lab_a["summary_stats"]["zero_crossing_rate"]["mean"], 4),
            "file_b": round(lab_b["summary_stats"]["zero_crossing_rate"]["mean"], 4),
            "difference": round(abs(lab_a["summary_stats"]["zero_crossing_rate"]["mean"] - lab_b["summary_stats"]["zero_crossing_rate"]["mean"]), 4),
        },
    }

    return {
        "similarity_score_percent": similarity_percentage,
        "cosine_distance": round(cos_dist, 4),
        "normalized_euclidean_distance": round(norm_euclidean_dist, 4),
        "dtw_normalized_cost": round(normalized_dtw, 4),
        "metrics_comparison": metrics_delta,
        "lab_a": lab_a,
        "lab_b": lab_b,
        "features_a": feats_a.tolist(),
        "features_b": feats_b.tolist(),
        "distance_method": "Cosine Metric on 26-MFCC Vectors & Normalized Dynamic Time Warping (DTW)",
        "limitations": (
            "Interpretation Note: A high acoustic similarity score indicates comparable spectral distribution "
            "and timbral envelopes. It does NOT prove that the recordings share the same physical sound source, "
            "same recording environment, or identical provenance."
        ),
    }

