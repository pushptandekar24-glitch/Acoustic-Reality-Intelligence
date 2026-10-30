"""
Segment-based audio analysis and acoustic anomaly exploration.
Divides audio into time windows, tracks acoustic feature drift over time,
flags unusual acoustic discontinuities, and evaluates segment-level authenticity.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import librosa

import config
from src.acoustic_features import extract_model_features


def analyze_audio_segments(
    signal: np.ndarray,
    sample_rate: int = config.TARGET_SAMPLE_RATE,
    segment_duration_sec: float = 2.0,
    overlap_ratio: float = 0.5,
    classifier_model: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Partitions the recording into overlapping time segments and evaluates:
    - Acoustic statistics per segment
    - Feature evolution trajectory
    - Statistical anomaly detection (Z-score based transition detection)
    - Optional segment-level SVM authenticity classification
    """
    if signal is None or signal.size == 0:
        raise ValueError("Cannot perform segment analysis on empty audio.")

    signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)
    total_samples = signal.size
    total_duration = total_samples / sample_rate

    seg_samples = int(segment_duration_sec * sample_rate)
    hop_samples = max(int(seg_samples * (1.0 - overlap_ratio)), sample_rate // 4)

    if seg_samples > total_samples:
        seg_samples = total_samples
        hop_samples = total_samples

    segments: List[Dict[str, Any]] = []
    rms_list = []
    centroid_list = []
    zcr_list = []
    flatness_list = []
    ai_probs_list = []

    start_idx = 0
    seg_idx = 0

    while start_idx < total_samples:
        end_idx = min(start_idx + seg_samples, total_samples)
        seg_audio = signal[start_idx:end_idx]

        if seg_audio.size < 1024:
            break

        t_start = round(start_idx / sample_rate, 2)
        t_end = round(end_idx / sample_rate, 2)

        # Segment acoustic metrics
        seg_rms = float(np.sqrt(np.mean(seg_audio**2)))
        seg_centroid = float(np.mean(librosa.feature.spectral_centroid(y=seg_audio, sr=sample_rate)[0]))
        seg_zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=seg_audio)[0]))
        seg_flatness = float(np.mean(librosa.feature.spectral_flatness(y=seg_audio)[0]))

        rms_list.append(seg_rms)
        centroid_list.append(seg_centroid)
        zcr_list.append(seg_zcr)
        flatness_list.append(seg_flatness)

        # Optional segment authenticity classification
        seg_pred_label = None
        seg_ai_prob = None
        if classifier_model is not None and seg_audio.size >= 2048:
            try:
                feats = extract_model_features(seg_audio, sample_rate)
                pred_c = int(classifier_model.predict(feats)[0])
                seg_pred_label = "Real" if pred_c == 0 else "AI-generated"
                if hasattr(classifier_model, "predict_proba"):
                    probs = classifier_model.predict_proba(feats)[0]
                    # Class 1 is AI-generated
                    seg_ai_prob = round(float(probs[1]), 4)
                    ai_probs_list.append(seg_ai_prob)
            except Exception:
                pass

        segments.append({
            "segment_id": seg_idx + 1,
            "start_time": t_start,
            "end_time": t_end,
            "duration": round(t_end - t_start, 2),
            "rms_energy": round(seg_rms, 5),
            "spectral_centroid": round(seg_centroid, 1),
            "zero_crossing_rate": round(seg_zcr, 4),
            "spectral_flatness": round(seg_flatness, 4),
            "predicted_authenticity": seg_pred_label,
            "ai_probability": seg_ai_prob,
            "is_anomaly": False,
            "anomaly_reason": "",
        })

        if end_idx >= total_samples:
            break

        start_idx += hop_samples
        seg_idx += 1

    # Anomaly Detection Method:
    # Uses modified Z-score against recording baseline to identify points of sudden acoustic transition
    # (e.g. sharp energy surge, extreme spectral jump, or abrupt silence).
    if len(segments) >= 3:
        rms_arr = np.array(rms_list)
        centroid_arr = np.array(centroid_list)

        rms_mean = np.mean(rms_arr)
        rms_std = np.std(rms_arr) + 1e-9
        centroid_mean = np.mean(centroid_arr)
        centroid_std = np.std(centroid_arr) + 1e-9

        # Sudden jump between consecutive segments
        for i in range(len(segments)):
            reasons = []
            # Check deviation from mean
            z_rms = abs(rms_arr[i] - rms_mean) / rms_std
            z_cent = abs(centroid_arr[i] - centroid_mean) / centroid_std

            if z_rms > 2.5:
                reasons.append(f"Significant energy divergence (Z={z_rms:.1f})")
            if z_cent > 2.5:
                reasons.append(f"Spectral centroid anomaly (Z={z_cent:.1f})")

            # Check consecutive delta (discontinuity)
            if i > 0:
                delta_rms = abs(rms_arr[i] - rms_arr[i-1]) / rms_std
                delta_cent = abs(centroid_arr[i] - centroid_arr[i-1]) / centroid_std
                if delta_rms > 3.0:
                    reasons.append("Abrupt energy step-change across boundary")
                if delta_cent > 3.0:
                    reasons.append("Sharp spectral timbre shift")

            if reasons:
                segments[i]["is_anomaly"] = True
                segments[i]["anomaly_reason"] = "; ".join(reasons)

    anomaly_count = sum(1 for s in segments if s["is_anomaly"])

    return {
        "total_segments": len(segments),
        "segment_duration_sec": segment_duration_sec,
        "overlap_ratio": overlap_ratio,
        "total_duration_sec": round(total_duration, 2),
        "segments": segments,
        "anomaly_count": anomaly_count,
        "timeline_data": {
            "start_times": [s["start_time"] for s in segments],
            "rms_energies": rms_list,
            "spectral_centroids": centroid_list,
            "zero_crossing_rates": zcr_list,
            "spectral_flatness": flatness_list,
            "ai_probabilities": ai_probs_list if ai_probs_list else None,
        },
        "anomaly_method": (
            "Z-Score Discontinuity Exploration: Measures segment feature deviations "
            "and inter-segment transition rates against overall recording variance."
        ),
        "disclaimer": (
            "Acoustic anomalies are technical points for investigation (such as ambient noise bursts, "
            "environmental transients, or abrupt mic movements). They do NOT necessarily indicate "
            "tampering, editing, or AI generation."
        ),
    }

