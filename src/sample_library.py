"""
Sample Audio Demonstration Library for Acoustic Reality Intelligence.
Manages sample manifest loading, audio validation, cached SVM inference,
ground-truth verification, and side-by-side comparison.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import config
from src.audio_processing import decode_audio_bytes, AudioProcessingError
from src.synthetic_detection import SyntheticAudioDetector, compute_file_sha256


# In-memory cache for analyzed audio files keyed by file SHA-256
_ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}


def load_sample_manifest(manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads and validates the sample audio manifest.
    Falls back to config.SAMPLE_MANIFEST_FILE if no path provided.
    """
    path = manifest_path or config.SAMPLE_MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"Sample manifest not found at: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse sample manifest JSON: {exc}") from exc

    if "categories" not in data:
        raise ValueError("Invalid sample manifest: 'categories' key missing.")

    return data


def resolve_sample_path(filename: Optional[str], sample_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Resolves the absolute path of a sample audio file.
    Returns None if filename is None or empty.
    """
    if not filename:
        return None
    dir_path = sample_dir or config.SAMPLE_AUDIO_DIR
    target = dir_path / filename
    return target if target.exists() else target


def verify_ground_truth(sample_info: Dict[str, Any], predicted_class: int) -> Dict[str, Any]:
    """
    Compares the experimental model's predicted class against the verified ground truth.
    Strictly verifies authenticity only if metadata confirms ground truth.
    """
    is_verified = bool(sample_info.get("is_verified", False))
    gt_class = sample_info.get("ground_truth_class")
    gt_label = sample_info.get("ground_truth_label", "Demo audio — authenticity label unverified")

    if is_verified and gt_class is not None:
        gt_class_int = int(gt_class)
        matches = bool(predicted_class == gt_class_int)
        return {
            "is_verified": True,
            "ground_truth_class": gt_class_int,
            "ground_truth_label": gt_label,
            "prediction_matches_ground_truth": matches,
            "status_code": "MATCH" if matches else "MISMATCH",
            "status_text": (
                f"✓ Prediction matches verified ground truth ({gt_label})"
                if matches
                else f"⚠ Prediction does not match verified ground truth ({gt_label})"
            ),
            "badge_class": "real" if matches else "ai",
        }

    return {
        "is_verified": False,
        "ground_truth_class": None,
        "ground_truth_label": gt_label,
        "prediction_matches_ground_truth": None,
        "status_code": "UNVERIFIED",
        "status_text": "Ground truth unverified — authenticity label not verified",
        "badge_class": "neutral",
    }


def analyze_sample_audio(
    sample_info: Dict[str, Any],
    sample_dir: Optional[Path] = None,
    detector: Optional[SyntheticAudioDetector] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Executes full pipeline analysis for a given sample dictionary.
    
    Returns structured analysis result including:
    - Status ('ready', 'missing', 'error')
    - Audio metrics (duration, sr, channels)
    - Prediction (predicted_class, headline, probabilities)
    - Model verification status
    - Ground-truth validation
    """
    filename = sample_info.get("filename")
    if not filename:
        return {
            "status": "missing",
            "sample_id": sample_info.get("sample_id", "unknown"),
            "filename": None,
            "error_message": (
                sample_info.get("notes")
                or "No audio file associated with this sample. Verified pair could not be obtained."
            ),
            "is_verified": False,
            "ground_truth_label": sample_info.get("ground_truth_label", "Missing sample"),
        }

    file_path = resolve_sample_path(filename, sample_dir)
    if not file_path or not file_path.exists():
        return {
            "status": "missing",
            "sample_id": sample_info.get("sample_id", "unknown"),
            "filename": filename,
            "file_path": str(file_path) if file_path else None,
            "error_message": f"Audio file not found on disk: {filename}",
            "is_verified": sample_info.get("is_verified", False),
            "ground_truth_label": sample_info.get("ground_truth_label", "Unknown"),
        }

    # Compute file hash for caching & integrity
    try:
        file_sha256 = compute_file_sha256(file_path)
    except Exception as exc:
        return {
            "status": "error",
            "sample_id": sample_info.get("sample_id", "unknown"),
            "filename": filename,
            "error_message": f"Failed to read audio file: {exc}",
        }

    cache_key = f"{file_sha256}_{sample_info.get('sample_id', filename)}"
    if use_cache and cache_key in _ANALYSIS_CACHE:
        return _ANALYSIS_CACHE[cache_key]

    # Decode audio bytes
    try:
        raw_bytes = file_path.read_bytes()
        signal, sr, meta = decode_audio_bytes(raw_bytes)
    except AudioProcessingError as exc:
        return {
            "status": "error",
            "sample_id": sample_info.get("sample_id", "unknown"),
            "filename": filename,
            "error_message": f"Audio processing error: {exc}",
        }
    except Exception as exc:
        return {
            "status": "error",
            "sample_id": sample_info.get("sample_id", "unknown"),
            "filename": filename,
            "error_message": f"Audio decoding failed: {exc}",
        }

    # Run SVM model inference
    det = detector or SyntheticAudioDetector()
    pred_res = det.predict_audio(signal, sr)

    pred_class = pred_res["prediction_class"]
    verification = verify_ground_truth(sample_info, pred_class)

    # Check model SHA256 integrity
    model_sha_ok = bool(pred_res.get("model_sha256") == config.EXPECTED_MODEL_SHA256)

    result = {
        "status": "ready",
        "sample_id": sample_info.get("sample_id", filename),
        "display_title": sample_info.get("display_title", filename),
        "filename": filename,
        "file_path": str(file_path),
        "file_sha256": file_sha256,
        "file_bytes": raw_bytes,
        "role": sample_info.get("role", "Audio Sample"),
        "source": sample_info.get("source", "Unknown"),
        "license": sample_info.get("license", "Unknown"),
        "generator": sample_info.get("generator", "Unknown"),
        "caption": sample_info.get("caption", ""),
        "notes": sample_info.get("notes", ""),
        # Signal metrics
        "duration_seconds": round(float(meta.get("duration_seconds", 0.0)), 2),
        "sample_rate": int(meta.get("sample_rate", sr)),
        "channels": meta.get("channels", 1),
        "channels_desc": "Mono (1-Ch)" if meta.get("channels", 1) == 1 else "Stereo (2-Ch)",
        "peak_amplitude": round(float(meta.get("peak_amplitude", 0.0)), 3),
        "rms_energy": round(float(meta.get("rms_energy", 0.0)), 4),
        "signal": signal,
        "meta": meta,
        # Model predictions
        "prediction_class": pred_class,
        "class_name": pred_res["class_name"],
        "headline": pred_res["headline"],
        "is_ai_generated": pred_res["is_ai_generated"],
        "real_probability": pred_res["real_probability"],
        "ai_probability": pred_res["ai_probability"],
        "probabilities": pred_res["probabilities"],
        "feature_vector": pred_res.get("feature_vector"),
        # Verification
        "model_sha256": pred_res.get("model_sha256"),
        "model_verified": model_sha_ok,
        "is_verified": verification["is_verified"],
        "ground_truth_class": verification["ground_truth_class"],
        "ground_truth_label": verification["ground_truth_label"],
        "prediction_matches_ground_truth": verification["prediction_matches_ground_truth"],
        "status_code": verification["status_code"],
        "status_text": verification["status_text"],
        "badge_class": verification["badge_class"],
        "disclaimer": pred_res.get("disclaimer"),
    }

    if use_cache:
        _ANALYSIS_CACHE[cache_key] = result

    return result


def compare_two_samples(
    sample_a_info: Dict[str, Any],
    sample_b_info: Dict[str, Any],
    sample_dir: Optional[Path] = None,
    detector: Optional[SyntheticAudioDetector] = None,
) -> Dict[str, Any]:
    """
    Executes comparative analysis between two audio samples.
    Builds side-by-side metric tables and verification comparisons.
    """
    det = detector or SyntheticAudioDetector()
    res_a = analyze_sample_audio(sample_a_info, sample_dir=sample_dir, detector=det)
    res_b = analyze_sample_audio(sample_b_info, sample_dir=sample_dir, detector=det)

    both_ready = bool(res_a.get("status") == "ready" and res_b.get("status") == "ready")

    # Build comparison rows for presentation
    rows = []
    if both_ready:
        rows = [
            {
                "Attribute": "File Name",
                "Sample A": res_a.get("filename"),
                "Sample B": res_b.get("filename"),
            },
            {
                "Attribute": "Category Role",
                "Sample A": res_a.get("role"),
                "Sample B": res_b.get("role"),
            },
            {
                "Attribute": "Audio Source",
                "Sample A": res_a.get("source"),
                "Sample B": res_b.get("source"),
            },
            {
                "Attribute": "Generator / Origin",
                "Sample A": res_a.get("generator"),
                "Sample B": res_b.get("generator"),
            },
            {
                "Attribute": "Duration",
                "Sample A": f"{res_a.get('duration_seconds', 0.0):.2f}s",
                "Sample B": f"{res_b.get('duration_seconds', 0.0):.2f}s",
            },
            {
                "Attribute": "Sample Rate & Channels",
                "Sample A": f"{res_a.get('sample_rate')} Hz · {res_a.get('channels_desc')}",
                "Sample B": f"{res_b.get('sample_rate')} Hz · {res_b.get('channels_desc')}",
            },
            {
                "Attribute": "Verified Ground Truth",
                "Sample A": res_a.get("ground_truth_label"),
                "Sample B": res_b.get("ground_truth_label"),
            },
            {
                "Attribute": "Model Prediction",
                "Sample A": f"Class {res_a.get('prediction_class')} ({res_a.get('class_name')})",
                "Sample B": f"Class {res_b.get('prediction_class')} ({res_b.get('class_name')})",
            },
            {
                "Attribute": "Headline",
                "Sample A": res_a.get("headline"),
                "Sample B": res_b.get("headline"),
            },
            {
                "Attribute": "Model P(AI-generated)",
                "Sample A": f"{res_a.get('ai_probability', 0.0) * 100:.1f}%",
                "Sample B": f"{res_b.get('ai_probability', 0.0) * 100:.1f}%",
            },
            {
                "Attribute": "Ground-Truth Verification",
                "Sample A": res_a.get("status_text"),
                "Sample B": res_b.get("status_text"),
            },
        ]

    return {
        "sample_a": res_a,
        "sample_b": res_b,
        "both_ready": both_ready,
        "comparison_rows": rows,
        "disclaimer": (
            "Comparative Note: The experimental classifier evaluates 26 MFCC summary features. "
            "It does not guarantee correct separation between real and AI-generated audio clips. "
            "Predictions may vary based on recording conditions, compression, and generative models."
        ),
    }


def get_comparison_sample_lists(manifest: Dict[str, Any], sample_dir: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Gathers available Real samples and available AI-generated/demo samples
    for dropdown selectors in the comparison section.
    """
    real_samples: List[Dict[str, Any]] = []
    ai_samples: List[Dict[str, Any]] = []

    categories = manifest.get("categories", {})
    for cat_id, cat_info in categories.items():
        samples = cat_info.get("samples", {})
        for key, s in samples.items():
            fname = s.get("filename")
            if not fname:
                continue
            path = resolve_sample_path(fname, sample_dir)
            if not path or not path.exists():
                continue

            entry = dict(s)
            entry["category_name"] = cat_info.get("name", cat_id)

            # Determine whether to group in Real or AI selector
            if entry.get("role") == "Real Audio" or entry.get("ground_truth_class") == 0:
                real_samples.append(entry)
            else:
                ai_samples.append(entry)

    return real_samples, ai_samples

