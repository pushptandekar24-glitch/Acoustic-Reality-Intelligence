"""
Synthetic audio assessment engine.
Loads and verifies the trained SVM pipeline, validates SHA-256 checksum,
and executes inference using the verified class mapping.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.acoustic_features import extract_model_features


class ModelVerificationError(Exception):
    """Raised when model file, checksum, or metadata cannot be validated."""
    pass


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def find_model_and_metadata_paths() -> Tuple[Path, Path]:
    """Locates the model and metadata files from standard or fallback locations."""
    if config.MODEL_FILE.exists() and config.METADATA_FILE.exists():
        return config.MODEL_FILE, config.METADATA_FILE
    if config.ROOT_MODEL_FILE.exists() and config.ROOT_METADATA_FILE.exists():
        return config.ROOT_MODEL_FILE, config.ROOT_METADATA_FILE
    raise FileNotFoundError(
        f"Model files not found. Looked in {config.MODELS_DIR} and {config.BASE_DIR}."
    )


def verify_and_load_model() -> Tuple[Pipeline, Dict[str, Any], str]:
    """
    Verifies model file SHA-256 fingerprint against the verified reference,
    validates metadata consistency, and loads the model pipeline.

    Returns:
        (pipeline_model, metadata_dict, model_sha256)
    """
    model_path, metadata_path = find_model_and_metadata_paths()

    # 1. Check SHA-256 fingerprint
    actual_hash = compute_file_sha256(model_path)
    if actual_hash != config.EXPECTED_MODEL_SHA256:
        raise ModelVerificationError(
            f"Model SHA-256 mismatch!\nExpected: {config.EXPECTED_MODEL_SHA256}\nActual:   {actual_hash}"
        )

    # 2. Read and validate metadata
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelVerificationError(f"Failed to read metadata file: {exc}") from exc

    if metadata.get("model_sha256") != actual_hash:
        raise ModelVerificationError(
            f"Metadata model_sha256 ({metadata.get('model_sha256')}) does not match model hash ({actual_hash})."
        )

    # 3. Load model pipeline
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise ModelVerificationError(f"Failed to unpickle model: {exc}") from exc

    # 4. Validate model structure
    if not isinstance(model, Pipeline):
        raise ModelVerificationError(f"Expected a scikit-learn Pipeline, got {type(model)}.")

    classes = getattr(model, "classes_", None)
    if classes is None or not np.array_equal(classes, np.array([0, 1])):
        raise ModelVerificationError(f"Expected binary classes [0, 1], got {classes}.")

    n_features = getattr(model, "n_features_in_", None)
    if n_features != config.FEATURE_DIMENSION:
        raise ModelVerificationError(
            f"Expected {config.FEATURE_DIMENSION} input features, got {n_features}."
        )

    return model, metadata, actual_hash


class SyntheticAudioDetector:
    """
    High-level classifier wrapper managing verified inference, probability estimation,
    and verified headline formatting.
    """

    def __init__(self):
        self.model, self.metadata, self.model_hash = verify_and_load_model()
        self.classes = self.model.classes_

        # Verified mapping: Class 0 = Real, Class 1 = AI-generated
        self.class_names = {0: "Real", 1: "AI-generated"}
        self.headlines = config.VERIFIED_HEADLINES

    def predict_audio(
        self,
        signal: np.ndarray,
        sample_rate: int = config.TARGET_SAMPLE_RATE
    ) -> Dict[str, Any]:
        """
        Executes end-to-end synthetic audio assessment:
        1. Extracts 26-MFCC feature vector
        2. Scaler transforms & SVM predicts
        3. Formulates verified headline and probability breakdown
        """
        feature_vector = extract_model_features(signal, sample_rate)
        pred_class = int(self.model.predict(feature_vector)[0])

        probabilities: Dict[str, float] = {}
        ai_prob = 0.5
        real_prob = 0.5

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feature_vector)[0]
            # Map probabilities using model.classes_
            for idx, cls in enumerate(self.classes):
                cls_int = int(cls)
                name = self.class_names.get(cls_int, f"Class {cls_int}")
                p_val = float(probs[idx])
                probabilities[name] = round(p_val, 4)
                if cls_int == 1:
                    ai_prob = p_val
                elif cls_int == 0:
                    real_prob = p_val

        headline = self.headlines.get(pred_class, f"Prediction: Class {pred_class}")
        class_name = self.class_names.get(pred_class, "Unknown")
        is_ai = bool(pred_class == 1)

        return {
            "prediction_class": pred_class,
            "class_name": class_name,
            "headline": headline,
            "is_ai_generated": is_ai,
            "real_probability": round(real_prob, 4),
            "ai_probability": round(ai_prob, 4),
            "probabilities": probabilities,
            "feature_vector": feature_vector.flatten().tolist(),
            "model_sha256": self.model_hash,
            "disclaimer": (
                "Disclaimer: This prediction is an experimental machine learning estimate based on 26 MFCC statistics. "
                "It does not provide conclusive forensic proof of audio authenticity."
            ),
        }

