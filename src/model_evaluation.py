"""
Model Evaluation Laboratory for Acoustic Reality Intelligence.
Provides rigorous performance assessment, confusion matrix calculation,
per-class metrics, benchmark baseline reproduction, and CSV reporting.
"""
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import pandas as pd


# Benchmark baseline recorded from original training notebook (Acoustic_Reality_Intelligence.ipynb)
# Evaluated on 80-sample held-out test split with SVM (RBF kernel, 26 MFCC features)
BENCHMARK_BASELINE = {
    "source": "Original Training Notebook (Acoustic_Reality_Intelligence.ipynb)",
    "evaluation_type": "Held-Out Test Split (Random State = 42)",
    "total_samples": 80,
    "class_0_real_samples": 40,
    "class_1_ai_samples": 40,
    "accuracy": 0.8375,
    "precision_macro": 0.84,
    "recall_macro": 0.84,
    "f1_macro": 0.84,
    "confusion_matrix": {
        "true_negative_real": 34,
        "false_positive_ai": 6,
        "false_negative_real": 7,
        "true_positive_ai": 33
    },
    "per_class": {
        "Real (Class 0)": {
            "precision": 0.83,
            "recall": 0.85,
            "f1_score": 0.84,
            "support": 40
        },
        "AI-generated (Class 1)": {
            "precision": 0.85,
            "recall": 0.83,
            "f1_score": 0.84,
            "support": 40
        }
    },
    "note": (
        "This baseline reflects the specific 80-sample experimental split in the training notebook. "
        "It provides a reference point and should not be confused with performance on novel acoustic distributions."
    )
}


def compute_evaluation_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Computes complete statistical classification metrics:
    - Confusion Matrix (TN, FP, FN, TP)
    - Accuracy, Precision, Recall, Specificity, F1-Score
    - Per-class breakdown for Class 0 (Real) and Class 1 (AI-generated)
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)

    if y_t.size == 0 or y_p.size == 0:
        raise ValueError("Ground truth and prediction arrays must not be empty.")
    if y_t.shape != y_p.shape:
        raise ValueError("Ground truth and predictions must have identical dimensions.")

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    # Structure of binary cm with labels=[0, 1]:
    # row 0 (True=0, Real): cm[0,0]=TN, cm[0,1]=FP (falsely called AI)
    # row 1 (True=1, AI):   cm[1,0]=FN (falsely called Real), cm[1,1]=TP (correctly called AI)
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Per-class metrics
    real_prec = float(precision_score(y_t, y_p, pos_label=0, zero_division=0))
    real_rec = float(recall_score(y_t, y_p, pos_label=0, zero_division=0))
    real_f1 = float(f1_score(y_t, y_p, pos_label=0, zero_division=0))

    ai_prec = prec
    ai_rec = rec
    ai_f1 = f1

    return {
        "total_samples": int(len(y_t)),
        "accuracy": round(acc, 4),
        "accuracy_percent": round(acc * 100, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "true_negatives_real": tn,
            "false_positives_ai": fp,
            "false_negatives_real": fn,
            "true_positives_ai": tp,
            "raw_matrix": cm.tolist(),
        },
        "per_class": {
            "Real (Class 0)": {
                "precision": round(real_prec, 4),
                "recall": round(real_rec, 4),
                "f1_score": round(real_f1, 4),
                "support": int(np.sum(y_t == 0)),
            },
            "AI-generated (Class 1)": {
                "precision": round(ai_prec, 4),
                "recall": round(ai_rec, 4),
                "f1_score": round(ai_f1, 4),
                "support": int(np.sum(y_t == 1)),
            },
        },
    }


def generate_evaluation_dataframe(
    sample_names: List[str],
    y_true: List[int],
    y_pred: List[int],
    ai_probabilities: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Formats individual evaluation instances into an exportable Pandas DataFrame.
    """
    class_map = {0: "Real", 1: "AI-generated"}
    records = []
    for idx in range(len(sample_names)):
        yt = y_true[idx]
        yp = y_pred[idx]
        is_correct = bool(yt == yp)
        prob = ai_probabilities[idx] if ai_probabilities else None

        records.append({
            "Sample Name": sample_names[idx],
            "Ground Truth Class": yt,
            "Ground Truth Label": class_map.get(yt, "Unknown"),
            "Predicted Class": yp,
            "Predicted Label": class_map.get(yp, "Unknown"),
            "Is Correct": is_correct,
            "AI Probability": round(prob, 4) if prob is not None else "N/A",
        })

    return pd.DataFrame(records)

