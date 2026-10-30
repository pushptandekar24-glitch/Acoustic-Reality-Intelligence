"""
Tests for Model Evaluation module.
"""
import pytest
from src.model_evaluation import (
    compute_evaluation_metrics,
    BENCHMARK_BASELINE,
    generate_evaluation_dataframe
)


def test_compute_evaluation_metrics_exact():
    # 4 samples: [TN, FP, FN, TP]
    # Class 0: Real, Class 1: AI
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    metrics = compute_evaluation_metrics(y_true, y_pred)

    assert metrics["total_samples"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"]["true_negatives_real"] == 1
    assert metrics["confusion_matrix"]["false_positives_ai"] == 1
    assert metrics["confusion_matrix"]["false_negatives_real"] == 1
    assert metrics["confusion_matrix"]["true_positives_ai"] == 1

    per_class = metrics["per_class"]
    assert "Real (Class 0)" in per_class
    assert "AI-generated (Class 1)" in per_class


def test_benchmark_baseline_constants():
    base = BENCHMARK_BASELINE
    assert base["total_samples"] == 80
    assert base["accuracy"] == 0.8375
    cm = base["confusion_matrix"]
    total_cm = (
        cm["true_negative_real"] +
        cm["false_positive_ai"] +
        cm["false_negative_real"] +
        cm["true_positive_ai"]
    )
    assert total_cm == 80


def test_generate_evaluation_dataframe():
    names = ["sample_1.wav", "sample_2.wav"]
    y_t = [0, 1]
    y_p = [0, 1]
    df = generate_evaluation_dataframe(names, y_t, y_p)
    assert len(df) == 2
    assert "Is Correct" in df.columns
    assert df["Is Correct"].all()

