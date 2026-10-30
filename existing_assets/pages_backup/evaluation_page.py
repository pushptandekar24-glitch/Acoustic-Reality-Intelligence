"""
Model Evaluation Laboratory Page for Acoustic Reality Intelligence.
Displays verified benchmark baseline metrics from the training notebook,
provides ground-truth evaluation tools, confusion matrix heatmaps, and CSV report export.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import config
from src.ui_helpers import render_header, render_empty_state, setup_dark_matplotlib_figure
from src.model_evaluation import (
    BENCHMARK_BASELINE,
    compute_evaluation_metrics,
    generate_evaluation_dataframe
)
from src.synthetic_detection import SyntheticAudioDetector


def render_evaluation_page():
    render_header(
        title="Model Evaluation Laboratory",
        subtitle="Performance validation, confusion matrix decomposition, and empirical ground-truth testing.",
        badge_text="Statistical Evaluation"
    )

    tab_baseline, tab_custom = st.tabs([
        "Benchmark Baseline (Notebook Replication)",
        "Custom Dataset Ground-Truth Testing"
    ])

    with tab_baseline:
        st.markdown("### Original Training Notebook Evaluation (Reproduced)")
        st.caption("Documented experimental results from the original 80-sample test set (Acoustic_Reality_Intelligence.ipynb).")

        base = BENCHMARK_BASELINE
        # Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="ari-card">
                    <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">Test Accuracy</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #31d2ff;">{base['accuracy'] * 100:.2f}%</div>
                    <div style="font-size: 0.72rem; color: #9aa9c1;">80 Held-Out Samples</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div class="ari-card">
                    <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">Macro Precision</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #2bd49a;">{base['precision_macro']:.2f}</div>
                    <div style="font-size: 0.72rem; color: #9aa9c1;">Balanced across classes</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f"""
                <div class="ari-card">
                    <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">Macro Recall</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #9561ff;">{base['recall_macro']:.2f}</div>
                    <div style="font-size: 0.72rem; color: #9aa9c1;">True Positive Rate</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c4:
            st.markdown(
                f"""
                <div class="ari-card">
                    <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">Macro F1-Score</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #f3f6ff;">{base['f1_macro']:.2f}</div>
                    <div style="font-size: 0.72rem; color: #9aa9c1;">Harmonic Mean</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br/>", unsafe_allow_html=True)

        col_cm, col_perclass = st.columns([1, 1], gap="medium")

        with col_cm:
            st.markdown("#### Baseline Confusion Matrix")
            # Draw Confusion Matrix Heatmap
            cm_dict = base["confusion_matrix"]
            cm_matrix = np.array([
                [cm_dict["true_negative_real"], cm_dict["false_positive_ai"]],
                [cm_dict["false_negative_real"], cm_dict["true_positive_ai"]]
            ])

            fig, ax = setup_dark_matplotlib_figure(figsize=(5.5, 4.0))
            im = ax.imshow(cm_matrix, cmap="Blues", interpolation="nearest")

            classes = ["Real (0)", "AI-gen (1)"]
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(classes, color="#f3f6ff", fontsize=9)
            ax.set_yticklabels(classes, color="#f3f6ff", fontsize=9)
            ax.set_xlabel("Predicted Class", color="#9aa9c1", fontsize=9, labelpad=6)
            ax.set_ylabel("Ground Truth Class", color="#9aa9c1", fontsize=9, labelpad=6)
            ax.set_title("Replicated Confusion Matrix (N = 80)", color="#f3f6ff", fontsize=10, fontweight="bold", pad=8)

            for i in range(2):
                for j in range(2):
                    val = cm_matrix[i, j]
                    text_color = "#ffffff" if val > 20 else "#080c16"
                    ax.text(j, i, str(val), ha="center", va="center", color=text_color, fontsize=12, fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_perclass:
            st.markdown("#### Per-Class Diagnostic Breakdown")
            per_class_rows = [
                {
                    "Class": "Real Sound (Class 0)",
                    "Precision": f"{base['per_class']['Real (Class 0)']['precision']:.2f}",
                    "Recall": f"{base['per_class']['Real (Class 0)']['recall']:.2f}",
                    "F1-Score": f"{base['per_class']['Real (Class 0)']['f1_score']:.2f}",
                    "Support": base['per_class']['Real (Class 0)']['support'],
                },
                {
                    "Class": "AI-Generated (Class 1)",
                    "Precision": f"{base['per_class']['AI-generated (Class 1)']['precision']:.2f}",
                    "Recall": f"{base['per_class']['AI-generated (Class 1)']['recall']:.2f}",
                    "F1-Score": f"{base['per_class']['AI-generated (Class 1)']['f1_score']:.2f}",
                    "Support": base['per_class']['AI-generated (Class 1)']['support'],
                }
            ]
            st.dataframe(pd.DataFrame(per_class_rows), use_container_width=True)

            st.markdown(
                f"""
                <div class="ari-card" style="margin-top: 14px; border-left: 3px solid #31d2ff;">
                    <div style="font-size: 0.75rem; color: #9aa9c1; line-height: 1.45;">
                        <b>Evaluation Provenance:</b> {base['note']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab_custom:
        st.markdown("### Evaluate Custom Ground-Truth Dataset")
        st.caption("Upload a CSV file containing ground truth labels (`y_true`: 0 for Real, 1 for AI-generated) to compute live metrics.")

        uploaded_csv = st.file_uploader(
            "Upload Evaluation CSV (`y_true`, and optionally `y_pred` or feature columns)",
            type=["csv"],
            key="custom_eval_uploader"
        )

        if uploaded_csv is not None:
            try:
                df_user = pd.read_csv(uploaded_csv)
                st.write("Uploaded dataset preview:", df_user.head(4))

                if "y_true" in df_user.columns:
                    y_true = df_user["y_true"].tolist()

                    if "y_pred" in df_user.columns:
                        y_pred = df_user["y_pred"].tolist()
                    else:
                        st.info("Computing predictions using trained SVM model...")
                        detector = SyntheticAudioDetector()
                        # Look for 26 feature columns or compute
                        feat_cols = [c for c in df_user.columns if "mfcc" in c.lower() or "feature" in c.lower()]
                        if len(feat_cols) == 26:
                            X = df_user[feat_cols].values
                            y_pred = detector.model.predict(X).tolist()
                        else:
                            st.error(f"Found {len(feat_cols)} feature columns. Expected 26 features.")
                            st.stop()

                    metrics = compute_evaluation_metrics(y_true, y_pred)
                    st.success(f"Evaluated {len(y_true)} samples · Live Accuracy: **{metrics['accuracy_percent']}%**")

                    # Export button
                    names = [f"Sample_{i+1}" for i in range(len(y_true))]
                    df_out = generate_evaluation_dataframe(names, y_true, y_pred)
                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Evaluation Results CSV",
                        data=csv_bytes,
                        file_name="model_evaluation_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Uploaded CSV must include a 'y_true' column with binary labels (0 for Real, 1 for AI).")
            except Exception as e:
                st.error(f"Failed to process evaluation dataset: {e}")
        else:
            render_empty_state(
                title="No Evaluation Dataset Uploaded",
                description="To benchmark new audio datasets against the trained model, upload a labeled CSV file containing ground-truth annotations.",
                icon="📊"
            )

