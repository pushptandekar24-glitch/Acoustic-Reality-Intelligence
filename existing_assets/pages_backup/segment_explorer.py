"""
Segment Explorer Page for Acoustic Reality Intelligence.
Divides audio into time windows, visualizes temporal feature trajectories,
identifies abrupt acoustic anomalies, and provides segment-level authenticity evaluation.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import config
from src.ui_helpers import render_header, render_empty_state, setup_dark_matplotlib_figure
from src.segment_analysis import analyze_audio_segments
from src.synthetic_detection import SyntheticAudioDetector


def render_segment_explorer_page():
    render_header(
        title="Segment Explorer & Anomaly Detection",
        subtitle="Time-windowed temporal dissection, boundary transition analysis, and localized anomaly exploration.",
        badge_text="Temporal DSP"
    )

    signal = st.session_state.get("current_signal")
    sr = st.session_state.get("current_sr", config.TARGET_SAMPLE_RATE)
    filename = st.session_state.get("current_filename")

    if signal is None:
        render_empty_state(
            title="No Audio Loaded",
            description="Upload an audio recording to partition it into temporal segments and explore acoustic transition dynamics.",
            icon="✂"
        )
        return

    # User Controls Row
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1.2])
    with ctrl_col1:
        seg_duration = st.slider("Segment Window (seconds)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    with ctrl_col2:
        overlap = st.slider("Window Overlap Ratio", min_value=0.0, max_value=0.75, value=0.5, step=0.25)
    with ctrl_col3:
        enable_model = st.checkbox("Run SVM Classifier on Segments", value=True, help="Evaluates authenticity for each temporal window.")

    # Load Model if enabled
    model_inst = None
    if enable_model:
        try:
            detector = SyntheticAudioDetector()
            model_inst = detector.model
        except Exception:
            pass

    with st.spinner("Partitioning audio into windows and detecting acoustic transitions..."):
        seg_res = analyze_audio_segments(
            signal,
            sample_rate=sr,
            segment_duration_sec=seg_duration,
            overlap_ratio=overlap,
            classifier_model=model_inst
        )
        st.session_state.current_segments = seg_res

    # Summary Row
    total_segs = seg_res["total_segments"]
    anomalies = seg_res["anomaly_count"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="ari-card">
                <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">Total Windows</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #f3f6ff;">{total_segs}</div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">Window Size: {seg_duration:.1f}s</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="ari-card">
                <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">Acoustic Anomalies</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {'#ff6d8b' if anomalies > 0 else '#2bd49a'};">{anomalies}</div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">Discontinuities Detected</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="ari-card">
                <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">Detection Method</div>
                <div style="font-size: 1.0rem; font-weight: 700; color: #31d2ff;">Z-Score Shift</div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">Boundary Step-Change</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Disclaimer
    st.info(f"**Interpretation Guide:** {seg_res['disclaimer']}")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Timeline Charts
    st.markdown("### Temporal Feature Evolution")
    t_data = seg_res["timeline_data"]
    start_times = t_data["start_times"]

    if len(start_times) >= 2:
        num_subplots = 3 if t_data.get("ai_probabilities") else 2
        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 2.3 * num_subplots), dpi=130, sharex=True)
        fig.patch.set_facecolor("#101725")

        # RMS Energy Timeline
        ax_rms = axes[0]
        ax_rms.set_facecolor("#0a0f1d")
        ax_rms.plot(start_times, t_data["rms_energies"], color="#2bd49a", marker="o", markersize=3.5, linewidth=1.5)
        ax_rms.set_ylabel("RMS Energy", color="#9aa9c1", fontsize=8)
        ax_rms.set_title("Segment RMS Energy Dynamics", color="#f3f6ff", fontsize=9, fontweight="bold")
        ax_rms.tick_params(colors="#9aa9c1", labelsize=7.5)
        for s in ax_rms.spines.values(): s.set_color("#263248")

        # Spectral Centroid Timeline
        ax_cent = axes[1]
        ax_cent.set_facecolor("#0a0f1d")
        ax_cent.plot(start_times, t_data["spectral_centroids"], color="#9561ff", marker="s", markersize=3.5, linewidth=1.5)
        ax_cent.set_ylabel("Centroid (Hz)", color="#9aa9c1", fontsize=8)
        ax_cent.set_title("Spectral Centroid Trajectory (Timbral Brightness)", color="#f3f6ff", fontsize=9, fontweight="bold")
        ax_cent.tick_params(colors="#9aa9c1", labelsize=7.5)
        for s in ax_cent.spines.values(): s.set_color("#263248")

        # AI Probability Timeline (if enabled)
        if t_data.get("ai_probabilities"):
            ax_prob = axes[2]
            ax_prob.set_facecolor("#0a0f1d")
            ax_prob.plot(start_times[:len(t_data["ai_probabilities"])], [p * 100 for p in t_data["ai_probabilities"]],
                         color="#ff6d8b", marker="^", markersize=4, linewidth=1.5)
            ax_prob.axhline(50, color="#6b7c96", linestyle="--", linewidth=0.8, alpha=0.7)
            ax_prob.set_ylabel("AI Likelihood (%)", color="#9aa9c1", fontsize=8)
            ax_prob.set_xlabel("Timeline Position (seconds)", color="#9aa9c1", fontsize=8)
            ax_prob.set_ylim([-5, 105])
            ax_prob.set_title("Segment-Level AI Probability Estimate", color="#f3f6ff", fontsize=9, fontweight="bold")
            ax_prob.tick_params(colors="#9aa9c1", labelsize=7.5)
            for s in ax_prob.spines.values(): s.set_color("#263248")
        else:
            axes[-1].set_xlabel("Timeline Position (seconds)", color="#9aa9c1", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Detailed Table of Segments
    st.markdown("### Segment-by-Segment Dissection")
    table_rows = []
    for s in seg_res["segments"]:
        table_rows.append({
            "Segment ID": s["segment_id"],
            "Window": f"{s['start_time']}s - {s['end_time']}s",
            "RMS Energy": f"{s['rms_energy']:.5f}",
            "Spectral Centroid": f"{s['spectral_centroid']} Hz",
            "Zero Crossing Rate": f"{s['zero_crossing_rate']:.4f}",
            "Authenticity": s.get("predicted_authenticity") or "N/A",
            "AI Prob": f"{s['ai_probability'] * 100:.1f}%" if s.get("ai_probability") is not None else "N/A",
            "Anomaly Flag": "⚠️ Yes" if s["is_anomaly"] else "Normal",
            "Diagnostic Note": s["anomaly_reason"] if s["is_anomaly"] else "Stable continuum",
        })
    df_segs = pd.DataFrame(table_rows)
    st.dataframe(df_segs, use_container_width=True)

