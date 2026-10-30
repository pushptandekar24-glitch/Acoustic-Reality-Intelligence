"""
Environmental Sound Recognition Page for Acoustic Reality Intelligence.
Identifies physical sound event categories and displays ranked likelihood distributions.
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.ui_helpers import render_header, render_empty_state, setup_dark_matplotlib_figure
from src.sound_recognition import EnvironmentalSoundRecognizer


def render_sound_recognition_page():
    render_header(
        title="Environmental Sound Recognition",
        subtitle="Classification of ambient audio events using acoustic spectral dynamics and temporal signatures.",
        badge_text="Audio Scene Analysis"
    )

    signal = st.session_state.get("current_signal")
    sr = st.session_state.get("current_sr", config.TARGET_SAMPLE_RATE)
    filename = st.session_state.get("current_filename")

    if signal is None:
        render_empty_state(
            title="No Audio Loaded",
            description="Upload an audio recording to perform environmental event classification and sound scene analysis.",
            icon="🌿"
        )
        return

    # Critical Independence Disclaimer
    st.warning(
        "**Independent Analysis Disclaimer:** Environmental sound recognition classifies what real-world event is audible in the audio. "
        "The identification of natural sounds (such as rain, bird song, or vehicles) does **NOT** indicate whether a recording is authentic or AI-generated."
    )

    sound_res = st.session_state.get("current_sound")
    if sound_res is None:
        with st.spinner("Classifying environmental sound events..."):
            recognizer = EnvironmentalSoundRecognizer()
            sound_res = recognizer.classify_sound_events(signal, sr)
            st.session_state.current_sound = sound_res

    top_event = sound_res["top_event"]
    top_score = sound_res["top_score"]
    top_preds = sound_res["top_predictions"]
    descriptors = sound_res["acoustic_descriptors"]

    # Top Prediction Callout Card
    st.markdown(
        f"""
        <div class="ari-card" style="border-color: #9561ff; padding: 20px;">
            <div style="font-size: 0.75rem; color: #9aa9c1; text-transform: uppercase; letter-spacing: 0.08em;">Primary Sound Event Detected</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #f3f6ff; margin: 4px 0;">
                {top_event}
                <span class="ari-badge neutral" style="font-size: 0.85rem; vertical-align: middle; margin-left: 10px;">
                    {top_score:.1%} Confidence
                </span>
            </div>
            <div style="font-size: 0.82rem; color: #9aa9c1;">
                Analyzed from: <b>{filename}</b> · Engine: <b>{sound_res.get('backend')}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9], gap="medium")

    with col1:
        st.markdown("### Ranked Category Probabilities")
        events = [p["event"] for p in reversed(top_preds)]
        scores = [p["score"] * 100 for p in reversed(top_preds)]

        fig, ax = setup_dark_matplotlib_figure(figsize=(8, 4.2))
        colors = ["#3878ff" if i < len(events) - 1 else "#9561ff" for i in range(len(events))]
        bars = ax.barh(events, scores, color=colors, height=0.55, edgecolor="#263248")

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 1.2,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                color="#f3f6ff",
                fontsize=8,
                fontweight="bold"
            )

        ax.set_xlim([0, max(scores) + 14])
        ax.set_xlabel("Likelihood Score (%)", color="#9aa9c1", fontsize=8)
        ax.set_title("Top Environmental Event Categories", color="#f3f6ff", fontsize=10, fontweight="bold", pad=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("### Physical Acoustic Descriptors")
        st.caption("Underlying DSP parameters utilized to compute the environmental profile.")

        st.markdown(
            f"""
            <div class="ari-card">
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">Spectral Centroid:</span>
                    <strong style="color: #31d2ff;">{descriptors['spectral_centroid_hz']} Hz</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">Spectral Flatness (Noise index):</span>
                    <strong style="color: #2bd49a;">{descriptors['spectral_flatness']}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">Spectral Bandwidth:</span>
                    <strong style="color: #f3f6ff;">{descriptors['spectral_bandwidth_hz']} Hz</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">Zero-Crossing Rate:</span>
                    <strong style="color: #f3f6ff;">{descriptors['zero_crossing_rate']}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">Low-Band Energy (< 500Hz):</span>
                    <strong style="color: #9561ff;">{descriptors['low_band_energy_ratio']:.1%}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">Mid-Band Energy (500-3000Hz):</span>
                    <strong style="color: #9561ff;">{descriptors['mid_band_energy_ratio']:.1%}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">High-Band Energy (> 3000Hz):</span>
                    <strong style="color: #9561ff;">{descriptors['high_band_energy_ratio']:.1%}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(f"Backend details: {sound_res.get('yamnet_status')}")

