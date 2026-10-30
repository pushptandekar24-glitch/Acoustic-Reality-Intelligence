"""
Synthetic Audio Assessment Page for Acoustic Reality Intelligence.
Evaluates audio recordings using the verified 26-MFCC Support Vector Machine (SVM) pipeline.
Displays verified dynamic headline, probability estimates, model verification badges, and technical disclaimers.
"""
import streamlit as st
import numpy as np

import config
from src.ui_helpers import render_header, render_empty_state
from src.synthetic_detection import SyntheticAudioDetector, ModelVerificationError


def render_authenticity_page():
    render_header(
        title="Synthetic Audio Assessment",
        subtitle="Binary authenticity estimation using statistical MFCC representations and a trained SVM classifier.",
        badge_text="Machine Learning Verification"
    )

    signal = st.session_state.get("current_signal")
    sr = st.session_state.get("current_sr", config.TARGET_SAMPLE_RATE)
    filename = st.session_state.get("current_filename")

    if signal is None:
        render_empty_state(
            title="No Recording Loaded",
            description="Upload an audio recording in the Overview page to run the verified synthetic audio assessment model.",
            icon="🛡"
        )
        return

    # Model Verification and Prediction
    try:
        auth_res = st.session_state.get("current_auth")
        if auth_res is None:
            with st.spinner("Extracting 26 MFCC features and evaluating SVM model..."):
                detector = SyntheticAudioDetector()
                auth_res = detector.predict_audio(signal, sr)
                st.session_state.current_auth = auth_res

    except ModelVerificationError as mve:
        st.error(f"Model Safety Lock: {mve}")
        st.stop()
    except Exception as e:
        st.error(f"Inference failed safely: {e}")
        st.stop()

    is_ai = auth_res["is_ai_generated"]
    headline = auth_res["headline"]
    class_style = "ai" if is_ai else "real"
    real_pct = auth_res["real_probability"] * 100
    ai_pct = auth_res["ai_probability"] * 100

    # 1. Clear Result Panel (Exact requirement: ONE clear headline, green for Real, red/pink for AI)
    st.markdown(
        f"""
        <div class="headline-card {class_style}">
            <div class="headline-title">{headline}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9], gap="medium")

    with col1:
        st.markdown("### Model Probability Estimates")
        st.caption("Statistical likelihood computed by the RBF Support Vector Classifier (uncalibrated experimental estimate).")

        # Visual progress bars for Real vs AI-generated
        st.markdown(
            f"""
            <div class="ari-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-weight: 600; color: #2bd49a;">Real Audio Likelihood:</span>
                    <strong style="color: #2bd49a;">{real_pct:.1f}%</strong>
                </div>
                <div style="background: #1c273c; height: 10px; border-radius: 5px; overflow: hidden; margin-bottom: 16px;">
                    <div style="background: #2bd49a; width: {real_pct}%; height: 100%;"></div>
                </div>

                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-weight: 600; color: #ff6d8b;">AI-Generated Likelihood:</span>
                    <strong style="color: #ff6d8b;">{ai_pct:.1f}%</strong>
                </div>
                <div style="background: #1c273c; height: 10px; border-radius: 5px; overflow: hidden;">
                    <div style="background: #ff6d8b; width: {ai_pct}%; height: 100%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Technical Disclaimer
        st.markdown(
            f"""
            <div class="ari-card" style="border-left: 3px solid #3878ff; background: rgba(20, 29, 44, 0.7);">
                <div style="font-size: 0.78rem; font-weight: 700; color: #f3f6ff; margin-bottom: 4px;">
                    Technical Integrity & Interpretation Notice
                </div>
                <div style="font-size: 0.74rem; color: #9aa9c1; line-height: 1.45;">
                    {auth_res['disclaimer']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### Model Architecture & Integrity")
        st.caption("Verifiable parameters of the active machine learning pipeline.")

        meta = st.session_state.get("current_meta", {})
        st.markdown(
            f"""
            <div class="ari-card">
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">Classifier Algorithm:</span>
                    <strong style="color: #f3f6ff;">Support Vector Machine (SVC)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">Kernel Type:</span>
                    <strong style="color: #f3f6ff;">Radial Basis Function (RBF)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">Preprocessing Pipeline:</span>
                    <strong style="color: #31d2ff;">StandardScaler (Fitted)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">Feature Dimensions:</span>
                    <strong style="color: #f3f6ff;">26 (13 MFCC Means + 13 Stds)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">Input Audio Standard:</span>
                    <strong style="color: #f3f6ff;">16 kHz Mono</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">Verified Class 0:</span>
                    <strong style="color: #2bd49a;">Real Audio</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">Verified Class 1:</span>
                    <strong style="color: #ff6d8b;">AI-Generated Audio</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                    <span style="color: #9aa9c1; font-size: 0.82rem;">SHA-256 Checksum:</span>
                    <span style="color: #2bd49a; font-family: monospace; font-size: 0.72rem;">Verified Match</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

