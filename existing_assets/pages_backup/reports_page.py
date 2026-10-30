"""
Reports Page for Acoustic Reality Intelligence.
Generates and downloads formal forensic PDF reports and CSV feature exports.
"""
import streamlit as st
import io

import config
from src.ui_helpers import render_header, render_empty_state
from src.report_generator import generate_pdf_report, generate_features_csv
from src.synthetic_detection import extract_model_features


def render_reports_page():
    render_header(
        title="Technical Reports & Data Export",
        subtitle="Forensic documentation generator, PDF technical reports, and raw acoustic feature export.",
        badge_text="Report Generator"
    )

    signal = st.session_state.get("current_signal")
    sr = st.session_state.get("current_sr", config.TARGET_SAMPLE_RATE)
    filename = st.session_state.get("current_filename")
    meta = st.session_state.get("current_meta")

    if signal is None or meta is None:
        render_empty_state(
            title="No Active Analysis Available",
            description="Please upload and analyze an audio recording in the Overview page before generating downloadable PDF and CSV reports.",
            icon="📑"
        )
        return

    auth_res = st.session_state.get("current_auth")
    sound_res = st.session_state.get("current_sound")
    lab_res = st.session_state.get("current_lab")
    seg_res = st.session_state.get("current_segments")

    st.markdown("### Report Summary & Included Modules")
    st.caption(f"Active Recording: **{filename}** · Duration: **{meta.get('duration_seconds', 0.0):.2f}s**")

    # Module Status Checkers
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="ari-card">
                <span class="ari-badge {'real' if auth_res else 'neutral'}">
                    {'✓ Included' if auth_res else 'Pending'}
                </span>
                <div style="font-weight: 700; margin-top: 6px;">Authenticity SVM</div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">{auth_res['headline'] if auth_res else 'Not evaluated'}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="ari-card">
                <span class="ari-badge {'real' if sound_res else 'neutral'}">
                    {'✓ Included' if sound_res else 'Pending'}
                </span>
                <div style="font-weight: 700; margin-top: 6px;">Environmental Scene</div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">{sound_res['top_event'] if sound_res else 'Not classified'}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="ari-card">
                <span class="ari-badge {'real' if lab_res else 'neutral'}">
                    {'✓ Included' if lab_res else 'Pending'}
                </span>
                <div style="font-weight: 700; margin-top: 6px;">Acoustic Laboratory</div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">{len(lab_res['summary_stats']) if lab_res else 0} DSP metrics</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="ari-card">
                <span class="ari-badge {'real' if seg_res else 'neutral'}">
                    {'✓ Included' if seg_res else 'Optional'}
                </span>
                <div style="font-weight: 700; margin-top: 6px;">Segment Explorer</div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">{seg_res['total_segments'] if seg_res else 0} windows</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Export Action Buttons
    down_col1, down_col2 = st.columns(2, gap="medium")

    with down_col1:
        st.markdown(
            """
            <div class="ari-card" style="border-color: #3878ff; padding: 20px;">
                <h4 style="color: #31d2ff; margin-bottom: 6px;">📄 Complete PDF Technical Report</h4>
                <p style="font-size: 0.82rem; margin-bottom: 16px;">
                    Generates a formal, printable document complete with metadata tables, embedded waveform and spectrogram charts,
                    synthetic sound assessment results, environmental sound rankings, model SHA-256 fingerprint, and technical disclaimers.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        try:
            with st.spinner("Compiling PDF report with embedded spectral figures..."):
                pdf_bytes = generate_pdf_report(
                    filename=filename,
                    metadata=meta,
                    lab_features=lab_res,
                    authenticity_result=auth_res,
                    sound_recognition_result=sound_res,
                    segment_result=seg_res,
                    signal=signal,
                    sample_rate=sr
                )

            clean_name = filename.rsplit('.', 1)[0]
            st.download_button(
                label="⬇ Download PDF Technical Report",
                data=pdf_bytes,
                file_name=f"{clean_name}_Acoustic_Intelligence_Report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"PDF compilation error: {e}")

    with down_col2:
        st.markdown(
            """
            <div class="ari-card" style="border-color: #2bd49a; padding: 20px;">
                <h4 style="color: #2bd49a; margin-bottom: 6px;">📊 26-MFCC Acoustic Features (CSV)</h4>
                <p style="font-size: 0.82rem; margin-bottom: 16px;">
                    Exports structured numerical feature columns matching the exact input format expected by the Support Vector Classifier:
                    13 MFCC means and 13 MFCC standard deviations along with audio metadata.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        try:
            feats = extract_model_features(signal, sr)
            csv_str = generate_features_csv(feats, meta, filename)
            clean_name = filename.rsplit('.', 1)[0]

            st.download_button(
                label="⬇ Download Structured Features (CSV)",
                data=csv_str.encode("utf-8"),
                file_name=f"{clean_name}_features_26mfcc.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"CSV compilation error: {e}")

