"""
Voice / Audio Detection Page for Acoustic Reality Intelligence.
Executes the Experimental Real vs AI-Generated Audio Assessment using the verified SVM pipeline.
Displays prominent voice-limitation notice, verified headlines, uncalibrated probability estimates,
model verification status, and exploratory DSP inspection tools.
"""
import streamlit as st
import numpy as np

import config
from src.ui_helpers import render_header, render_empty_state
from src.audio_processing import validate_audio_file, decode_audio_bytes, AudioProcessingError
from src.synthetic_detection import SyntheticAudioDetector, ModelVerificationError
from src.acoustic_features import compute_audio_laboratory_features
from src.segment_analysis import analyze_audio_segments
from src.report_generator import generate_pdf_report, generate_features_csv


VOICE_LIMITATION_NOTICE = (
    "This is an experimental general-audio authenticity assessment based on the project's existing SVM model. "
    "It is not an independently validated human-voice or AI voice-clone detector. "
    "Results may be incorrect, particularly for unfamiliar sound types, speech, cloned voices, or recordings that differ from the training data."
)


def render_detection_page():
    render_header(
        title="Voice / Audio Detection",
        subtitle="Experimental Real vs AI-Generated Audio Assessment using 26 MFCC statistics and trained SVM.",
        badge_text="Authenticity Assessment"
    )

    # Mandatory Voice Limitation Notice
    st.warning(f"**Notice on Scope & Limitations:**\n\n{VOICE_LIMITATION_NOTICE}")

    # Audio State Check
    signal = st.session_state.get("current_signal")
    sr = st.session_state.get("current_sr", config.TARGET_SAMPLE_RATE)
    filename = st.session_state.get("current_filename")
    audio_bytes = st.session_state.get("current_audio_bytes")
    meta = st.session_state.get("current_meta")

    # If no audio loaded from Overview, offer upload directly here
    if signal is None:
        st.markdown("### Upload Audio for Assessment")
        uploaded_file = st.file_uploader(
            "Drop audio recording here (WAV, MP3, FLAC, OGG, M4A)",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            key="detection_inline_uploader"
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            try:
                validate_audio_file(file_bytes, uploaded_file.name)
                signal, sr, meta = decode_audio_bytes(file_bytes)
                st.session_state.current_audio_bytes = file_bytes
                st.session_state.current_filename = uploaded_file.name
                st.session_state.current_signal = signal
                st.session_state.current_sr = sr
                st.session_state.current_meta = meta
                st.session_state.current_auth = None
                filename = uploaded_file.name
                audio_bytes = file_bytes
                st.rerun()
            except AudioProcessingError as e:
                st.error(f"Audio Validation Error: {e}")
                return
            except Exception as e:
                st.error(f"Audio Decoding Error: {e}")
                return
        else:
            render_empty_state(
                title="Awaiting Audio Input",
                description="Upload an audio recording above or load a sample on the Overview page to run the experimental authenticity model.",
                icon="🎙"
            )
            return

    # Audio Playback & Active File Info
    st.markdown(
        f"""
        <div class="ari-card" style="border-color: #263248; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-weight: 700; font-size: 1.05rem; color: #f3f6ff;">{filename}</span>
                    <div style="font-size: 0.78rem; color: #9aa9c1; margin-top: 2px;">
                        Duration: {meta.get('duration_seconds', 0.0):.2f}s · Resampled: 16 kHz Mono · 26 MFCC Pipeline
                    </div>
                </div>
                <div style="margin-top: 4px;">
                    <span class="ari-badge real" style="font-size: 0.72rem;">SVM SHA-256 Verified</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if audio_bytes:
        st.audio(audio_bytes)

    # Execute Model Prediction
    auth_res = st.session_state.get("current_auth")
    if auth_res is None:
        try:
            with st.spinner("Extracting 26 MFCC statistical features and evaluating SVM classifier..."):
                detector = SyntheticAudioDetector()
                auth_res = detector.predict_audio(signal, sr)
                st.session_state.current_auth = auth_res
        except ModelVerificationError as mve:
            st.error(f"Model Safety Interlock Error: {mve}")
            return
        except Exception as exc:
            st.error(f"Inference execution failed: {exc}")
            return

    # Verified Headline Display
    is_ai = auth_res["is_ai_generated"]
    headline = auth_res["headline"]
    class_style = "ai" if is_ai else "real"
    real_pct = auth_res["real_probability"] * 100
    ai_pct = auth_res["ai_probability"] * 100

    # Prominent Detection Headline Card
    st.markdown(
        f"""
        <div class="headline-card {class_style}">
            <div class="headline-title">{headline}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Probability & Model Details Columns
    col1, col2 = st.columns([1.1, 0.9], gap="medium")

    with col1:
        st.markdown("### Uncalibrated Model Probability Estimates")
        st.caption("Statistical estimation from the SVM decision function (uncalibrated experimental model estimate).")

        st.markdown(
            f"""
            <div class="ari-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-weight: 600; color: #2bd49a;">Real Audio Estimate:</span>
                    <strong style="color: #2bd49a;">{real_pct:.1f}%</strong>
                </div>
                <div style="background: #1c273c; height: 10px; border-radius: 5px; overflow: hidden; margin-bottom: 16px;">
                    <div style="background: #2bd49a; width: {real_pct}%; height: 100%;"></div>
                </div>

                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-weight: 600; color: #ff6d8b;">AI-Generated Estimate:</span>
                    <strong style="color: #ff6d8b;">{ai_pct:.1f}%</strong>
                </div>
                <div style="background: #1c273c; height: 10px; border-radius: 5px; overflow: hidden;">
                    <div style="background: #ff6d8b; width: {ai_pct}%; height: 100%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="ari-card" style="border-left: 3px solid #3878ff; background: rgba(20, 29, 44, 0.65);">
                <div style="font-size: 0.74rem; color: #9aa9c1; line-height: 1.45;">
                    <b>Assessment Interpretation:</b> {auth_res['disclaimer']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### Model Provenance & Verification")
        st.caption("Cryptographic and architectural specifications of the active classifier.")

        st.markdown(
            f"""
            <div class="ari-card">
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.8rem;">Feature Pipeline:</span>
                    <strong style="color: #f3f6ff;">26 MFCC Statistics (13 Mean + 13 Std)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.8rem;">Input Standard:</span>
                    <strong style="color: #f3f6ff;">16 kHz Mono</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.8rem;">Classifier Architecture:</span>
                    <strong style="color: #f3f6ff;">StandardScaler + SVC (RBF Kernel)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.8rem;">Verified Class 0:</span>
                    <strong style="color: #2bd49a;">Real Audio</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1c273c;">
                    <span style="color: #9aa9c1; font-size: 0.8rem;">Verified Class 1:</span>
                    <strong style="color: #ff6d8b;">AI-Generated Audio</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                    <span style="color: #9aa9c1; font-size: 0.8rem;">SHA-256 Checksum:</span>
                    <span style="color: #2bd49a; font-family: monospace; font-size: 0.72rem;">Verified Match</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Exploratory Auxiliary Tools in Collapsible Expanders
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Exploratory Forensic Tools (Optional)")

    # Expander 1: Acoustic Properties & Spectrogram
    with st.expander("🔬 Exploratory Acoustic Properties & Spectral Inspection", expanded=False):
        st.caption("Detailed physical DSP features describing the physical acoustic properties of the recording.")
        lab_res = st.session_state.get("current_lab")
        if lab_res is None:
            lab_res = compute_audio_laboratory_features(signal, sr)
            st.session_state.current_lab = lab_res

        stats = lab_res["summary_stats"]
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.metric("RMS Energy Mean", f"{stats['rms_energy']['mean']:.4f}")
        with ec2:
            st.metric("Spectral Centroid", f"{stats['spectral_centroid']['mean']:.1f} Hz")
        with ec3:
            st.metric("Zero-Crossing Rate", f"{stats['zero_crossing_rate']['mean']:.4f}")
        with ec4:
            st.metric("Dynamic Range", f"{stats.get('dynamic_range_db', 0.0):.1f} dB")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 2.8), dpi=120)
        fig.patch.set_facecolor("#101725")
        ax.set_facecolor("#0a0f1d")
        times = np.linspace(0, len(signal) / sr, num=len(signal))
        ax.plot(times, signal, color="#31d2ff", linewidth=0.7)
        ax.set_title("Amplitude Waveform", color="#f3f6ff", fontsize=9, fontweight="bold")
        ax.set_xlabel("Time (seconds)", color="#9aa9c1", fontsize=7.5)
        ax.set_ylabel("Amplitude", color="#9aa9c1", fontsize=7.5)
        ax.tick_params(colors="#9aa9c1", labelsize=7)
        for s in ax.spines.values(): s.set_color("#263248")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Expander 2: Segment-Level Analysis
    with st.expander("✂ Exploratory Segment-Level Timeline & Anomaly Exploration", expanded=False):
        st.caption("Partitions the recording into 2.0-second sliding windows to inspect temporal feature drift and boundary discontinuities.")
        seg_res = analyze_audio_segments(signal, sample_rate=sr, segment_duration_sec=2.0)
        st.write(f"Total Windows: **{seg_res['total_segments']}** · Flagged Discontinuities: **{seg_res['anomaly_count']}**")
        st.info(f"Note: {seg_res['disclaimer']}")

    # Expander 3: Export Technical Report (PDF & CSV)
    with st.expander("📑 Export Analysis Documentation (PDF & CSV)", expanded=False):
        st.caption("Generate printable technical PDF documentation or export structured 26-MFCC feature tables.")
        rc1, rc2 = st.columns(2)
        with rc1:
            try:
                pdf_bytes = generate_pdf_report(
                    filename=filename,
                    metadata=meta,
                    authenticity_result=auth_res,
                    signal=signal,
                    sample_rate=sr
                )
                clean_name = filename.rsplit('.', 1)[0]
                st.download_button(
                    label="⬇ Download Technical PDF Report",
                    data=pdf_bytes,
                    file_name=f"{clean_name}_technical_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF generation error: {e}")
        with rc2:
            try:
                from src.synthetic_detection import extract_model_features
                feats = extract_model_features(signal, sr)
                csv_str = generate_features_csv(feats, meta, filename)
                clean_name = filename.rsplit('.', 1)[0]
                st.download_button(
                    label="⬇ Download 26-MFCC Features (CSV)",
                    data=csv_str.encode("utf-8"),
                    file_name=f"{clean_name}_features_26mfcc.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"CSV export error: {e}")

