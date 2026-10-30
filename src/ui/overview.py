"""
Overview Page for Acoustic Reality Intelligence.
Provides project introduction, working audio uploader, playback, metadata inspection,
model verification status, and clear navigation to Voice / Audio Detection.
"""
import streamlit as st
from pathlib import Path

import config
from src.ui_helpers import render_header, render_empty_state
from src.audio_processing import validate_audio_file, decode_audio_bytes, AudioProcessingError
from src.synthetic_detection import SyntheticAudioDetector, find_model_and_metadata_paths, compute_file_sha256
from src.ui.sample_audio_view import render_sample_library_section


def render_overview_page():
    render_header(
        title="Acoustic Reality Intelligence (ARI)",
        subtitle="An AI-Powered Environmental Audio Analysis and Synthetic Sound Assessment System.",
        badge_text="Audio Intelligence v2.0"
    )

    # Top System Status Banner
    model_ok = False
    model_sha_short = "Unknown"
    try:
        model_path, _ = find_model_and_metadata_paths()
        actual_sha = compute_file_sha256(model_path)
        if actual_sha == config.EXPECTED_MODEL_SHA256:
            model_ok = True
            model_sha_short = actual_sha[:16] + "..."
    except Exception:
        model_ok = False

    st.markdown(
        f"""
        <div class="ari-card" style="border-left: 4px solid #3878ff; padding: 18px 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h3 style="margin: 0 0 4px; font-size: 1.15rem; color: #f3f6ff;">Platform Overview & Objective</h3>
                    <div style="font-size: 0.85rem; color: #9aa9c1; max-width: 780px; line-height: 1.5;">
                        Acoustic Reality Intelligence analyzes environmental sound recordings using digital signal processing (DSP)
                        and machine learning. Audio recordings are ingested, resampled to 16 kHz mono, and transformed into 
                        26 statistical Mel-Frequency Cepstral Coefficients (MFCCs) evaluated by an RBF Support Vector Classifier.
                    </div>
                </div>
                <div>
                    <span class="ari-badge {'real' if model_ok else 'ai'}" style="font-size: 0.78rem;">
                        {'✓ SVM SHA-256 Verified' if model_ok else '⚠ Checksum Mismatch'}
                    </span>
                    <div style="font-size: 0.68rem; color: #6b7c96; margin-top: 4px; font-family: monospace;">
                        {model_sha_short}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 0.9], gap="medium")

    with col_left:
        st.markdown("### 1. Select Audio Recording")
        st.caption("Upload an environmental audio clip (WAV, MP3, FLAC, OGG, M4A) or test with built-in samples.")

        uploaded_file = st.file_uploader(
            "Upload audio recording",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            help="Audio is validated and decoded to 16 kHz mono for model compatibility while preserving native metrics.",
            key="overview_uploader"
        )

        # Built-in demonstration sample buttons
        sample_dir = Path("sample_audio")
        if sample_dir.exists():
            st.markdown("<div style='font-size: 0.8rem; color: #9aa9c1; margin-top: 8px;'>Or load a built-in demonstration recording:</div>", unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            chosen_sample = None
            with s1:
                if st.button("🌧 Rain & Thunder", use_container_width=True):
                    chosen_sample = sample_dir / "sample_rain_thunder.wav"
            with s2:
                if st.button("🐦 Birds & Insects", use_container_width=True):
                    chosen_sample = sample_dir / "sample_birds_insects.wav"
            with s3:
                if st.button("🚨 Siren & Alarm", use_container_width=True):
                    chosen_sample = sample_dir / "sample_siren_alarm.wav"

            if chosen_sample and chosen_sample.exists():
                file_bytes = chosen_sample.read_bytes()
                try:
                    with st.spinner(f"Loading {chosen_sample.name}..."):
                        validate_audio_file(file_bytes, chosen_sample.name)
                        signal, sr, meta = decode_audio_bytes(file_bytes)
                        st.session_state.current_audio_bytes = file_bytes
                        st.session_state.current_filename = chosen_sample.name
                        st.session_state.current_signal = signal
                        st.session_state.current_sr = sr
                        st.session_state.current_meta = meta
                        st.session_state.current_auth = None  # Reset prediction until user triggers detection
                        st.success(f"Loaded: **{chosen_sample.name}**")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading sample: {e}")

        if uploaded_file is not None and st.session_state.get("current_filename") != uploaded_file.name:
            file_bytes = uploaded_file.getvalue()
            try:
                with st.spinner("Validating and decoding audio recording..."):
                    validate_audio_file(file_bytes, uploaded_file.name)
                    signal, sr, meta = decode_audio_bytes(file_bytes)
                    st.session_state.current_audio_bytes = file_bytes
                    st.session_state.current_filename = uploaded_file.name
                    st.session_state.current_signal = signal
                    st.session_state.current_sr = sr
                    st.session_state.current_meta = meta
                    st.session_state.current_auth = None
                    st.success(f"Successfully loaded: **{uploaded_file.name}**")
                    st.rerun()
            except AudioProcessingError as e:
                st.error(f"Audio Validation Error: {e}")
            except Exception as e:
                st.error(f"Audio Decoding Error: {e}")

    with col_right:
        st.markdown("### 2. Audio Inspection & Next Steps")
        st.caption("Metadata and direct navigation to authenticity assessment.")

        active_sig = st.session_state.get("current_signal")
        active_name = st.session_state.get("current_filename")
        active_bytes = st.session_state.get("current_audio_bytes")
        active_meta = st.session_state.get("current_meta")

        if active_sig is None or active_meta is None:
            render_empty_state(
                title="No Recording Loaded",
                description="Upload an audio file or select a demonstration sample to inspect audio metadata and proceed to detection.",
                icon="🎧"
            )
        else:
            # Metadata Card
            st.markdown(
                f"""
                <div class="ari-card" style="border-color: #3878ff;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <strong style="font-size: 1.05rem; color: #f3f6ff;">{active_name}</strong>
                        <span class="ari-badge neutral">{active_meta.get('file_size_bytes', 0) / 1024:.1f} KB</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; font-size: 0.8rem;">
                        <div><span style="color: #9aa9c1;">Duration:</span> <strong>{active_meta['duration_seconds']:.2f} s</strong></div>
                        <div><span style="color: #9aa9c1;">Resampled SR:</span> <strong>16,000 Hz</strong></div>
                        <div><span style="color: #9aa9c1;">Channels:</span> <strong>Mono (1-Ch)</strong></div>
                        <div><span style="color: #9aa9c1;">Peak Amplitude:</span> <strong>{active_meta['peak_amplitude']:.3f}</strong></div>
                        <div><span style="color: #9aa9c1;">RMS Energy:</span> <strong>{active_meta['rms_energy']:.4f}</strong></div>
                        <div><span style="color: #9aa9c1;">Audio State:</span> 
                            <strong style="color: {'#ff6d8b' if active_meta.get('is_silent') else '#2bd49a'};">
                                {'Silent' if active_meta.get('is_silent') else 'Active Signal'}
                            </strong>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if active_bytes:
                st.audio(active_bytes)

            st.markdown("<br/>", unsafe_allow_html=True)

            # Direct Navigation to Detection Page
            if st.button("🚀 Proceed to Voice / Audio Detection ➔", type="primary", use_container_width=True):
                st.session_state.active_nav = "Voice / Audio Detection"
                st.rerun()

    # Built-in Sample Audio Showcase: Real vs AI-Generated
    render_sample_library_section()

    # Lower Section: Pipeline Information Cards
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Core Technical Specifications")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            """
            <div class="ari-card">
                <h4 style="color: #31d2ff; margin-bottom: 4px;">📡 16 kHz Mono Ingestion</h4>
                <p style="font-size: 0.82rem; line-height: 1.45;">
                    Unified standard ensuring consistent frequency bounds across varied file formats (WAV, MP3, FLAC, OGG, M4A).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            """
            <div class="ari-card">
                <h4 style="color: #2bd49a; margin-bottom: 4px;">🎛 26-MFCC Representations</h4>
                <p style="font-size: 0.82rem; line-height: 1.45;">
                    13 Mel-Frequency Cepstral Coefficients summarized as 13 temporal means and 13 standard deviations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            """
            <div class="ari-card">
                <h4 style="color: #9561ff; margin-bottom: 4px;">🛡 Verified RBF SVM</h4>
                <p style="font-size: 0.82rem; line-height: 1.45;">
                    Fitted StandardScaler and Support Vector Classifier with cryptographically validated SHA-256 fingerprint.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

