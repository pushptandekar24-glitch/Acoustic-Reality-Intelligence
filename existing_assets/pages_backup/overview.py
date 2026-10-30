"""
Overview Dashboard page for Acoustic Reality Intelligence.
Displays high-level system metrics, recent session activity, quick audio uploader, and module shortcuts.
"""
import streamlit as st
import datetime
from pathlib import Path

import config
from src.ui_helpers import render_header, render_empty_state
from src.audio_processing import validate_audio_file, decode_audio_bytes, AudioProcessingError
from src.acoustic_features import compute_audio_laboratory_features
from src.synthetic_detection import SyntheticAudioDetector
from src.sound_recognition import EnvironmentalSoundRecognizer


def render_overview_page():
    render_header(
        title="Intelligence Dashboard",
        subtitle="Environmental audio forensics, acoustic laboratory analysis, and synthetic sound assessment.",
        badge_text="Real-Time DSP Engine"
    )

    # Top Metrics Row
    history = st.session_state.get("session_history", [])
    total_analyzed = len(history)
    latest_item = history[-1] if history else None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="ari-card">
                <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">Session Analyzed</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #f3f6ff;">{total_analyzed}</div>
                <div style="font-size: 0.72rem; color: #31d2ff;">Active Workspace Session</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        latest_headline = latest_item.get("headline", "None yet") if latest_item else "No analyses yet"
        badge_cls = "real" if (latest_item and not latest_item.get("is_ai")) else ("ai" if latest_item else "neutral")
        st.markdown(
            f"""
            <div class="ari-card">
                <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">Latest Assessment</div>
                <div style="font-size: 1.05rem; font-weight: 700; margin: 4px 0;">
                    <span class="ari-badge {badge_cls}">{latest_headline}</span>
                </div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">{latest_item.get('filename', 'Waiting for input') if latest_item else 'Awaiting audio upload'}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        latest_event = latest_item.get("top_event", "None yet") if latest_item else "No recordings"
        st.markdown(
            f"""
            <div class="ari-card">
                <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">Acoustic Event</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #9561ff;">{latest_event}</div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">Environmental Profiler</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div class="ari-card">
                <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">SVM Verification</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #2bd49a;">Verified SHA-256</div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">26 MFCC Coefficients</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Main Split: Audio Upload & Analysis Trigger on left, Recent Activity on right
    left_col, right_col = st.columns([1.1, 0.9], gap="medium")

    with left_col:
        st.markdown("### Upload Audio Recording")
        st.caption("Upload an environmental audio clip (WAV, MP3, FLAC, OGG, M4A) for unified analysis.")

        uploaded_file = st.file_uploader(
            "Drop audio recording here",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            help="File is decoded and resampled to 16 kHz mono for ML processing while preserving original metrics.",
            key="overview_uploader"
        )

        # Quick Sample Audio Loader
        sample_dir = Path("sample_audio")
        if sample_dir.exists():
            st.markdown("<div style='font-size: 0.8rem; color: #9aa9c1; margin-top: 6px;'>Or test immediately with built-in demonstration audio:</div>", unsafe_allow_html=True)
            s_col1, s_col2, s_col3 = st.columns(3)
            sample_to_load = None
            with s_col1:
                if st.button("🌧 Rain & Thunder", use_container_width=True):
                    sample_to_load = sample_dir / "sample_rain_thunder.wav"
            with s_col2:
                if st.button("🐦 Birds & Insects", use_container_width=True):
                    sample_to_load = sample_dir / "sample_birds_insects.wav"
            with s_col3:
                if st.button("🚨 Siren & Alarm", use_container_width=True):
                    sample_to_load = sample_dir / "sample_siren_alarm.wav"

            if sample_to_load is not None and sample_to_load.exists():
                file_bytes = sample_to_load.read_bytes()
                try:
                    with st.spinner(f"Loading sample {sample_to_load.name}..."):
                        signal, sr, meta = decode_audio_bytes(file_bytes)
                        detector = SyntheticAudioDetector()
                        auth_res = detector.predict_audio(signal, sr)
                        recognizer = EnvironmentalSoundRecognizer()
                        sound_res = recognizer.classify_sound_events(signal, sr)
                        lab_res = compute_audio_laboratory_features(signal, sr)

                        st.session_state.current_audio_bytes = file_bytes
                        st.session_state.current_filename = sample_to_load.name
                        st.session_state.current_signal = signal
                        st.session_state.current_sr = sr
                        st.session_state.current_meta = meta
                        st.session_state.current_auth = auth_res
                        st.session_state.current_sound = sound_res
                        st.session_state.current_lab = lab_res

                        history_entry = {
                            "filename": sample_to_load.name,
                            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                            "duration": meta["duration_seconds"],
                            "headline": auth_res["headline"],
                            "class_name": auth_res["class_name"],
                            "is_ai": auth_res["is_ai_generated"],
                            "top_event": sound_res["top_event"],
                            "ai_prob": auth_res["ai_probability"],
                        }
                        st.session_state.session_history.append(history_entry)
                        st.success(f"Loaded sample: **{sample_to_load.name}**")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to load sample: {e}")

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            # If new file or re-uploaded, process it
            if st.session_state.get("current_filename") != uploaded_file.name:
                try:
                    with st.spinner("Decoding audio and running acoustic analysis..."):
                        validate_audio_file(file_bytes, uploaded_file.name)
                        signal, sr, meta = decode_audio_bytes(file_bytes)

                        # Run detectors
                        detector = SyntheticAudioDetector()
                        auth_res = detector.predict_audio(signal, sr)

                        recognizer = EnvironmentalSoundRecognizer()
                        sound_res = recognizer.classify_sound_events(signal, sr)

                        lab_res = compute_audio_laboratory_features(signal, sr)

                        # Store in session state
                        st.session_state.current_audio_bytes = file_bytes
                        st.session_state.current_filename = uploaded_file.name
                        st.session_state.current_signal = signal
                        st.session_state.current_sr = sr
                        st.session_state.current_meta = meta
                        st.session_state.current_auth = auth_res
                        st.session_state.current_sound = sound_res
                        st.session_state.current_lab = lab_res

                        # Record in history if not already present
                        history_entry = {
                            "filename": uploaded_file.name,
                            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                            "duration": meta["duration_seconds"],
                            "headline": auth_res["headline"],
                            "class_name": auth_res["class_name"],
                            "is_ai": auth_res["is_ai_generated"],
                            "top_event": sound_res["top_event"],
                            "ai_prob": auth_res["ai_probability"],
                        }
                        st.session_state.session_history.append(history_entry)

                        st.success(f"Successfully processed **{uploaded_file.name}**")
                        st.rerun()

                except AudioProcessingError as e:
                    st.error(f"Validation Error: {e}")
                except Exception as e:
                    st.error(f"Processing failed: {e}")

        # Active File Preview Card
        if st.session_state.get("current_signal") is not None:
            meta = st.session_state.current_meta
            auth = st.session_state.current_auth
            sound = st.session_state.current_sound

            st.markdown(
                f"""
                <div class="ari-card" style="border-color: #3878ff;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; font-size: 1.05rem;">{st.session_state.current_filename}</span>
                        <span class="ari-badge {'ai' if auth['is_ai_generated'] else 'real'}">{auth['class_name']}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #9aa9c1; margin-top: 4px;">
                        Duration: {meta['duration_seconds']:.2f}s · Sample Rate: {meta['sample_rate']} Hz · Channels: Mono
                    </div>
                    <div style="margin-top: 10px; font-size: 0.88rem;">
                        <b>Detected Sound Event:</b> <span style="color: #31d2ff;">{sound['top_event']}</span> ({sound['top_score']:.1%})
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.audio(st.session_state.current_audio_bytes)

    with right_col:
        st.markdown("### Recent Analyses")
        st.caption("All recordings evaluated in the current interactive session.")

        if not history:
            render_empty_state(
                title="No Session History",
                description="Upload an environmental audio recording to see live analysis results and session tracking here.",
                icon="📊"
            )
        else:
            for item in reversed(history[-5:]):
                badge_type = "ai" if item["is_ai"] else "real"
                st.markdown(
                    f"""
                    <div class="ari-card" style="padding: 10px 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <strong style="color: #f3f6ff; font-size: 0.9rem;">{item['filename']}</strong>
                            <span style="font-size: 0.72rem; color: #9aa9c1;">{item['timestamp']}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                            <span style="font-size: 0.8rem; color: #9561ff;">♫ {item['top_event']}</span>
                            <span class="ari-badge {badge_type}">{item['class_name']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Module Directory Navigation Grid
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Explore Intelligence Modules")
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown(
            """
            <div class="ari-card">
                <h4 style="color: #31d2ff; margin-bottom: 4px;">🔬 Audio Laboratory</h4>
                <p style="font-size: 0.8rem;">Waveforms, STFT spectrograms, Mel-spectrograms, MFCC matrices, and spectral descriptors.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="ari-card">
                <h4 style="color: #2bd49a; margin-bottom: 4px;">🛡 Authenticity Analysis</h4>
                <p style="font-size: 0.8rem;">Evaluate synthetic audio signatures using the verified 26-MFCC SVM classifier.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col2:
        st.markdown(
            """
            <div class="ari-card">
                <h4 style="color: #9561ff; margin-bottom: 4px;">🌿 Sound Recognition</h4>
                <p style="font-size: 0.8rem;">Identify environmental sound categories like rain, bird calls, vehicles, sirens, and crowd sounds.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="ari-card">
                <h4 style="color: #3878ff; margin-bottom: 4px;">✂ Segment Explorer</h4>
                <p style="font-size: 0.8rem;">Partition recordings into time windows to detect sudden acoustic transitions and temporal anomalies.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col3:
        st.markdown(
            """
            <div class="ari-card">
                <h4 style="color: #ff6d8b; margin-bottom: 4px;">⚖ Compare Recordings</h4>
                <p style="font-size: 0.8rem;">Side-by-side comparative analysis of two audio tracks using Cosine and DTW distances.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="ari-card">
                <h4 style="color: #f3f6ff; margin-bottom: 4px;">📑 Reports & Evaluation</h4>
                <p style="font-size: 0.8rem;">Generate comprehensive PDF reports, export feature CSVs, and examine benchmark baseline metrics.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
