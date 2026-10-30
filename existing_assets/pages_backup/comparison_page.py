"""
Compare Recordings Page for Acoustic Reality Intelligence.
Performs side-by-side comparative DSP evaluation of two audio tracks,
computing MFCC Cosine Similarity, DTW sequence distance, and acoustic delta statistics.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import config
from src.ui_helpers import render_header, render_empty_state, setup_dark_matplotlib_figure
from src.audio_processing import validate_audio_file, decode_audio_bytes, AudioProcessingError
from src.recording_comparison import compare_recordings


def render_comparison_page():
    render_header(
        title="Compare Recordings",
        subtitle="Side-by-side comparative analysis of two audio tracks across temporal, spectral, and MFCC domains.",
        badge_text="Comparative DSP"
    )

    st.markdown("### Select Two Audio Tracks for Comparison")
    col_a, col_b = st.columns(2, gap="medium")

    # Upload Slot A
    with col_a:
        st.markdown("#### Recording A (Reference)")
        upload_a = st.file_uploader(
            "Upload Track A",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            key="compare_upload_a"
        )
        # Fall back to currently active recording if user has one loaded and hasn't uploaded a new track A
        active_signal = st.session_state.get("current_signal")
        active_meta = st.session_state.get("current_meta")
        active_name = st.session_state.get("current_filename")

        sig_a, sr_a, meta_a, name_a = None, None, None, None

        if upload_a is not None:
            bytes_a = upload_a.getvalue()
            try:
                validate_audio_file(bytes_a, upload_a.name)
                sig_a, sr_a, meta_a = decode_audio_bytes(bytes_a)
                name_a = upload_a.name
                st.audio(bytes_a)
            except Exception as e:
                st.error(f"Error loading Track A: {e}")
        elif active_signal is not None:
            sig_a = active_signal
            sr_a = st.session_state.get("current_sr", config.TARGET_SAMPLE_RATE)
            meta_a = active_meta
            name_a = f"{active_name} (Active Recording)"
            if st.session_state.get("current_audio_bytes"):
                st.audio(st.session_state.get("current_audio_bytes"))
            st.caption(f"Using currently loaded recording: {active_name}")

    # Upload Slot B
    with col_b:
        st.markdown("#### Recording B (Candidate / Comparison)")
        upload_b = st.file_uploader(
            "Upload Track B",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            key="compare_upload_b"
        )
        sig_b, sr_b, meta_b, name_b = None, None, None, None

        if upload_b is not None:
            bytes_b = upload_b.getvalue()
            try:
                validate_audio_file(bytes_b, upload_b.name)
                sig_b, sr_b, meta_b = decode_audio_bytes(bytes_b)
                name_b = upload_b.name
                st.audio(bytes_b)
            except Exception as e:
                st.error(f"Error loading Track B: {e}")

    # Ensure both files are present
    if sig_a is None or sig_b is None:
        render_empty_state(
            title="Awaiting Both Audio Tracks",
            description="Upload both Recording A and Recording B above to generate comparative visualizations, similarity scores, and acoustic delta tables.",
            icon="⚖"
        )
        return

    # Run Comparison Engine
    with st.spinner("Executing comparative acoustic extraction and DTW alignment..."):
        comp_res = compare_recordings(
            signal_a=sig_a,
            sr_a=sr_a,
            meta_a=meta_a,
            signal_b=sig_b,
            sr_b=sr_b,
            meta_b=meta_b
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Similarity Score Card
    sim_pct = comp_res["similarity_score_percent"]
    cos_dist = comp_res["cosine_distance"]
    dtw_cost = comp_res["dtw_normalized_cost"]

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        st.markdown(
            f"""
            <div class="ari-card" style="border-color: #3878ff; text-align: center;">
                <div style="font-size: 0.75rem; color: #9aa9c1; text-transform: uppercase;">Acoustic Similarity</div>
                <div style="font-size: 2.1rem; font-weight: 700; color: #31d2ff;">{sim_pct:.1f}%</div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">Cosine MFCC Metric</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_sim2:
        st.markdown(
            f"""
            <div class="ari-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #9aa9c1; text-transform: uppercase;">Cosine Distance</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #f3f6ff;">{cos_dist:.4f}</div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">Feature Space Distance</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_sim3:
        st.markdown(
            f"""
            <div class="ari-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #9aa9c1; text-transform: uppercase;">DTW Sequence Cost</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #9561ff;">{dtw_cost:.4f}</div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">Temporal Alignment Cost</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Scientific Limitations Disclaimer
    st.info(f"**Interpretation Limits:** {comp_res['limitations']}")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Visual Comparison: Side-by-Side Waveforms and Spectrograms
    st.markdown("### Visual Spectral & Temporal Comparison")

    fig, axes = plt.subplots(2, 2, figsize=(10, 5.0), dpi=130)
    fig.patch.set_facecolor("#101725")

    times_a = np.linspace(0, len(sig_a) / sr_a, num=len(sig_a))
    times_b = np.linspace(0, len(sig_b) / sr_b, num=len(sig_b))

    # Waveform A
    axes[0, 0].set_facecolor("#0a0f1d")
    axes[0, 0].plot(times_a, sig_a, color="#31d2ff", linewidth=0.7)
    axes[0, 0].set_title(f"Track A Waveform: {name_a[:28]}", color="#f3f6ff", fontsize=8.5, fontweight="bold")
    axes[0, 0].tick_params(colors="#9aa9c1", labelsize=7)
    for s in axes[0, 0].spines.values(): s.set_color("#263248")

    # Waveform B
    axes[0, 1].set_facecolor("#0a0f1d")
    axes[0, 1].plot(times_b, sig_b, color="#ff6d8b", linewidth=0.7)
    axes[0, 1].set_title(f"Track B Waveform: {name_b[:28]}", color="#f3f6ff", fontsize=8.5, fontweight="bold")
    axes[0, 1].tick_params(colors="#9aa9c1", labelsize=7)
    for s in axes[0, 1].spines.values(): s.set_color("#263248")

    # Spectrogram A
    axes[1, 0].set_facecolor("#0a0f1d")
    axes[1, 0].imshow(comp_res["lab_a"]["stft_db"], aspect="auto", origin="lower",
                      extent=[0, times_a[-1], 0, sr_a / 2], cmap="magma")
    axes[1, 0].set_title("Track A Spectrogram (dB)", color="#f3f6ff", fontsize=8.5, fontweight="bold")
    axes[1, 0].set_xlabel("Time (s)", color="#9aa9c1", fontsize=7.5)
    axes[1, 0].tick_params(colors="#9aa9c1", labelsize=7)
    for s in axes[1, 0].spines.values(): s.set_color("#263248")

    # Spectrogram B
    axes[1, 1].set_facecolor("#0a0f1d")
    axes[1, 1].imshow(comp_res["lab_b"]["stft_db"], aspect="auto", origin="lower",
                      extent=[0, times_b[-1], 0, sr_b / 2], cmap="magma")
    axes[1, 1].set_title("Track B Spectrogram (dB)", color="#f3f6ff", fontsize=8.5, fontweight="bold")
    axes[1, 1].set_xlabel("Time (s)", color="#9aa9c1", fontsize=7.5)
    axes[1, 1].tick_params(colors="#9aa9c1", labelsize=7)
    for s in axes[1, 1].spines.values(): s.set_color("#263248")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Detailed Comparison Table
    st.markdown("### Acoustic Metrics Delta Table")
    import pandas as pd
    delta = comp_res["metrics_comparison"]
    delta_rows = [
        {
            "Acoustic Dimension": "Duration (seconds)",
            "Track A": f"{delta['duration_sec']['file_a']:.2f} s",
            "Track B": f"{delta['duration_sec']['file_b']:.2f} s",
            "Absolute Difference": f"{delta['duration_sec']['difference']:.2f} s"
        },
        {
            "Acoustic Dimension": "RMS Energy Mean",
            "Track A": f"{delta['rms_energy_mean']['file_a']:.5f}",
            "Track B": f"{delta['rms_energy_mean']['file_b']:.5f}",
            "Absolute Difference": f"{delta['rms_energy_mean']['difference']:.5f}"
        },
        {
            "Acoustic Dimension": "Spectral Centroid Mean",
            "Track A": f"{delta['spectral_centroid_mean']['file_a']} Hz",
            "Track B": f"{delta['spectral_centroid_mean']['file_b']} Hz",
            "Absolute Difference": f"{delta['spectral_centroid_mean']['difference']} Hz"
        },
        {
            "Acoustic Dimension": "Zero-Crossing Rate Mean",
            "Track A": f"{delta['zero_crossing_rate_mean']['file_a']:.4f}",
            "Track B": f"{delta['zero_crossing_rate_mean']['file_b']:.4f}",
            "Absolute Difference": f"{delta['zero_crossing_rate_mean']['difference']:.4f}"
        },
    ]
    st.dataframe(pd.DataFrame(delta_rows), use_container_width=True)

