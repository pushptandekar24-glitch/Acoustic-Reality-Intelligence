"""
Audio Laboratory Page for Acoustic Reality Intelligence.
Provides deep visual DSP inspection: Waveforms, Spectrograms, Mel-Spectrograms,
MFCC representations, temporal curves, and comprehensive acoustic statistics.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

import config
from src.ui_helpers import render_header, render_empty_state, setup_dark_matplotlib_figure
from src.acoustic_features import compute_audio_laboratory_features


def render_audio_lab_page():
    render_header(
        title="Audio Laboratory",
        subtitle="Spectral decomposition, psychoacoustic feature analysis, and temporal DSP dynamics.",
        badge_text="Acoustic Signal Processing"
    )

    signal = st.session_state.get("current_signal")
    sr = st.session_state.get("current_sr", config.TARGET_SAMPLE_RATE)
    filename = st.session_state.get("current_filename")
    audio_bytes = st.session_state.get("current_audio_bytes")

    if signal is None:
        render_empty_state(
            title="No Active Recording Selected",
            description="Please upload an audio recording in the Overview page to explore its physical acoustic properties and spectral representations here.",
            icon="🔬"
        )
        return

    # Technical Disclaimer Banner
    st.info(
        "**Technical Note:** Acoustic descriptors quantify physical signal attributes (timbral shape, frequency distribution, and energy dynamics). "
        "They provide forensic insight into sound properties, but no individual acoustic metric proves whether a sound is natural or AI-generated."
    )

    # Audio Playback & File Header
    meta = st.session_state.get("current_meta", {})
    st.markdown(
        f"""
        <div class="ari-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; font-size: 1.1rem; color: #f3f6ff;">{filename}</span>
                <span class="ari-badge neutral">{meta.get('duration_seconds', 0.0):.2f}s · {sr} Hz</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if audio_bytes:
        st.audio(audio_bytes)

    # Compute/Retrieve Lab Features
    lab = st.session_state.get("current_lab")
    if lab is None:
        with st.spinner("Computing laboratory acoustic metrics..."):
            lab = compute_audio_laboratory_features(signal, sr)
            st.session_state.current_lab = lab

    # Key Statistics Cards
    stats = lab["summary_stats"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="ari-card">
                <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">Peak Amplitude</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #31d2ff;">{meta.get('peak_amplitude', 0.0):.3f}</div>
                <div style="font-size: 0.72rem; color: {'#ff6d8b' if meta.get('is_clipped') else '#2bd49a'};">
                    {'Clipping Detected' if meta.get('is_clipped') else 'Clean Dynamic Headroom'}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="ari-card">
                <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">RMS Energy Mean</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #2bd49a;">{stats['rms_energy']['mean']:.4f}</div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">Std: {stats['rms_energy']['std']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="ari-card">
                <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">Spectral Centroid</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #9561ff;">{stats['spectral_centroid']['mean']:.1f} <span style="font-size: 0.8rem;">Hz</span></div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">Center of Spectral Mass</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="ari-card">
                <div style="font-size: 0.72rem; color: #9aa9c1; text-transform: uppercase;">Dynamic Range</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #f3f6ff;">{stats.get('dynamic_range_db', 0.0):.1f} <span style="font-size: 0.8rem;">dB</span></div>
                <div style="font-size: 0.72rem; color: #9aa9c1;">Crest Factor: {stats.get('crest_factor', 0.0):.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Visualizations Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Waveform & Spectrogram",
        "Mel-Frequency Spectrogram",
        "MFCC Matrix (13 Bands)",
        "Spectral Descriptor Curves",
        "Statistical Summary Table"
    ])

    with tab1:
        st.markdown("#### Waveform & Short-Time Fourier Transform (STFT)")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.6), dpi=130)
        fig.patch.set_facecolor("#101725")

        # Waveform
        times = np.linspace(0, len(signal) / sr, num=len(signal))
        ax1.set_facecolor("#0a0f1d")
        ax1.plot(times, signal, color="#31d2ff", linewidth=0.7, alpha=0.95)
        ax1.set_title("Amplitude Waveform", color="#f3f6ff", fontsize=10, fontweight="bold", pad=5)
        ax1.set_ylabel("Amplitude", color="#9aa9c1", fontsize=8)
        ax1.set_xlim([0, times[-1]])
        ax1.tick_params(colors="#9aa9c1", labelsize=7.5)
        for s in ax1.spines.values():
            s.set_color("#263248")

        # STFT Spectrogram
        ax2.set_facecolor("#0a0f1d")
        im = ax2.imshow(
            lab["stft_db"],
            aspect="auto",
            origin="lower",
            extent=[0, times[-1], 0, sr / 2],
            cmap="magma"
        )
        ax2.set_title("Magnitude Spectrogram (dB)", color="#f3f6ff", fontsize=10, fontweight="bold", pad=5)
        ax2.set_xlabel("Time (seconds)", color="#9aa9c1", fontsize=8)
        ax2.set_ylabel("Frequency (Hz)", color="#9aa9c1", fontsize=8)
        ax2.tick_params(colors="#9aa9c1", labelsize=7.5)
        for s in ax2.spines.values():
            s.set_color("#263248")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.markdown("#### Perceptual Mel-Frequency Spectrogram (128 Mel Bins)")
        fig, ax = setup_dark_matplotlib_figure(figsize=(10, 3.8))
        im = ax.imshow(
            lab["mel_spec_db"],
            aspect="auto",
            origin="lower",
            extent=[0, len(signal) / sr, 0, 128],
            cmap="viridis"
        )
        ax.set_title("Mel Spectrogram (dB)", color="#f3f6ff", fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Time (seconds)", color="#9aa9c1", fontsize=8)
        ax.set_ylabel("Mel Frequency Bins", color="#9aa9c1", fontsize=8)
        fig.colorbar(im, ax=ax, label="Power (dB)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.markdown("#### MFCC Matrix (13 Coefficients across Time Frames)")
        st.caption("The foundation of the trained classifier: 13 Mel-Frequency Cepstral Coefficients summarized as 26 mean and standard deviation features.")
        fig, ax = setup_dark_matplotlib_figure(figsize=(10, 3.8))
        im = ax.imshow(
            lab["mfcc_matrix"],
            aspect="auto",
            origin="lower",
            extent=[0, len(signal) / sr, 1, 13],
            cmap="coolwarm"
        )
        ax.set_title("MFCC Coefficients 1 - 13", color="#f3f6ff", fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Time (seconds)", color="#9aa9c1", fontsize=8)
        ax.set_ylabel("MFCC Index", color="#9aa9c1", fontsize=8)
        fig.colorbar(im, ax=ax, label="Coefficient Amplitude")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        st.markdown("#### Temporal Feature Trajectories")
        times_frames = lab["times"]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6.2), dpi=130, sharex=True)
        fig.patch.set_facecolor("#101725")

        # RMS Energy
        ax1.set_facecolor("#0a0f1d")
        ax1.plot(times_frames, lab["rms_curve"], color="#2bd49a", linewidth=1.2)
        ax1.set_ylabel("RMS Energy", color="#9aa9c1", fontsize=8)
        ax1.tick_params(colors="#9aa9c1", labelsize=7.5)
        for s in ax1.spines.values(): s.set_color("#263248")

        # Spectral Centroid
        ax2.set_facecolor("#0a0f1d")
        ax2.plot(times_frames, lab["centroid_curve"], color="#9561ff", linewidth=1.2)
        ax2.set_ylabel("Centroid (Hz)", color="#9aa9c1", fontsize=8)
        ax2.tick_params(colors="#9aa9c1", labelsize=7.5)
        for s in ax2.spines.values(): s.set_color("#263248")

        # Zero-Crossing Rate
        ax3.set_facecolor("#0a0f1d")
        ax3.plot(times_frames, lab["zcr_curve"], color="#31d2ff", linewidth=1.2)
        ax3.set_ylabel("ZCR", color="#9aa9c1", fontsize=8)
        ax3.set_xlabel("Time (seconds)", color="#9aa9c1", fontsize=8)
        ax3.tick_params(colors="#9aa9c1", labelsize=7.5)
        for s in ax3.spines.values(): s.set_color("#263248")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab5:
        st.markdown("#### Comprehensive Acoustic Summary Table")
        import pandas as pd
        rows = []
        for feature_name, metrics in stats.items():
            if isinstance(metrics, dict):
                rows.append({
                    "Feature": feature_name.replace("_", " ").title(),
                    "Mean": f"{metrics['mean']:.5f}",
                    "Std Dev": f"{metrics['std']:.5f}",
                    "Min": f"{metrics['min']:.5f}",
                    "Max": f"{metrics['max']:.5f}",
                    "Median": f"{metrics['median']:.5f}",
                })
        df_stats = pd.DataFrame(rows)
        st.dataframe(df_stats, use_container_width=True)

