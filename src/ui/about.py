"""
About Page for Acoustic Reality Intelligence.
Provides architectural documentation, DSP mathematical formulation,
model verification provenance, voice-domain limitations, and academic viva defense guide.
"""
import streamlit as st

import config
from src.ui_helpers import render_header


def render_about_page():
    render_header(
        title="About Acoustic Reality Intelligence (ARI)",
        subtitle="System architecture, mathematical foundations, model provenance, and academic project specifications.",
        badge_text="College Mini-Project Documentation"
    )

    st.markdown(
        """
        <div class="ari-card" style="border-left: 4px solid #3878ff; padding: 18px 20px;">
            <h3 style="margin-top: 0; color: #f3f6ff; font-size: 1.15rem;">Project Identity & Academic Scope</h3>
            <p style="font-size: 0.88rem; line-height: 1.55; color: #d7e2f5; margin: 0;">
                <strong>Acoustic Reality Intelligence (ARI)</strong> is an academic college mini-project demonstrating 
                applied Digital Signal Processing (DSP) and Machine Learning for environmental audio inspection and experimental
                synthetic sound assessment. It establishes a multi-dimensional laboratory exploring spectral representations,
                MFCC distribution statistics, and machine learning classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Signal Processing & Model Pipeline",
        "Model Provenance & Checksum",
        "Voice Domain Scope & Environmental Distinction",
        "Academic Defense & Viva Guide"
    ])

    with tab1:
        st.markdown("### Digital Signal Processing (DSP) Pipeline")
        st.markdown(
            """
            The feature extraction architecture reproduces the verified training pipeline from the original research:
            1. **Standardized Audio Ingestion:** Recordings in WAV, MP3, FLAC, OGG, or M4A format are loaded and resampled to a standardized **16 kHz (16,000 Hz) mono** waveform.
            2. **Frame-Based Spectral Analysis:** An STFT frame window of 2048 samples with a 512-sample hop length decomposes the time-domain signal into spectral frames.
            3. **Mel-Frequency Cepstral Analysis (13 Coefficients):** 13 Mel-Frequency Cepstral Coefficients (MFCCs) are calculated across all frames.
            4. **Statistical Summarization (26 Dimensions):**
               For each coefficient \\( k \\in [1, 13] \\):
               - Temporal Mean: \\( \\mu_k = \\frac{1}{T} \\sum_{t=1}^T \\text{MFCC}_{k,t} \\)
               - Temporal Standard Deviation: \\( \\sigma_k = \\sqrt{\\frac{1}{T} \\sum_{t=1}^T (\\text{MFCC}_{k,t} - \\mu_k)^2} \\)
               The resulting feature vector \\( \\mathbf{x} = [\\mu_1, \\dots, \\mu_{13}, \\sigma_1, \\dots, \\sigma_{13}]^T \\in \\mathbb{R}^{26} \\) serves as model input.
            5. **Fitted Scaling & RBF Classification:** Features are normalized using a pre-fitted `StandardScaler` and evaluated by a Support Vector Classifier (`SVC`) with a Radial Basis Function (`RBF`) kernel.
            """
        )

    with tab2:
        st.markdown("### Cryptographic Model Provenance & Verified Class Mapping")
        st.markdown(
            f"""
            #### Model Integrity Verification
            To guarantee model authenticity and ensure zero silent model tampering, the model file is cryptographically validated at startup:
            - **Expected SHA-256 Fingerprint:**
              `b6d965c31ceba6a6be2483af2e378dead36f59a44ea97f5c3b3cf6f5a8160f91`
            - **Model Type:** Scikit-Learn `Pipeline(steps=[('standardscaler', StandardScaler()), ('svc', SVC(kernel='rbf', probability=True))])`

            #### Ground-Truth Class Mapping
            As verified directly in the original training notebook (`Acoustic_Reality_Intelligence.ipynb`):
            - `y.append(1 if label == "fake" else 0)`
            - `classification_report(labels=[0, 1], target_names=["Real", "AI-generated"])`
            
            Therefore:
            - **Numeric Class 0 = Real Audio** $\\rightarrow$ Headline: <span style="color: #2bd49a; font-weight: bold;">Your audio appears to be REAL</span>
            - **Numeric Class 1 = AI-Generated Audio** $\\rightarrow$ Headline: <span style="color: #ff6d8b; font-weight: bold;">Your audio appears to be AI-GENERATED</span>
            """
        )

    with tab3:
        st.markdown("### Voice-Specific Limitations & Environmental Distinction")
        st.markdown(
            """
            #### Voice Domain Clarification
            - **Not a Dedicated Speech Deepfake Detector:** The existing SVM model was trained on general/environmental audio recordings using 26 broad MFCC summary statistics.
              It is **not** an independently validated human-voice or AI voice-clone detector.
            - **Speech Complexity:** Cloned voices and modern speech synthesis algorithms (e.g. VITS, XTTS, ElevenLabs) produce nuanced vocal tract articulatory cues that require specialized phoneme-level acoustic architectures (e.g. RawNet, WavLM, Whisper representations). The SVM general-audio model may yield incorrect outputs on human speech or voice clips.

            #### Environmental Sound Distinction
            - **Acoustic Profiling vs. Trained Classifier:** Any environmental sound recognition heuristic implemented in the platform is a rule-based DSP acoustic profile (measuring spectral centroid, flatness, and sub-band energies). It is **not** a validated deep-learning environmental sound classifier like AudioSet YAMNet.
            - **Independence Principle:** Identifying recognizable real-world sounds (such as rain, bird song, or vehicle noise) does **NOT** prove that a recording is authentic, because modern generative AI can synthesize natural acoustic scenes with high fidelity.
            """
        )

    with tab4:
        st.markdown("### College Viva & Defense Guide")
        st.markdown(
            """
            Key technical talking points for academic presentation and viva evaluation:
            1. **Why 26 MFCC Features?** Mel-Frequency Cepstral Coefficients model the human ear's logarithmic frequency resolution (Mel scale). The first 13 coefficients capture broad spectral envelope and resonance. Computing both temporal mean and variance captures static spectral shape alongside dynamic range.
            2. **Why Support Vector Machines (SVM)?** SVM with an RBF kernel projects the 26-dimensional feature space into an infinite-dimensional Hilbert space, constructing an optimal maximum-margin hyperplane. This prevents the overfitting that commonly plagues deep neural networks when trained on compact tabular feature sets.
            3. **What Are the Real-World Limitations?** The 83.75% baseline test accuracy was measured on a specific 80-sample experimental split. Compressed audio (e.g. low-bitrate MP3/AAC) or poor microphone quality can introduce compression artifacts that distort higher-order MFCCs.
            4. **How Is Model Integrity Enforced?** The application computes a SHA-256 cryptographic hash of `acoustic_model.joblib` before every inference session. If a single byte differs from `b6d965c3...`, a fail-safe interlock prevents execution.
            """
        )

