"""
About Page for Acoustic Reality Intelligence.
Provides project documentation, mathematical and DSP methodology,
architecture breakdown, model verification specs, and academic defense guide.
"""
import streamlit as st

import config
from src.ui_helpers import render_header


def render_about_page():
    render_header(
        title="About Acoustic Reality Intelligence",
        subtitle="System architecture, mathematical foundations, model provenance, and academic project identity.",
        badge_text="College Mini-Project Documentation"
    )

    st.markdown(
        """
        <div class="ari-card" style="border-left: 4px solid #3878ff; padding: 20px;">
            <h3 style="margin-top: 0; color: #f3f6ff;">An AI-Powered Environmental Audio Analysis and Synthetic Sound Assessment System</h3>
            <p style="font-size: 0.9rem; line-height: 1.6; color: #d7e2f5;">
                <strong>Acoustic Reality Intelligence</strong> is an advanced audio intelligence platform designed to address the growing challenge of
                synthetic, generative, and edited environmental sound recordings. Unlike naive real-versus-fake detectors, this platform unifies
                rigorous <em>Digital Signal Processing (DSP)</em>, <em>Acoustic Psychoacoustics</em>, and <em>Machine Learning</em> to inspect audio signals across
                multiple orthogonal dimensions: spectral distribution, temporal continuity, environmental sound-event classification, and statistical authenticity estimation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Signal Processing Pipeline",
        "Model Provenance & Checksum",
        "Environmental Sound Engine",
        "Academic Defense Guide"
    ])

    with tab1:
        st.markdown("### Digital Signal Processing (DSP) Architecture")
        st.markdown(
            """
            The feature extraction pipeline strictly adheres to the verified training specifications of the original research:
            1. **Standardized Ingestion:** Audio recordings in WAV, MP3, FLAC, OGG, or M4A format are decoded and resampled to a standardized **16 kHz (16,000 Hz) mono** signal.
            2. **Short-Time Spectral Decomposition:** An STFT frame window of 2048 samples with a 512-sample hop length transforms time-domain amplitudes into time-frequency matrices.
            3. **Mel-Frequency Cepstral Analysis:** 13 Mel-Frequency Cepstral Coefficients (MFCCs) are computed across all temporal frames, capturing spectral envelopes and timbral resonance.
            4. **Feature Vector Formulation (26 Dimensions):**
               For each coefficient \\( k \\in [1, 13] \\):
               - \\( \\mu_k = \\frac{1}{T} \\sum_{t=1}^T \\text{MFCC}_{k,t} \\) (Temporal Mean)
               - \\( \\sigma_k = \\sqrt{\\frac{1}{T} \\sum_{t=1}^T (\\text{MFCC}_{k,t} - \\mu_k)^2} \\) (Temporal Standard Deviation)
               The resulting vector \\( \\mathbf{x} = [\\mu_1, \\dots, \\mu_{13}, \\sigma_1, \\dots, \\sigma_{13}]^T \\in \\mathbb{R}^{26} \\) serves as the model input.
            5. **Statistical Scaling:** Features are normalized using a pre-fitted `StandardScaler` \\( z = \\frac{x - \\mu_{\\text{train}}}{\\sigma_{\\text{train}}} \\).
            """
        )

    with tab2:
        st.markdown("### Model Provenance, Checksum & Verified Class Mapping")
        st.markdown(
            f"""
            #### Cryptographic Verification
            To guarantee model integrity and eliminate silent substitution or tampering, the model file is cryptographically validated at startup:
            - **Expected SHA-256 Fingerprint:**
              `b6d965c31ceba6a6be2483af2e378dead36f59a44ea97f5c3b3cf6f5a8160f91`
            - **Algorithm:** Support Vector Classifier (`SVC`) with Radial Basis Function (`RBF`) kernel and fitted `StandardScaler`.
            - **Hyperparameters:** `kernel='rbf'`, `class_weight='balanced'`, `probability=True`, `random_state=42`.

            #### Verified Ground-Truth Class Mapping
            As verified in the original training notebook (`Acoustic_Reality_Intelligence.ipynb`):
            - `y.append(1 if label == "fake" else 0)`
            - `classification_report(labels=[0, 1], target_names=["Real", "AI-generated"])`
            
            Therefore:
            - **Numeric Class 0 = Real Audio** \\(\\rightarrow\\) Headline: <span style="color: #2bd49a; font-weight: bold;">Your audio appears to be REAL</span>
            - **Numeric Class 1 = AI-Generated Audio** \\(\\rightarrow\\) Headline: <span style="color: #ff6d8b; font-weight: bold;">Your audio appears to be AI-GENERATED</span>
            """
        )

    with tab3:
        st.markdown("### Decoupling Environmental Event Recognition from Authenticity")
        st.markdown(
            """
            A foundational design principle of this platform is the **strict separation** between *what* sound is present and *how* it was generated:
            - **Environmental Sound Recognition** identifies sound classes (e.g. rain, thunder, sirens, birdsong, traffic) by analyzing physical acoustic properties:
              - Spectral Centroid (timbral brightness)
              - Spectral Flatness (tonality vs. broadband white/pink noise)
              - Sub-band Energy Ratios (low rumble vs. high sibilance)
              - Envelope Burstiness (continuous flow vs. periodic spikes)
            - **Important Scientific Distinction:** Modern generative AI (e.g. AudioLDM, ElevenLabs, Suno, Stable Audio) can easily generate highly convincing synthetic rain, birdsong, or vehicle sounds.
              Therefore, detecting the presence of a natural sound event is **NOT** evidence that the recording is genuine.
            """
        )

    with tab4:
        st.markdown("### College Viva & Defense Guide")
        st.markdown(
            """
            When demonstrating this mini-project to academic evaluators, emphasize these technical points:
            1. **Why 26 MFCC features?** MFCCs model the human auditory system's non-linear frequency perception (Mel scale). The first 13 coefficients capture broad spectral tilt and formant envelopes. Computing both mean and variance captures static spectral shape as well as temporal dynamic range.
            2. **Why Support Vector Machines (SVM)?** SVM with an RBF kernel finds an optimal maximum-margin hyperplane in high-dimensional Hilbert space, making it effective for structured tabular features with limited sample sizes where deep neural networks would severely overfit.
            3. **What are the limitations?** The 83.75% baseline was measured on a specific 80-sample test split. Audio compressed with aggressive MP3/AAC codecs or recorded through low-quality microphones may exhibit spectral artifacts that shift MFCC coefficients. The system provides an experimental estimation rather than legally admissible forensic proof.
            4. **What is Segment Analysis?** Generative audio stitching or splice editing often creates abrupt phase or spectral discontinuities at boundary transitions. Segment analysis allows investigators to pinpoint localized temporal anomalies rather than relying solely on global averages.
            """
        )

