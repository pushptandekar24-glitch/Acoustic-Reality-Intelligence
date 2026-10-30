import html
import hashlib
import io
import json
from pathlib import Path

import joblib
import librosa
import numpy as np
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Acoustic Reality Intelligence",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --page: #080c16;
        --panel: #101725;
        --panel-light: #141d2e;
        --border: #263248;
        --text: #f3f6ff;
        --muted: #9aa9c1;
        --blue: #3878ff;
        --cyan: #31d2ff;
        --purple: #9561ff;
        --green: #2bd49a;
        --rose: #ff6d8b;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        color: var(--text);
        background:
            radial-gradient(ellipse at 75% -5%, rgba(52, 75, 181, .19), transparent 34%),
            radial-gradient(ellipse at 8% 2%, rgba(17, 146, 196, .11), transparent 24%),
            var(--page);
    }
    .block-container {
        max-width: 1370px;
        padding: .65rem 1.15rem 1.2rem;
    }
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text);
        letter-spacing: -.045em;
    }
    p, label, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }
    header[data-testid="stHeader"] {
        height: 0;
        background: transparent;
    }
    [data-testid="stToolbar"] { display: none; }

    .navbar {
        min-height: 48px;
        padding: 6px 13px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        border: 1px solid #222d42;
        border-radius: 12px;
        background: linear-gradient(110deg, rgba(17, 25, 41, .98), rgba(11, 17, 29, .98));
        box-shadow: 0 8px 26px rgba(0, 0, 0, .22);
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #f4f7ff;
        font-weight: 700;
        font-size: .9rem;
        white-space: nowrap;
    }
    .brand-icon {
        width: 30px;
        height: 30px;
        display: grid;
        place-items: center;
        border: 1px solid rgba(81, 137, 255, .48);
        border-radius: 9px;
        background: linear-gradient(145deg, rgba(77, 102, 255, .26), rgba(25, 191, 255, .14));
        box-shadow: 0 0 18px rgba(45, 136, 255, .17);
        color: #73dfff;
        font-size: 15px;
    }
    .nav-items { display: flex; align-items: center; gap: 5px; }
    .nav-items a {
        color: #aab6cb;
        text-decoration: none;
        font-size: .78rem;
        font-weight: 600;
        padding: 8px 10px;
        border-radius: 8px;
        white-space: nowrap;
        transition: .18s ease;
    }
    .nav-items a:hover { color: white; background: #19243a; }
    .nav-items a.current {
        color: #fff;
        background: linear-gradient(110deg, #275de1, #347df4);
        box-shadow: 0 5px 16px rgba(38, 101, 238, .3);
    }

    .hero {
        position: relative;
        height: 226px;
        overflow: hidden;
        margin: 11px 0 12px;
        padding: 22px 24px;
        border: 1px solid #25334b;
        border-radius: 15px;
        background:
            radial-gradient(ellipse at 77% 55%, rgba(45, 74, 174, .18), transparent 33%),
            linear-gradient(112deg, #131c2c 0%, #0e1625 53%, #101528 100%);
        box-shadow: inset 0 1px rgba(255,255,255,.025), 0 14px 34px rgba(0,0,0,.18);
    }
    .hero-copy {
        position: relative;
        z-index: 2;
        width: 57%;
    }
    .hero-kicker {
        color: #a8b8d5;
        text-transform: uppercase;
        letter-spacing: .16em;
        font-weight: 700;
        font-size: .64rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 3.25vw, 2.85rem);
        line-height: .99;
        margin: 8px 0 7px;
        font-weight: 700;
    }
    .hero-gradient {
        background: linear-gradient(90deg, #a777ff, #4a9cff 62%, #35d7fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #d7e2f5;
        font-size: .92rem;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .hero-description {
        color: #9facbf;
        font-size: .75rem;
        line-height: 1.5;
        max-width: 515px;
        margin-bottom: 11px;
    }
    .hero-metrics { display: flex; gap: 8px; }
    .hero-metric {
        display: flex;
        align-items: center;
        gap: 7px;
        padding: 7px 10px;
        min-height: 34px;
        border: 1px solid #27473f;
        border-radius: 8px;
        color: #c9f8e9;
        background: rgba(26, 107, 81, .17);
        font-size: .68rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .hero-metric.accuracy {
        border-color: #274566;
        color: #cce8ff;
        background: rgba(34, 97, 158, .17);
    }
    .metric-value { font-family: 'Space Grotesk', sans-serif; font-size: .91rem; font-weight: 700; }
    .hero-art {
        position: absolute;
        right: 7px;
        top: 0;
        width: 48%;
        height: 100%;
        z-index: 1;
        pointer-events: none;
    }
    .hero-art svg { width: 100%; height: 100%; overflow: visible; }
    .float-label {
        position: absolute;
        right: 8px;
        z-index: 3;
        padding: 7px 10px;
        border-radius: 8px;
        background: rgba(13, 20, 34, .86);
        border: 1px solid rgba(117, 151, 192, .32);
        box-shadow: 0 8px 20px rgba(0,0,0,.25);
        font-size: .63rem;
        line-height: 1.25;
        color: #dce8f8;
        white-space: nowrap;
    }
    .float-label strong { display: block; font-size: .67rem; }
    .float-label.real { top: 34px; }
    .float-label.ai { bottom: 31px; right: 20px; }
    .float-label.real strong { color: #74e6bd; }
    .float-label.ai strong { color: #e794ff; }

    [data-testid="stHorizontalBlock"] { gap: .8rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: linear-gradient(150deg, rgba(17, 25, 39, .98), rgba(13, 19, 30, .98));
        box-shadow: 0 9px 24px rgba(0, 0, 0, .16), inset 0 1px rgba(255,255,255,.02);
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: .83rem .88rem;
    }
    .panel-heading {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #f1f5ff;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: .95rem;
        margin: 0 0 3px;
    }
    .panel-icon {
        width: 25px;
        height: 25px;
        display: grid;
        place-items: center;
        border-radius: 7px;
        background: rgba(146, 98, 255, .14);
        color: #b393ff;
        border: 1px solid rgba(146, 98, 255, .23);
        font-size: .86rem;
    }
    .panel-icon.blue { background: rgba(42, 138, 255, .14); border-color: rgba(54,151,255,.23); color: #66caff; }
    .panel-icon.purple { background: rgba(146, 98, 255, .14); border-color: rgba(146, 98, 255, .23); color: #b393ff; }
    .panel-subtitle { color: #929fb5; font-size: .67rem; margin-bottom: 11px; }
    .step {
        display: flex;
        align-items: flex-start;
        gap: 9px;
        padding: 8px 0;
        border-bottom: 1px solid rgba(100, 119, 147, .12);
    }
    .step:last-child { border-bottom: 0; }
    .step-icon {
        flex: 0 0 28px;
        height: 28px;
        display: grid;
        place-items: center;
        border-radius: 7px;
        color: #e9efff;
        font-size: .82rem;
        border: 1px solid rgba(102, 129, 255, .25);
        background: linear-gradient(145deg, rgba(121, 82, 255, .23), rgba(38, 125, 255, .13));
    }
    .step-title { color: #edf2fb; font-weight: 700; font-size: .72rem; margin: 0 0 2px; }
    .step-copy { color: #8f9db3; font-size: .62rem; line-height: 1.35; }

    div[data-testid="stFileUploader"] {
        background: #0c1421;
        border: 1px dashed #465879;
        border-radius: 10px;
        padding: 10px;
    }
    div[data-testid="stFileUploader"] section {
        background: transparent;
        border: 0;
        padding: .25rem;
    }
    div[data-testid="stFileUploader"] button {
        background: #1c2a43;
        border: 1px solid #344664;
        border-radius: 7px;
        color: #e5edfb;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span {
        color: #c1ccdd;
        font-size: .72rem;
    }
    div[data-testid="stAudio"] { margin-top: 6px; }
    div.stButton > button {
        min-height: 42px;
        border: 1px solid rgba(130, 160, 255, .35);
        border-radius: 9px;
        color: white;
        font-weight: 700;
        background: linear-gradient(105deg, #6046e8, #356cf4 54%, #238be9);
        box-shadow: 0 8px 18px rgba(42, 100, 235, .2);
        transition: filter .18s ease, transform .18s ease;
    }
    div.stButton > button:hover { filter: brightness(1.12); transform: translateY(-1px); border-color: #8cbbff; }
    div.stButton > button:disabled { background: #202b3d; color: #71809a; box-shadow: none; }

    .format-list { color: #8494ac; font-size: .63rem; margin: 5px 0 8px; text-align: center; }
    .selected-file {
        border: 1px solid #2a3c55;
        border-radius: 8px;
        background: #111d2e;
        padding: 8px 9px;
        margin: 6px 0;
        color: #d7e3f5;
        font-size: .68rem;
        overflow-wrap: anywhere;
    }
    .file-meta { color: #92a2ba; font-size: .62rem; margin-top: 3px; }
    .empty-result {
        min-height: 205px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        border: 1px dashed #34445d;
        border-radius: 10px;
        background: radial-gradient(ellipse at center, rgba(46, 79, 133, .14), transparent 70%);
        padding: 14px;
        margin: 8px 0;
    }
    .empty-result-icon { font-size: 1.75rem; margin-bottom: 5px; color: #70bfff; }
    .empty-result strong { color: #dfe9f8; font-size: .78rem; }
    .empty-result span { color: #8d9cb3; font-size: .65rem; line-height: 1.5; margin-top: 4px; max-width: 210px; }
    .result-card {
        padding: 11px;
        border-radius: 9px;
        border: 1px solid rgba(43, 212, 154, .35);
        background: linear-gradient(135deg, rgba(17, 72, 59, .42), rgba(13, 31, 31, .55));
        margin: 8px 0 9px;
    }
    .result-card.fake {
        border-color: rgba(255, 109, 139, .38);
        background: linear-gradient(135deg, rgba(91, 33, 54, .42), rgba(37, 21, 33, .6));
    }
    .result-card.neutral {
        border-color: rgba(154, 169, 193, .35);
        background: linear-gradient(135deg, rgba(38, 49, 67, .48), rgba(20, 27, 40, .62));
    }
    .result-label { color: #7ce9c0; font-size: .62rem; text-transform: uppercase; letter-spacing: .11em; font-weight: 700; }
    .result-card.fake .result-label { color: #ff9aaf; }
    .result-class { color: #effbf7; font-family: 'Space Grotesk', sans-serif; font-size: 1.03rem; font-weight: 700; margin: 3px 0 2px; }
    .result-card.fake .result-class { color: #fff0f3; }
    .result-card.neutral .result-class { color: #e4ebf6; }
    .result-explainer { color: #adbecc; font-size: .64rem; line-height: 1.42; }
    .score-header { display: flex; justify-content: space-between; align-items: baseline; color: #a9b8cb; font-size: .68rem; margin: 9px 0 5px; }
    .score-value { color: #73e5bf; font-family: 'Space Grotesk', sans-serif; font-size: 1.48rem; font-weight: 700; }
    .score-value.fake { color: #ff93aa; }
    .score-track { height: 7px; background: #252d3b; border-radius: 99px; overflow: hidden; }
    .score-fill { height: 100%; background: linear-gradient(90deg, #2bd49a, #28bfff); border-radius: inherit; }
    .score-fill.fake { background: linear-gradient(90deg, #ff819f, #f05178); }
    .disclaimer {
        border: 1px solid #344055;
        border-radius: 8px;
        padding: 8px 9px;
        background: rgba(20, 29, 44, .75);
        color: #a1afc2;
        font-size: .61rem;
        line-height: 1.48;
        margin-top: 9px;
    }
    .sample-section {
        margin-top: 12px;
        padding: 11px 14px;
        border: 1px solid #263248;
        border-radius: 12px;
        background: linear-gradient(110deg, rgba(17, 25, 39, .98), rgba(13, 19, 30, .98));
    }
    .sample-title { color: #f1f5ff; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: .95rem; }
    .sample-subtitle { color: #8e9db3; font-size: .66rem; margin: 2px 0 9px; }
    .sample-empty {
        padding: 11px;
        border: 1px dashed #344259;
        border-radius: 8px;
        text-align: center;
        color: #8797ae;
        font-size: .68rem;
        background: rgba(9, 15, 25, .48);
    }
    .footer-note { color: #687991; font-size: .58rem; text-align: center; padding: 7px 0 0; }

    @media (max-width: 900px) {
        .hero { height: auto; min-height: 245px; }
        .hero-copy { width: 70%; }
        .hero-art { width: 43%; opacity: .8; }
        .float-label { display: none; }
        .nav-items a { padding: 7px 6px; font-size: .7rem; }
    }
    @media (max-width: 640px) {
        .block-container { padding: .45rem .65rem 1rem; }
        .navbar { align-items: flex-start; flex-direction: column; }
        .nav-items { width: 100%; overflow-x: auto; }
        .hero { min-height: 300px; padding: 17px 15px; }
        .hero-copy { width: 100%; }
        .hero-art { width: 65%; height: 120px; top: auto; bottom: 0; right: -5px; opacity: .46; }
        .hero-description { max-width: 100%; }
        .hero-metrics { flex-wrap: wrap; }
        .hero-metric { white-space: normal; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "acoustic_model.joblib"
    return joblib.load(model_path)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

model_classes = np.asarray(getattr(model, "classes_", []))
expected_features = getattr(model, "n_features_in_", None)
has_fitted_scaler = isinstance(model, Pipeline) and any(
    isinstance(step, StandardScaler)
    and hasattr(step, "mean_")
    and hasattr(step, "scale_")
    for _, step in model.steps
)
if (
    expected_features != 26
    or not np.array_equal(model_classes, np.array([0, 1]))
    or not has_fitted_scaler
):
    st.error(
        "The saved model is not compatible with this app. "
        "Expected a fitted StandardScaler pipeline, 26 input features, "
        "and class labels [0, 1]."
    )
    st.stop()

def load_class_name_mapping(model_path, classes):
    metadata_path = model_path.with_name("acoustic_model_metadata.json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(metadata, dict) or metadata.get("metadata_version") != 1:
        return {}
    if metadata.get("model_sha256") != hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest():
        return {}

    class_names = metadata.get("class_names")
    expected_labels = [
        str(label.item() if isinstance(label, np.generic) else label)
        for label in classes
    ]
    if not isinstance(class_names, dict) or set(class_names) != set(expected_labels):
        return {}

    names = [class_names[label] for label in expected_labels]
    mapping_source = metadata.get("mapping_source")
    if (
        set(names) != {"Real", "AI-generated"}
        or not all(isinstance(name, str) for name in names)
        or not isinstance(mapping_source, str)
        or not mapping_source.strip()
    ):
        return {}

    return dict(zip(classes.tolist(), names))


model_path = Path(__file__).parent / "acoustic_model.joblib"
class_name_by_label = load_class_name_mapping(model_path, model_classes)

@st.cache_data(show_spinner=False)
def decode_audio(audio_bytes):
    signal, sample_rate = librosa.load(
        io.BytesIO(audio_bytes),
        sr=16000,
        mono=True,
    )
    if signal.size == 0:
        raise ValueError("The selected audio file contains no audio samples.")
    return signal, sample_rate


def extract_features(signal, sample_rate):
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    feature_vector = np.concatenate(
        (np.mean(mfcc, axis=1), np.std(mfcc, axis=1))
    )
    if feature_vector.shape != (26,):
        raise ValueError(
            f"Feature extraction returned {feature_vector.size} values; expected 26."
        )
    return feature_vector.reshape(1, -1)


st.markdown(
    """
    <div id="top" class="navbar">
        <div class="brand"><span class="brand-icon">♬</span>Acoustic Reality Intelligence</div>
        <nav class="nav-items" aria-label="Main navigation">
            <a class="current" href="#top">⌂ &nbsp;Home</a>
            <a href="#how-it-works">⟲ &nbsp;How It Works</a>
            <a href="#about-project">◉ &nbsp;About</a>
            <a href="#contact">✉ &nbsp;Contact</a>
        </nav>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="hero-copy">
            <div class="hero-kicker">Machine Learning · Audio Intelligence</div>
            <h1>Acoustic Reality<br><span class="hero-gradient">Intelligence</span></h1>
            <div class="hero-subtitle">AI-Generated Environmental Sound Detection</div>
            <div class="hero-description">
                Explore acoustic patterns to estimate whether environmental audio is real or AI-generated.
                An experimental classifier, not proof of authenticity.
            </div>
            <div class="hero-metrics">
                <div class="hero-metric"><span>●</span> Trained with SVM Machine Learning</div>
                <div class="hero-metric accuracy"><span class="metric-value">83.75%</span> Test Accuracy</div>
            </div>
        </div>
        <div class="hero-art">
            <svg viewBox="0 0 590 250" role="img" aria-label="Stylized headphones and glowing audio waveform">
                <defs>
                    <linearGradient id="waveGradient" x1="0" x2="1">
                        <stop offset="0%" stop-color="#20d9ff"/>
                        <stop offset="35%" stop-color="#357eff"/>
                        <stop offset="64%" stop-color="#a34fff"/>
                        <stop offset="82%" stop-color="#fa51bf"/>
                        <stop offset="100%" stop-color="#ff5068"/>
                    </linearGradient>
                    <linearGradient id="headsetGradient" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stop-color="#97eaff"/>
                        <stop offset="45%" stop-color="#647fff"/>
                        <stop offset="100%" stop-color="#a45bff"/>
                    </linearGradient>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="5" result="blur"/>
                        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                    </filter>
                    <radialGradient id="artGlow">
                        <stop offset="0%" stop-color="#347aff" stop-opacity=".3"/>
                        <stop offset="100%" stop-color="#347aff" stop-opacity="0"/>
                    </radialGradient>
                </defs>
                <ellipse cx="340" cy="128" rx="210" ry="125" fill="url(#artGlow)"/>
                <path d="M12 149 C48 149 45 132 66 132 S87 166 105 166 S124 105 143 105 S159 184 177 184 S194 77 212 77 S230 196 248 196 S265 101 283 101 S300 174 318 174 S337 116 354 116 S372 158 389 158 S407 92 425 92 S443 180 460 180 S479 132 495 132 S515 149 576 149"
                    fill="none" stroke="url(#waveGradient)" stroke-width="3.3" opacity=".92" filter="url(#glow)"/>
                <path d="M20 149 C65 149 66 143 100 143 S143 153 177 153 S220 143 254 143 S297 153 331 153 S374 143 408 143 S451 153 485 153 S528 149 572 149"
                    fill="none" stroke="url(#waveGradient)" stroke-width="1.2" opacity=".58"/>
                <path d="M232 133 V104 C232 44 274 18 324 18 C374 18 416 44 416 104 V133"
                    fill="none" stroke="#6479d8" stroke-opacity=".28" stroke-width="30" stroke-linecap="round"/>
                <path d="M232 128 V103 C232 53 273 30 324 30 C375 30 416 53 416 103 V128"
                    fill="none" stroke="url(#headsetGradient)" stroke-width="12" stroke-linecap="round" filter="url(#glow)"/>
                <rect x="218" y="106" width="38" height="73" rx="15" fill="#192b4d" stroke="#66caff" stroke-width="2.2"/>
                <rect x="225" y="115" width="24" height="54" rx="10" fill="url(#headsetGradient)" opacity=".78"/>
                <rect x="392" y="106" width="38" height="73" rx="15" fill="#192b4d" stroke="#a875ff" stroke-width="2.2"/>
                <rect x="399" y="115" width="24" height="54" rx="10" fill="url(#headsetGradient)" opacity=".82"/>
                <path d="M255 179 C270 196 286 204 305 207" fill="none" stroke="#5d87f4" stroke-width="3" opacity=".45"/>
                <circle cx="180" cy="92" r="2" fill="#35d8ff"/><circle cx="454" cy="105" r="2" fill="#d56dff"/>
                <circle cx="460" cy="197" r="1.5" fill="#ff6cae"/><circle cx="186" cy="197" r="1.5" fill="#678eff"/>
            </svg>
        </div>
        <div class="float-label real"><strong>Real Sound</strong>Natural Patterns</div>
        <div class="float-label ai"><strong>AI-Generated</strong>Synthetic Patterns</div>
    </section>
    """,
    unsafe_allow_html=True,
)

left, center, right = st.columns([0.94, 1.35, 1.02], gap="small")

with left:
    with st.container(border=True):
        st.markdown(
            '<div id="how-it-works" class="panel-heading"><span class="panel-icon purple">⚙</span>How It Works</div>'
            '<div class="panel-subtitle">From sound waves to an informed estimate.</div>',
            unsafe_allow_html=True,
        )
        workflow_steps = [
            ("↑", "Upload Audio", "Choose an environmental recording to analyse."),
            ("≋", "Feature Extraction", "Summarise 13 MFCCs as 26 numeric features."),
            ("✦", "AI Analysis", "The trained SVM evaluates the feature pattern."),
            ("✓", "Get Prediction", "Review its class and AI-generated score."),
        ]
        for icon, title, description in workflow_steps:
            st.markdown(
                f'<div class="step"><div class="step-icon">{icon}</div>'
                f'<div><div class="step-title">{title}</div>'
                f'<div class="step-copy">{description}</div></div></div>',
                unsafe_allow_html=True,
            )

with center:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-heading"><span class="panel-icon blue">⇧</span>Upload Audio File</div>'
            '<div class="panel-subtitle">Select a recording to begin acoustic analysis.</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Drop an audio file here or browse",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            accept_multiple_files=False,
            help="Audio formats are supported when compatible decoders are installed.",
            key="audio_upload",
            label_visibility="collapsed",
        )
        audio_signal = None
        audio_sample_rate = None
        audio_duration = None
        audio_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        audio_error = None

        if uploaded_file is not None and audio_bytes is not None:
            try:
                audio_signal, audio_sample_rate = decode_audio(audio_bytes)
                audio_duration = librosa.get_duration(
                    y=audio_signal,
                    sr=audio_sample_rate,
                )
                safe_name = html.escape(uploaded_file.name)
                st.markdown(
                    f'<div class="selected-file"><strong>{safe_name}</strong>'
                    f'<div class="file-meta">{len(audio_bytes) / 1024:.1f} KB'
                    f' &nbsp;·&nbsp; {audio_duration:.2f} seconds &nbsp;·&nbsp; 16 kHz mono</div></div>',
                    unsafe_allow_html=True,
                )
                mime_types = {
                    ".wav": "audio/wav",
                    ".mp3": "audio/mpeg",
                    ".ogg": "audio/ogg",
                    ".flac": "audio/flac",
                    ".m4a": "audio/mp4",
                }
                audio_format = mime_types.get(
                    Path(uploaded_file.name).suffix.lower(),
                    "audio/wav",
                )
                st.audio(audio_bytes, format=audio_format)
            except Exception as error:
                audio_error = error
                st.error(f"Unable to decode this recording: {error}")

        st.markdown(
            '<div class="format-list">Supported formats: WAV · MP3 · OGG · FLAC · M4A</div>',
            unsafe_allow_html=True,
        )
        detect_clicked = st.button(
            "Detect Sound",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None or audio_signal is None,
        )
        if detect_clicked and audio_signal is not None and audio_sample_rate is not None:
            try:
                with st.spinner("Extracting acoustic features and analysing with the trained SVM..."):
                    feature_vector = extract_features(audio_signal, audio_sample_rate)
                    prediction = model.predict(feature_vector)[0]
                    st.session_state.analysis_result = {
                        "prediction": prediction.item()
                        if isinstance(prediction, np.generic)
                        else prediction,
                    }
            except Exception as error:
                st.error(f"Analysis failed: {error}")
        if audio_error is not None:
            st.caption("Try a supported format that can be decoded by the installed audio libraries.")

with right:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-heading"><span class="panel-icon blue">▥</span>Detection Result</div>'
            '<div class="panel-subtitle">Your model prediction appears here.</div>',
            unsafe_allow_html=True,
        )
        result = st.session_state.analysis_result
        if result is None:
            st.markdown(
                '<div class="empty-result"><div class="empty-result-icon">⌁</div>'
                '<strong>Ready to analyse</strong>'
                '<span>Upload an audio recording and select Detect Sound to see the model output.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            prediction = result["prediction"]
            class_name = class_name_by_label.get(prediction)
            if class_name == "Real":
                headline = "Your audio appears to be REAL"
                result_style = " real"
            elif class_name == "AI-generated":
                headline = "Your audio appears to be AI-GENERATED"
                result_style = " fake"
            else:
                headline = "Class mapping must be supplied before naming this prediction"
                result_style = " neutral"
            st.markdown(
                f'<div class="result-card{result_style}">'
                f'<div class="result-class">{headline}</div></div>',
                unsafe_allow_html=True,
            )

st.markdown(
    """
    <section class="sample-section">
        <div class="sample-title">Try Sample Sounds</div>
        <div class="sample-subtitle">Test with these example environmental sounds</div>
        <div class="sample-empty">
            No sample audio files are included in this project yet. Upload your own recording above to run a real analysis.
        </div>
    </section>
    <div id="about-project" class="footer-note">
        The 83.75% baseline was measured on the original 80-sample random test split; it does not guarantee performance on new recordings.
        <span id="contact"></span>
    </div>
    """,
    unsafe_allow_html=True,
)
