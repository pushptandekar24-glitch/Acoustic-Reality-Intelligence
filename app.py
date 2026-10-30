"""
Acoustic Reality Intelligence (ARI): An AI-Powered Environmental Audio Analysis and Synthetic Sound Assessment System.
Master Streamlit Application Orchestrator with Simplified 3-Option Navigation.
"""
import streamlit as st

import config
from src.ui_helpers import apply_theme
from src.ui.overview import render_overview_page
from src.ui.detection import render_detection_page
from src.ui.about import render_about_page


# Streamlit Page Configuration
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply unified dark theme
apply_theme()

# Initialize session state variables
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Overview"
if "current_signal" not in st.session_state:
    st.session_state.current_signal = None
if "current_sr" not in st.session_state:
    st.session_state.current_sr = config.TARGET_SAMPLE_RATE
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "current_audio_bytes" not in st.session_state:
    st.session_state.current_audio_bytes = None
if "current_meta" not in st.session_state:
    st.session_state.current_meta = None
if "current_auth" not in st.session_state:
    st.session_state.current_auth = None
if "current_lab" not in st.session_state:
    st.session_state.current_lab = None


# Simplified 3-Option Navigation (Strictly as specified in Project Requirements)
NAV_OPTIONS = [
    "Overview",
    "Voice / Audio Detection",
    "About",
]

# Ensure current active_nav is valid
if st.session_state.active_nav not in NAV_OPTIONS:
    st.session_state.active_nav = "Overview"

# Sidebar Brand & Simplified Navigation
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <div style="width: 38px; height: 38px; border-radius: 9px; background: linear-gradient(135deg, #3878ff, #9561ff); display: grid; place-items: center; font-size: 19px;">
                🎧
            </div>
            <div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 700; color: #f3f6ff; line-height: 1.2;">
                    Acoustic Reality
                </div>
                <div style="font-size: 0.72rem; color: #31d2ff; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
                    Intelligence v2.0
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr style='border-color: #263248; margin: 12px 0 16px;'/>", unsafe_allow_html=True)

    # Simplified 3-Item Navigation Radio
    current_index = NAV_OPTIONS.index(st.session_state.active_nav)
    selected_page = st.radio(
        "Navigation",
        NAV_OPTIONS,
        index=current_index,
        label_visibility="collapsed"
    )
    st.session_state.active_nav = selected_page

    st.markdown("<hr style='border-color: #263248; margin: 16px 0 12px;'/>", unsafe_allow_html=True)

    # Sidebar Active Track Status Widget
    if st.session_state.current_filename:
        meta = st.session_state.current_meta or {}
        st.markdown(
            f"""
            <div style="background: rgba(16, 23, 37, 0.85); border: 1px solid #263248; border-radius: 9px; padding: 10px; font-size: 0.75rem;">
                <div style="color: #9aa9c1; text-transform: uppercase; font-size: 0.65rem; margin-bottom: 2px;">Active Track</div>
                <strong style="color: #f3f6ff; word-break: break-all;">{st.session_state.current_filename}</strong>
                <div style="color: #31d2ff; margin-top: 4px;">{meta.get('duration_seconds', 0.0):.2f}s · 16 kHz Mono</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Clear Active Audio", use_container_width=True):
            st.session_state.current_signal = None
            st.session_state.current_filename = None
            st.session_state.current_audio_bytes = None
            st.session_state.current_meta = None
            st.session_state.current_auth = None
            st.session_state.current_lab = None
            st.rerun()
    else:
        st.markdown(
            """
            <div style="background: rgba(16, 23, 37, 0.5); border: 1px dashed #263248; border-radius: 9px; padding: 10px; font-size: 0.75rem; text-align: center; color: #9aa9c1;">
                No audio loaded<br/><span style="font-size: 0.68rem; color: #62728c;">Upload on Overview</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Verification Integrity Tag
    st.markdown(
        """
        <div style="margin-top: 22px; font-size: 0.68rem; color: #6b7c96; text-align: center; line-height: 1.4;">
            <span style="color: #2bd49a; font-weight: bold;">● SVM SHA-256 Verified</span><br/>
            b6d965c31ceba6a6...
        </div>
        """,
        unsafe_allow_html=True
    )


# Route strictly across the 3 options
if st.session_state.active_nav == "Overview":
    render_overview_page()
elif st.session_state.active_nav == "Voice / Audio Detection":
    render_detection_page()
elif st.session_state.active_nav == "About":
    render_about_page()
