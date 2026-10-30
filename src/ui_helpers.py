"""
UI Helper functions and styling for Acoustic Reality Intelligence.
Provides unified dark-theme styling, navigation components, metric cards, and charts.
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional


DARK_THEME_CSS = """
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
        radial-gradient(ellipse at 85% -5%, rgba(56, 120, 255, .15), transparent 35%),
        radial-gradient(ellipse at 10% 5%, rgba(49, 210, 255, .10), transparent 25%),
        var(--page);
}

.block-container {
    max-width: 1380px;
    padding: 1.0rem 1.4rem 2rem;
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text);
    letter-spacing: -0.03em;
}

p, label, [data-testid="stCaptionContainer"] {
    color: var(--muted);
}

header[data-testid="stHeader"] {
    height: 0;
    background: transparent;
}
[data-testid="stToolbar"] { display: none; }

/* Custom Metric Card */
.ari-card {
    background: linear-gradient(135deg, rgba(16, 23, 37, 0.95), rgba(12, 17, 28, 0.95));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}

.ari-card:hover {
    border-color: rgba(56, 120, 255, 0.4);
}

.ari-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.ari-badge.real {
    background: rgba(43, 212, 154, 0.15);
    color: #2bd49a;
    border: 1px solid rgba(43, 212, 154, 0.35);
}
.ari-badge.ai {
    background: rgba(255, 109, 139, 0.15);
    color: #ff6d8b;
    border: 1px solid rgba(255, 109, 139, 0.35);
}
.ari-badge.neutral {
    background: rgba(56, 120, 255, 0.15);
    color: #3878ff;
    border: 1px solid rgba(56, 120, 255, 0.35);
}

/* Result Headlines */
.headline-card {
    border-radius: 12px;
    padding: 18px 20px;
    margin: 14px 0;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.headline-card.real {
    border: 1.5px solid #2bd49a;
    background: linear-gradient(135deg, rgba(15, 60, 45, 0.45), rgba(8, 28, 22, 0.6));
}
.headline-card.ai {
    border: 1.5px solid #ff6d8b;
    background: linear-gradient(135deg, rgba(75, 20, 35, 0.45), rgba(35, 10, 18, 0.6));
}
.headline-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
}
.headline-card.real .headline-title { color: #2bd49a; }
.headline-card.ai .headline-title { color: #ff6d8b; }

/* Empty States */
.empty-box {
    border: 1.5px dashed #263248;
    border-radius: 12px;
    padding: 32px 20px;
    text-align: center;
    background: rgba(16, 23, 37, 0.4);
    margin: 16px 0;
}
.empty-box-icon {
    font-size: 2.2rem;
    color: #3878ff;
    margin-bottom: 8px;
}
.empty-box h4 {
    margin: 4px 0;
    color: #f3f6ff;
}
.empty-box p {
    font-size: 0.85rem;
    color: #9aa9c1;
    max-width: 420px;
    margin: 0 auto;
}

/* Primary Button Styling */
div.stButton > button[kind="primary"] {
    background: linear-gradient(105deg, #4f32e6, #2d62ea 55%, #1886e4);
    border: 1px solid rgba(130, 160, 255, 0.4);
    color: #ffffff;
    font-weight: 700;
    border-radius: 9px;
    padding: 0.55rem 1.2rem;
    box-shadow: 0 4px 14px rgba(45, 98, 234, 0.25);
    transition: all 0.2s ease;
}
div.stButton > button[kind="primary"]:hover {
    filter: brightness(1.15);
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(45, 98, 234, 0.35);
}
</style>
"""


def apply_theme():
    """Injects global dark theme stylesheet."""
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, badge_text: Optional[str] = None):
    """Renders a cohesive top title section."""
    badge_html = f'<span class="ari-badge neutral">{badge_text}</span>' if badge_text else ''
    st.markdown(
        f"""
        <div style="margin-bottom: 1.2rem;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                <h2 style="margin: 0; font-size: 1.7rem; font-weight: 700;">{title}</h2>
                {badge_html}
            </div>
            <div style="color: #9aa9c1; font-size: 0.88rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_empty_state(title: str, description: str, icon: str = "🎧"):
    """Renders a standard empty state box."""
    st.markdown(
        f"""
        <div class="empty-box">
            <div class="empty-box-icon">{icon}</div>
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def setup_dark_matplotlib_figure(figsize=(9, 3.2)):
    """Creates a matplotlib figure pre-styled for dark dashboard themes."""
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.patch.set_facecolor("#101725")
    ax.set_facecolor("#0a0f1d")
    ax.tick_params(colors="#9aa9c1", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#263248")
    return fig, ax

