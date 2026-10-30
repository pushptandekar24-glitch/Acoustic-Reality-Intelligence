"""
Sample Audio Demonstration Section for Acoustic Reality Intelligence.
Renders the 'Try Sample Audio — Real vs AI-Generated' showcase,
per-sample analysis with verified headlines, and two-sample comparison.
"""
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional

import config
from src.sample_library import (
    load_sample_manifest,
    resolve_sample_path,
    analyze_sample_audio,
    compare_two_samples,
    get_comparison_sample_lists,
)
from src.synthetic_detection import compute_file_sha256


def render_sample_card(
    sample_key: str,
    sample_info: Dict[str, Any],
    role_label: str,
    sample_dir: Optional[Path] = None,
):
    """
    Renders an individual sample card:
    - Heading and role (Sample A — Real Audio / Sample B — AI-Generated Audio)
    - Audio player
    - Filename, duration, source, and verified ground-truth status
    - One-click 'Analyze This Sample' button
    - Results card with headline, model verification, ground truth matching
    """
    sample_id = sample_info.get("sample_id", sample_key)
    filename = sample_info.get("filename")
    is_verified = bool(sample_info.get("is_verified", False))
    gt_label = sample_info.get("ground_truth_label", "Demo audio — authenticity label unverified")
    source = sample_info.get("source", "Unknown")
    license_info = sample_info.get("license", "Unknown")
    generator = sample_info.get("generator", "Unknown")
    caption = sample_info.get("caption", "")

    file_path = resolve_sample_path(filename, sample_dir) if filename else None
    file_exists = bool(file_path and file_path.exists())

    # Card Container
    st.markdown(
        f"""
        <div class="ari-card" style="border-top: 3px solid {'#2bd49a' if 'Real' in role_label else '#ff6d8b'}; min-height: 100%;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                <div>
                    <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: {'#2bd49a' if 'Real' in role_label else '#ff6d8b'}; font-weight: 700;">
                        {role_label}
                    </span>
                    <h4 style="margin: 2px 0 0; font-size: 1.0rem; color: #f3f6ff;">
                        {sample_info.get('display_title', filename or 'Unavailable Sample')}
                    </h4>
                </div>
                <span class="ari-badge {'real' if is_verified and sample_info.get('ground_truth_class') == 0 else ('ai' if is_verified else 'neutral')}" style="font-size: 0.68rem;">
                    {'✓ Verified Ground Truth' if is_verified else 'Unverified Label'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Missing file handling
    if not file_exists:
        st.markdown(
            f"""
            <div style="background: rgba(255, 109, 139, 0.08); border: 1px dashed rgba(255, 109, 139, 0.4); border-radius: 9px; padding: 14px; margin-bottom: 12px; font-size: 0.8rem; color: #ff9bb0;">
                <strong style="color: #ff6d8b;">⚠ Sample File Unavailable</strong><br/>
                {sample_info.get('notes') or 'No audio file is currently available for this category.'}
                <div style="margin-top: 6px; font-size: 0.72rem; color: #9aa9c1;">
                    Anti-fabrication standard: Procedurally generated tones are strictly prohibited from being labeled as AI-generated audio.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Audio playback
    try:
        audio_bytes = file_path.read_bytes()
        st.audio(audio_bytes)
    except Exception as exc:
        st.error(f"Could not load audio for playback: {exc}")
        return

    # Inspect duration & specs for display
    # Quick cached inspection from sample_library
    analysis_key = f"analysis_res_{sample_id}"
    res = st.session_state.get(analysis_key)

    # If not analyzed yet, run or peek metadata
    if res is None:
        # Pre-compute basic metadata for initial card display
        pre_res = analyze_sample_audio(sample_info, sample_dir=sample_dir, use_cache=True)
        dur_sec = pre_res.get("duration_seconds", 0.0)
        channels_desc = pre_res.get("channels_desc", "Mono")
        sr_desc = f"{pre_res.get('sample_rate', 16000)} Hz"
    else:
        dur_sec = res.get("duration_seconds", 0.0)
        channels_desc = res.get("channels_desc", "Mono")
        sr_desc = f"{res.get('sample_rate', 16000)} Hz"

    # Compact Metadata Section
    gt_badge_html = (
        f'<span style="color: #2bd49a; font-weight: 600;">{gt_label}</span>'
        if is_verified and sample_info.get("ground_truth_class") == 0
        else (
            f'<span style="color: #ff6d8b; font-weight: 600;">{gt_label}</span>'
            if is_verified
            else '<span style="color: #9aa9c1; font-style: italic;">Demo audio — authenticity label unverified</span>'
        )
    )

    st.markdown(
        f"""
        <div style="background: rgba(16, 23, 37, 0.6); border: 1px solid #263248; border-radius: 8px; padding: 10px 12px; margin-bottom: 12px; font-size: 0.78rem;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                <div><span style="color: #9aa9c1;">File:</span> <strong style="color: #f3f6ff;">{filename}</strong></div>
                <div><span style="color: #9aa9c1;">Duration:</span> <strong style="color: #31d2ff;">{dur_sec:.2f}s</strong></div>
                <div><span style="color: #9aa9c1;">Format:</span> <span style="color: #f3f6ff;">{sr_desc} · {channels_desc}</span></div>
                <div><span style="color: #9aa9c1;">Ground Truth:</span> {gt_badge_html}</div>
            </div>
            <div style="margin-top: 6px; padding-top: 6px; border-top: 1px dashed #263248; font-size: 0.72rem; color: #8e9db3;">
                <div><strong style="color: #9aa9c1;">Source:</strong> {source}</div>
                <div><strong style="color: #9aa9c1;">Generator/Origin:</strong> {generator}</div>
                <div><strong style="color: #9aa9c1;">License:</strong> {license_info}</div>
                {f"<div><strong style='color: #9aa9c1;'>Caption:</strong> <em>{caption}</em></div>" if caption else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # One-Click Analysis Button
    btn_col1, btn_col2 = st.columns([1.3, 1.0])
    with btn_col1:
        if st.button(f"🔍 Analyze This Sample", key=f"btn_analyze_{sample_id}", type="primary", use_container_width=True):
            with st.spinner("Processing audio through 26-MFCC DSP & SVM pipeline..."):
                analysis_result = analyze_sample_audio(sample_info, sample_dir=sample_dir, use_cache=True)
                st.session_state[analysis_key] = analysis_result
                res = analysis_result
                st.rerun()

    with btn_col2:
        if st.button("Set Active Track ➔", key=f"btn_set_active_{sample_id}", use_container_width=True):
            if res and res.get("status") == "ready":
                st.session_state.current_signal = res["signal"]
                st.session_state.current_filename = res["filename"]
                st.session_state.current_sr = res["sample_rate"]
                st.session_state.current_audio_bytes = res["file_bytes"]
                st.session_state.current_meta = res["meta"]
                st.session_state.current_auth = None
                st.success(f"Loaded '{filename}' as active track.")
                st.rerun()
            else:
                pre_res = analyze_sample_audio(sample_info, sample_dir=sample_dir, use_cache=True)
                if pre_res.get("status") == "ready":
                    st.session_state.current_signal = pre_res["signal"]
                    st.session_state.current_filename = pre_res["filename"]
                    st.session_state.current_sr = pre_res["sample_rate"]
                    st.session_state.current_audio_bytes = pre_res["file_bytes"]
                    st.session_state.current_meta = pre_res["meta"]
                    st.session_state.current_auth = None
                    st.success(f"Loaded '{filename}' as active track.")
                    st.rerun()

    # Display Analysis Results Card
    if res and res.get("status") == "ready":
        pred_class = res["prediction_class"]
        headline = res["headline"]
        is_ai = bool(pred_class == 1)
        card_class = "ai" if is_ai else "real"
        status_text = res["status_text"]
        model_verified = res["model_verified"]
        ai_prob = res["ai_probability"] * 100.0
        real_prob = res["real_probability"] * 100.0

        st.markdown(
            f"""
            <div class="headline-card {card_class}" style="margin: 10px 0 6px; padding: 14px 16px;">
                <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: {'#ff6d8b' if is_ai else '#2bd49a'}; font-weight: bold; margin-bottom: 2px;">
                    SVM Decision: Class {pred_class} ({res['class_name']})
                </div>
                <div class="headline-title" style="font-size: 1.15rem; margin-bottom: 6px;">
                    {headline}
                </div>
                <div style="display: flex; justify-content: center; gap: 16px; font-size: 0.78rem; color: #f3f6ff; margin-top: 4px;">
                    <span>P(Real): <strong>{real_prob:.1f}%</strong></span>
                    <span>P(AI-Generated): <strong>{ai_prob:.1f}%</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Ground Truth & Checksum Verification Details
        gt_display = (
            f"<span style='color: #2bd49a; font-weight: bold;'>Real Audio (Class 0)</span>"
            if res.get("ground_truth_class") == 0
            else (
                f"<span style='color: #ff6d8b; font-weight: bold;'>AI-Generated Audio (Class 1)</span>"
                if res.get("ground_truth_class") == 1
                else "<span style='color: #9aa9c1; font-style: italic;'>Demo audio — authenticity label unverified</span>"
            )
        )

        match_badge_color = (
            "#2bd49a" if res.get("prediction_matches_ground_truth") is True
            else ("#ff6d8b" if res.get("prediction_matches_ground_truth") is False else "#31d2ff")
        )

        st.markdown(
            f"""
            <div style="background: rgba(10, 15, 26, 0.7); border: 1px solid #263248; border-radius: 8px; padding: 10px 12px; margin-top: 6px; font-size: 0.75rem;">
                <div style="margin-bottom: 4px;">
                    <span style="color: #9aa9c1;">Verified Ground Truth:</span> {gt_display}
                </div>
                <div style="margin-bottom: 4px;">
                    <span style="color: #9aa9c1;">Verification Status:</span>
                    <strong style="color: {match_badge_color};">{status_text}</strong>
                </div>
                <div style="font-size: 0.68rem; color: #6b7c96; margin-top: 4px; border-top: 1px dotted #263248; padding-top: 4px;">
                    <span style="color: {'#2bd49a' if model_verified else '#ff6d8b'};">
                        {'✓ SVM SHA-256 Validated' if model_verified else '⚠ Checksum Mismatch'}
                    </span> · {res.get('model_sha256', '')[:16]}...
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_sample_library_section():
    """
    Renders the complete 'Try Sample Audio — Real vs AI-Generated' demonstration section.
    """
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: #263248; margin: 24px 0 18px;'/>", unsafe_allow_html=True)

    # Section Header
    st.markdown(
        """
        <div style="margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <h3 style="margin: 0; font-size: 1.45rem; color: #f3f6ff;">
                    🎧 Try Sample Audio — Real vs AI-Generated
                </h3>
                <span class="ari-badge neutral" style="font-size: 0.72rem;">Demonstration Library</span>
            </div>
            <p style="color: #9aa9c1; font-size: 0.86rem; margin-top: 4px; line-height: 1.45;">
                Explore example environmental audio recordings, listen to the samples, and run them through the 
                experimental audio authenticity classifier. These examples are for demonstration and do not guarantee correct predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        manifest = load_sample_manifest()
    except Exception as exc:
        st.error(f"Unable to load sample library manifest: {exc}")
        return

    categories = manifest.get("categories", {})

    # Render each of the three categories
    for cat_id, cat_info in categories.items():
        cat_name = cat_info.get("name", cat_id.replace("_", " ").title())
        cat_icon = cat_info.get("icon", "🎵")
        cat_desc = cat_info.get("description", "")
        samples = cat_info.get("samples", {})

        st.markdown(
            f"""
            <div style="margin-top: 18px; margin-bottom: 8px;">
                <h4 style="margin: 0; font-size: 1.15rem; color: #31d2ff;">
                    {cat_icon} {cat_name}
                </h4>
                <div style="font-size: 0.78rem; color: #8e9db3;">{cat_desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns(2, gap="medium")

        # Sample A — Real Audio
        sample_a = samples.get("sample_a_real", {})
        with col_a:
            render_sample_card(
                sample_key=f"{cat_id}_sample_a",
                sample_info=sample_a,
                role_label="Sample A — Real Audio",
            )

        # Sample B — AI-Generated Audio
        sample_b = samples.get("sample_b_ai", {})
        with col_b:
            render_sample_card(
                sample_key=f"{cat_id}_sample_b",
                sample_info=sample_b,
                role_label="Sample B — AI-Generated Audio",
            )

    # -------------------------------------------------------------
    # Two-Sample Comparison Feature
    # -------------------------------------------------------------
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: #263248; margin: 24px 0 18px;'/>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <h3 style="margin: 0; font-size: 1.35rem; color: #f3f6ff;">
                    ⚖ Compare Real vs AI-Generated Samples
                </h3>
                <span class="ari-badge neutral" style="font-size: 0.72rem;">Side-by-Side Assessment</span>
            </div>
            <p style="color: #9aa9c1; font-size: 0.84rem; margin-top: 4px; line-height: 1.45;">
                Select a real audio recording and an AI-generated sample to inspect spectral metrics, run comparative inference,
                and compare ground-truth labels against the SVM predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    real_list, ai_list = get_comparison_sample_lists(manifest)

    if not real_list or not ai_list:
        st.warning("Insufficient audio samples available on disk to populate comparison selectors.")
        return

    # Dropdown Selection
    sel_col1, sel_col2 = st.columns(2, gap="medium")

    with sel_col1:
        st.markdown("<strong style='font-size: 0.85rem; color: #2bd49a;'>Select Real Audio Sample:</strong>", unsafe_allow_html=True)
        real_names = [f"{s.get('category_name')}: {s.get('filename')}" for s in real_list]
        selected_real_idx = st.selectbox(
            "Select Real Sample",
            range(len(real_names)),
            format_func=lambda i: real_names[i],
            label_visibility="collapsed",
            key="comp_real_select"
        )
        chosen_real = real_list[selected_real_idx]

    with sel_col2:
        st.markdown("<strong style='font-size: 0.85rem; color: #ff6d8b;'>Select AI-Generated Audio Sample:</strong>", unsafe_allow_html=True)
        ai_names = [f"{s.get('category_name')}: {s.get('filename')}" for s in ai_list]
        selected_ai_idx = st.selectbox(
            "Select AI-Generated Sample",
            range(len(ai_names)),
            format_func=lambda i: ai_names[i],
            label_visibility="collapsed",
            key="comp_ai_select"
        )
        chosen_ai = ai_list[selected_ai_idx]

    # Side-by-side audio playback & details
    play_col1, play_col2 = st.columns(2, gap="medium")
    with play_col1:
        st.markdown(
            f"""
            <div style="background: rgba(16, 23, 37, 0.75); border: 1px solid #263248; border-radius: 9px; padding: 12px; margin-top: 6px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #2bd49a;">Real Sample: {chosen_real.get('filename')}</div>
                <div style="font-size: 0.72rem; color: #9aa9c1; margin-bottom: 6px;">{chosen_real.get('source')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        r_path = resolve_sample_path(chosen_real.get("filename"))
        if r_path and r_path.exists():
            st.audio(r_path.read_bytes())

    with play_col2:
        st.markdown(
            f"""
            <div style="background: rgba(16, 23, 37, 0.75); border: 1px solid #263248; border-radius: 9px; padding: 12px; margin-top: 6px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #ff6d8b;">AI-Generated Sample: {chosen_ai.get('filename')}</div>
                <div style="font-size: 0.72rem; color: #9aa9c1; margin-bottom: 6px;">{chosen_ai.get('source')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        ai_path = resolve_sample_path(chosen_ai.get("filename"))
        if ai_path and ai_path.exists():
            st.audio(ai_path.read_bytes())

    st.markdown("<br/>", unsafe_allow_html=True)

    # Comparison Trigger Button
    if st.button("📊 Analyze Both Recordings & Compare", type="primary", use_container_width=True, key="btn_run_comparison"):
        with st.spinner("Executing comparative 26-MFCC extraction and SVM predictions..."):
            comp_res = compare_two_samples(chosen_real, chosen_ai)
            st.session_state["active_comparison_result"] = comp_res

    # Display comparison if available in state
    comp_res = st.session_state.get("active_comparison_result")
    if comp_res and comp_res.get("both_ready"):
        res_a = comp_res["sample_a"]
        res_b = comp_res["sample_b"]

        st.markdown("<br/>", unsafe_allow_html=True)
        # Side-by-side prediction headlines
        head_c1, head_c2 = st.columns(2, gap="medium")
        with head_c1:
            st.markdown(
                f"""
                <div class="headline-card {'ai' if res_a['prediction_class'] == 1 else 'real'}" style="padding: 12px 14px;">
                    <div style="font-size: 0.7rem; font-weight: bold; text-transform: uppercase; color: {'#ff6d8b' if res_a['prediction_class'] == 1 else '#2bd49a'};">
                        Real Recording Prediction: Class {res_a['prediction_class']}
                    </div>
                    <div class="headline-title" style="font-size: 1.1rem; margin-top: 3px;">
                        {res_a['headline']}
                    </div>
                    <div style="font-size: 0.75rem; color: #f3f6ff; margin-top: 4px;">
                        P(AI): <strong>{res_a['ai_probability']*100:.1f}%</strong> · {res_a['status_text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with head_c2:
            st.markdown(
                f"""
                <div class="headline-card {'ai' if res_b['prediction_class'] == 1 else 'real'}" style="padding: 12px 14px;">
                    <div style="font-size: 0.7rem; font-weight: bold; text-transform: uppercase; color: {'#ff6d8b' if res_b['prediction_class'] == 1 else '#2bd49a'};">
                        AI Recording Prediction: Class {res_b['prediction_class']}
                    </div>
                    <div class="headline-title" style="font-size: 1.1rem; margin-top: 3px;">
                        {res_b['headline']}
                    </div>
                    <div style="font-size: 0.75rem; color: #f3f6ff; margin-top: 4px;">
                        P(AI): <strong>{res_b['ai_probability']*100:.1f}%</strong> · {res_b['status_text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Structured Comparison Table
        st.markdown("#### Comparison Matrix")
        table_rows = comp_res.get("comparison_rows", [])
        if table_rows:
            import pandas as pd
            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Important Comparative Limitation Notice
        st.info(
            f"**Important Comparative Notice:**\n\n{comp_res.get('disclaimer')}"
        )

