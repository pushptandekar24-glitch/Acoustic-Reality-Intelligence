"""
Report generator for Acoustic Reality Intelligence.
Generates comprehensive forensic PDF reports and CSV feature exports.
"""
import io
import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import config


def generate_plots_image_buffer(
    signal: np.ndarray,
    sample_rate: int,
    stft_db: Optional[np.ndarray] = None
) -> io.BytesIO:
    """
    Renders clean waveform and spectrogram figures into a high-resolution PNG buffer.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 3.4), dpi=150)
    fig.patch.set_facecolor("#101725")

    # Time axis
    times = np.linspace(0, len(signal) / sample_rate, num=len(signal))

    # 1. Waveform Plot
    ax1.set_facecolor("#0a0f1d")
    ax1.plot(times, signal, color="#31d2ff", linewidth=0.7, alpha=0.9)
    ax1.set_title("Audio Amplitude Waveform", color="#f3f6ff", fontsize=9, fontweight="bold", pad=4)
    ax1.set_xlabel("Time (seconds)", color="#9aa9c1", fontsize=7.5)
    ax1.set_ylabel("Amplitude", color="#9aa9c1", fontsize=7.5)
    ax1.tick_params(colors="#9aa9c1", labelsize=6.5)
    for spine in ax1.spines.values():
        spine.set_color("#263248")

    # 2. Spectrogram Plot
    ax2.set_facecolor("#0a0f1d")
    if stft_db is not None:
        im = ax2.imshow(
            stft_db,
            aspect="auto",
            origin="lower",
            extent=[0, times[-1] if len(times) > 0 else 1.0, 0, sample_rate / 2],
            cmap="magma",
        )
        ax2.set_title("STFT Magnitude Spectrogram (dB)", color="#f3f6ff", fontsize=9, fontweight="bold", pad=4)
        ax2.set_xlabel("Time (seconds)", color="#9aa9c1", fontsize=7.5)
        ax2.set_ylabel("Frequency (Hz)", color="#9aa9c1", fontsize=7.5)
        ax2.tick_params(colors="#9aa9c1", labelsize=6.5)
    for spine in ax2.spines.values():
        spine.set_color("#263248")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(
    filename: str,
    metadata: Dict[str, Any],
    lab_features: Optional[Dict[str, Any]] = None,
    authenticity_result: Optional[Dict[str, Any]] = None,
    sound_recognition_result: Optional[Dict[str, Any]] = None,
    segment_result: Optional[Dict[str, Any]] = None,
    signal: Optional[np.ndarray] = None,
    sample_rate: int = config.TARGET_SAMPLE_RATE
) -> bytes:
    """
    Generates a structured, multi-section PDF analysis report using ReportLab.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#12234e"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#4a5975"),
        spaceAfter=12,
    )
    sec_heading = ParagraphStyle(
        "SecHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1b3260"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#263248"),
    )
    callout_real = ParagraphStyle(
        "CalloutReal",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f7b57"),
    )
    callout_ai = ParagraphStyle(
        "CalloutAI",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#b81b47"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#6c7d97"),
    )

    story = []

    # Title & Header
    story.append(Paragraph(config.APP_NAME, title_style))
    story.append(Paragraph(f"{config.APP_SUBTITLE} · Technical Analysis Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3878ff"), spaceAfter=10))

    # Meta Info Table
    analysis_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_table_data = [
        [
            Paragraph("<b>Analyzed File:</b>", body_style),
            Paragraph(filename, body_style),
            Paragraph("<b>Timestamp:</b>", body_style),
            Paragraph(analysis_time, body_style),
        ],
        [
            Paragraph("<b>Duration:</b>", body_style),
            Paragraph(f"{metadata.get('duration_seconds', 0.0):.2f} sec", body_style),
            Paragraph("<b>Sample Rate:</b>", body_style),
            Paragraph(f"{metadata.get('sample_rate', 16000)} Hz", body_style),
        ],
        [
            Paragraph("<b>Channels:</b>", body_style),
            Paragraph("Mono (Resampled 16 kHz)" if metadata.get("mono") else "Multi-channel", body_style),
            Paragraph("<b>File Size:</b>", body_style),
            Paragraph(f"{metadata.get('file_size_bytes', 0) / 1024:.1f} KB", body_style),
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[90, 180, 90, 180])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f7fb")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d3dce8")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # Module E: Synthetic Audio Assessment (if available)
    if authenticity_result is not None:
        story.append(Paragraph("1. Synthetic Audio Assessment", sec_heading))
        is_ai = authenticity_result.get("is_ai_generated", False)
        headline = authenticity_result.get("headline", "")
        callout_fmt = callout_ai if is_ai else callout_real
        box_bg = colors.HexColor("#ffeef2") if is_ai else colors.HexColor("#ebfbf5")
        box_border = colors.HexColor("#ff6d8b") if is_ai else colors.HexColor("#2bd49a")

        auth_table_data = [
            [Paragraph(f"<b>Assessment Result:</b> {headline}", callout_fmt)],
            [Paragraph(
                f"<b>Model Prediction:</b> {authenticity_result.get('class_name')} · "
                f"<b>Real Probability Estimate:</b> {authenticity_result.get('real_probability', 0.0):.1%} · "
                f"<b>AI Probability Estimate:</b> {authenticity_result.get('ai_probability', 0.0):.1%}<br/>"
                f"<b>Model Fingerprint:</b> <font face='Courier' size=7>{authenticity_result.get('model_sha256', '')[:32]}...</font>",
                body_style
            )]
        ]
        t_auth = Table(auth_table_data, colWidths=[540])
        t_auth.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), box_bg),
            ("BOX", (0, 0), (-1, -1), 1.2, box_border),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(t_auth)
        story.append(Spacer(1, 8))

    # Module D: Environmental Sound Recognition (if available)
    if sound_recognition_result is not None:
        story.append(Paragraph("2. Environmental Sound-Event Recognition", sec_heading))
        top_event = sound_recognition_result.get("top_event", "N/A")
        top_score = sound_recognition_result.get("top_score", 0.0)
        top_preds = sound_recognition_result.get("top_predictions", [])

        rec_table_data = [
            [
                Paragraph("<b>Primary Detected Event:</b>", body_style),
                Paragraph(f"<b>{top_event}</b> ({top_score:.1%} confidence)", body_style),
                Paragraph("<b>Recognition Engine:</b>", body_style),
                Paragraph(sound_recognition_result.get("backend", "DSP Profiler"), body_style),
            ]
        ]
        t_rec = Table(rec_table_data, colWidths=[130, 150, 110, 150])
        t_rec.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f9fd")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d3dce8")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_rec)

        # Ranked breakdown table
        ranked_data = [["Rank", "Sound Event Category", "Likelihood Score"]]
        for rank, p in enumerate(top_preds[:4], 1):
            ranked_data.append([str(rank), p.get("event", ""), f"{p.get('score', 0.0):.1%}"])
        t_ranked = Table(ranked_data, colWidths=[50, 340, 150])
        t_ranked.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263248")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d3dce8")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(Spacer(1, 4))
        story.append(t_ranked)
        story.append(Spacer(1, 8))

    # Module C: Audio Laboratory & Acoustic Features (if available)
    if lab_features is not None:
        story.append(Paragraph("3. Acoustic Laboratory & Spectral Metrics", sec_heading))
        stats_data = lab_features.get("summary_stats", {})
        
        lab_table_data = [
            ["Acoustic Metric", "Mean Value", "Std Deviation", "Min", "Max"],
            [
                "RMS Energy",
                f"{stats_data.get('rms_energy', {}).get('mean', 0.0):.5f}",
                f"{stats_data.get('rms_energy', {}).get('std', 0.0):.5f}",
                f"{stats_data.get('rms_energy', {}).get('min', 0.0):.5f}",
                f"{stats_data.get('rms_energy', {}).get('max', 0.0):.5f}",
            ],
            [
                "Spectral Centroid (Hz)",
                f"{stats_data.get('spectral_centroid', {}).get('mean', 0.0):.1f}",
                f"{stats_data.get('spectral_centroid', {}).get('std', 0.0):.1f}",
                f"{stats_data.get('spectral_centroid', {}).get('min', 0.0):.1f}",
                f"{stats_data.get('spectral_centroid', {}).get('max', 0.0):.1f}",
            ],
            [
                "Zero-Crossing Rate",
                f"{stats_data.get('zero_crossing_rate', {}).get('mean', 0.0):.4f}",
                f"{stats_data.get('zero_crossing_rate', {}).get('std', 0.0):.4f}",
                f"{stats_data.get('zero_crossing_rate', {}).get('min', 0.0):.4f}",
                f"{stats_data.get('zero_crossing_rate', {}).get('max', 0.0):.4f}",
            ],
            [
                "Spectral Bandwidth (Hz)",
                f"{stats_data.get('spectral_bandwidth', {}).get('mean', 0.0):.1f}",
                f"{stats_data.get('spectral_bandwidth', {}).get('std', 0.0):.1f}",
                f"{stats_data.get('spectral_bandwidth', {}).get('min', 0.0):.1f}",
                f"{stats_data.get('spectral_bandwidth', {}).get('max', 0.0):.1f}",
            ],
            [
                "Spectral Flatness",
                f"{stats_data.get('spectral_flatness', {}).get('mean', 0.0):.4f}",
                f"{stats_data.get('spectral_flatness', {}).get('std', 0.0):.4f}",
                f"{stats_data.get('spectral_flatness', {}).get('min', 0.0):.4f}",
                f"{stats_data.get('spectral_flatness', {}).get('max', 0.0):.4f}",
            ]
        ]
        t_lab = Table(lab_table_data, colWidths=[160, 95, 95, 95, 95])
        t_lab.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b3260")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d3dce8")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_lab)
        story.append(Spacer(1, 8))

        # Embed Visual Spectrogram & Waveform
        if signal is not None:
            try:
                img_buf = generate_plots_image_buffer(signal, sample_rate, lab_features.get("stft_db"))
                story.append(RLImage(img_buf, width=540, height=220))
                story.append(Spacer(1, 8))
            except Exception:
                pass

    # Module F: Segment Analysis (if available)
    if segment_result is not None:
        story.append(Paragraph("4. Segment Analysis & Discontinuity Exploration", sec_heading))
        seg_summary = [
            [
                Paragraph("<b>Total Segments:</b>", body_style),
                Paragraph(str(segment_result.get("total_segments", 0)), body_style),
                Paragraph("<b>Segment Duration:</b>", body_style),
                Paragraph(f"{segment_result.get('segment_duration_sec', 0.0):.1f} s", body_style),
                Paragraph("<b>Anomalies Flagged:</b>", body_style),
                Paragraph(str(segment_result.get("anomaly_count", 0)), body_style),
            ]
        ]
        t_seg = Table(seg_summary, colWidths=[90, 80, 110, 80, 110, 70])
        t_seg.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d3dce8")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_seg)
        story.append(Spacer(1, 8))

    # Limitations & Legal / Technical Disclaimers
    story.append(Paragraph("System Disclaimers & Technical Limitations", sec_heading))
    disclaimer_text = (
        "<b>Important Interpretation Notes:</b><br/>"
        "1. <b>Experimental Nature:</b> The predictions generated by Acoustic Reality Intelligence represent "
        "experimental statistical estimates based on 13 MFCC coefficients (26 mean and standard deviation features). "
        "The model does NOT provide legally or forensically binding proof of audio authenticity.<br/>"
        "2. <b>Independent Sound Events:</b> Environmental sound recognition is strictly decoupled from synthetic sound detection. "
        "The presence of identifiable real-world sounds (e.g. rain, birdsong, vehicles) does not indicate whether "
        "a clip is genuine or artificially synthesized.<br/>"
        "3. <b>Acoustic Anomalies:</b> Segment-level acoustic anomalies highlight points of sudden energy or timbral "
        "variation for investigative purposes; they must not be presumed to be proof of malicious tampering or generative synthesis."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def generate_features_csv(
    features_vector: np.ndarray,
    metadata: Dict[str, Any],
    filename: str
) -> str:
    """
    Exports extracted acoustic features to CSV format.
    """
    feats = features_vector.flatten()
    cols = [f"mfcc_{i+1}_mean" for i in range(13)] + [f"mfcc_{i+1}_std" for i in range(13)]

    data = {
        "filename": [filename],
        "duration_sec": [metadata.get("duration_seconds", 0.0)],
        "sample_rate": [metadata.get("sample_rate", 16000)],
        "channels": [metadata.get("channels", 1)],
    }
    for col, val in zip(cols, feats):
        data[col] = [round(float(val), 6)]

    df = pd.DataFrame(data)
    return df.to_csv(index=False)

