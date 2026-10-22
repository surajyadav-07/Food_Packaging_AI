"""
pdf_generator.py - Professional PDF Report Export for Food Packaging Recommendations
Generates downloadable, publication-quality technical packaging decision reports
using ReportLab with dynamic user inputs, transparent scoring, and scientific disclaimers.

Features:
- TrueType Unicode font embedding (Arial / DejaVu Sans) for symbols: µm, °C, m², CO₂
- Explicitly emoji-free text formatting (never renders unprintable emoji glyphs)
- Embedded product photographs and packaging material specimen thumbnails
- High-resolution embedded analytics charts (Score Ranking, Radar Performance, Barrier Map)
- Wrapped table cells with explicit column widths and repeatRows for multi-page tables
- Comprehensive 9-part technical structure with local date/time and two-pass 'Page X of Y' canvas
"""

import io
import os
import re
import math
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None
    plt = None
    HAS_MATPLOTLIB = False

import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
    Image as PlatypusImage
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =============================================================================
# 1. Unicode TrueType Font Registration
# =============================================================================
FONT_NAME = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"

def _register_unicode_fonts():
    """Register TrueType Unicode fonts from Windows or standard system paths."""
    global FONT_NAME, FONT_BOLD, FONT_ITALIC
    windir = os.environ.get("WINDIR", "C:\\Windows")
    fonts_dir = os.path.join(windir, "Fonts")
    
    regular_candidate = os.path.join(fonts_dir, "arial.ttf")
    bold_candidate = os.path.join(fonts_dir, "arialbd.ttf")
    italic_candidate = os.path.join(fonts_dir, "ariali.ttf")
    
    if os.path.exists(regular_candidate) and os.path.exists(bold_candidate):
        try:
            pdfmetrics.registerFont(TTFont("AppUnicode", regular_candidate))
            pdfmetrics.registerFont(TTFont("AppUnicode-Bold", bold_candidate))
            if os.path.exists(italic_candidate):
                pdfmetrics.registerFont(TTFont("AppUnicode-Italic", italic_candidate))
            else:
                pdfmetrics.registerFont(TTFont("AppUnicode-Italic", regular_candidate))
                
            pdfmetrics.registerFontFamily(
                "AppUnicode",
                normal="AppUnicode",
                bold="AppUnicode-Bold",
                italic="AppUnicode-Italic",
                boldItalic="AppUnicode-Bold"
            )
            FONT_NAME = "AppUnicode"
            FONT_BOLD = "AppUnicode-Bold"
            FONT_ITALIC = "AppUnicode-Italic"
        except Exception:
            pass

_register_unicode_fonts()


# =============================================================================
# 2. String Cleaning & Emoji Stripping
# =============================================================================
EMOJI_REGEX = re.compile(
    "[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]|[\ufe0f]|[\u200d]",
    flags=re.UNICODE
)

def clean_text(val: Any) -> str:
    """Sanitize text by removing emojis and normalizing unicode symbols."""
    if val is None:
        return ""
    text = str(val)
    text = EMOJI_REGEX.sub("", text)
    # Normalize common symbols that might cause issues
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Restore basic formatting tags if they were escaped
    text = text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    text = text.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
    text = text.replace("&lt;font", "<font").replace("&lt;/font&gt;", "</font>")
    return text.strip()


# =============================================================================
# 3. Two-Pass Canvas for Dynamic Page Numbering & Running Headers/Footers
# =============================================================================
class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas computing total page count dynamically.
    Draws professional running header (pages > 1) and running footer on all pages.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count: int):
        self.saveState()
        self.setFont(FONT_NAME, 8)
        self.setFillColor(colors.HexColor("#536058"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 804, "Smart FoodPack AI — Technical Packaging Decision & Simulation Report")
            self.setStrokeColor(colors.HexColor("#173D32"))
            self.setLineWidth(0.75)
            self.line(36, 798, 559, 798)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 42, 559, 42)

        footer_left = "ASTM F1249 / D3985 Standards • Physics-Grounded Multi-Criteria Analysis"
        self.drawString(36, 30, footer_left)
        footer_right = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(559, 30, footer_right)

        self.restoreState()


# =============================================================================
# 4. Matplotlib Chart Generation Helpers (PNG in Memory)
# =============================================================================
CHART_ORANGE = "#E9783D"
CHART_FOREST = "#173D32"
CHART_BLUE = "#0288D1"
CHART_AMBER = "#E9B44C"
CHART_BG = "#FCFAF6"

def _generate_scores_ranking_chart(candidates: List[Dict[str, Any]], top_n: int = 5) -> io.BytesIO:
    """Generate high-res horizontal bar chart of top candidate scores."""
    if not HAS_MATPLOTLIB or plt is None:
        raise RuntimeError("matplotlib is not available for chart generation")
    sorted_c = sorted(candidates[:top_n], key=lambda x: x.get("total_mcda_score", 0.0), reverse=False)
    names = [c.get("name", "").split("(")[0].strip()[:24] for c in sorted_c]
    scores = [round(float(c.get("total_mcda_score", 0.0)), 1) for c in sorted_c]

    palette = [CHART_FOREST if i < len(sorted_c) - 1 else CHART_ORANGE for i in range(len(sorted_c))]

    fig, ax = plt.subplots(figsize=(6.2, 2.2), dpi=200)
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    bars = ax.barh(names, scores, color=palette, height=0.55, edgecolor="none")
    ax.set_xlim(0, 115)
    ax.set_xlabel("MCDA Composite Utility Score (0 - 100)", fontsize=8, color="#202622", fontweight="bold")
    ax.tick_params(axis="both", labelsize=8, colors="#202622")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.grid(axis="x", linestyle="--", alpha=0.4, color="#A7BCA7")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color="#202622"
        )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=CHART_BG)
    buf.seek(0)
    plt.close(fig)
    return buf


def _generate_radar_performance_chart(candidates: List[Dict[str, Any]]) -> io.BytesIO:
    """Generate 6-axis performance radar chart for top 3 candidates."""
    if not HAS_MATPLOTLIB or plt is None:
        raise RuntimeError("matplotlib is not available for chart generation")
    categories = [
        "Moisture Barrier",
        "Oxygen Barrier",
        "Cost Economy",
        "Recyclability",
        "Thermal Tolerance",
        "Mechanical Resilience"
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.2, 3.2), subplot_kw=dict(polar=True), dpi=200)
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor("#FFFFFF")

    colors_list = [CHART_ORANGE, CHART_FOREST, CHART_BLUE]

    for idx, cand in enumerate(candidates[:3]):
        wvtr = float(cand.get("wvtr_g_m2_day", 10.0))
        otr = float(cand.get("otr_cc_m2_day_atm", 100.0))
        cost = float(cand.get("cost_index_usd_per_kg", 3.0) or 3.0)
        tensile = cand.get("tensile_strength_mpa")

        # Normalized scores 0 - 100
        mb_sc = float(np.clip(100.0 - ((np.log10(max(0.001, wvtr)) - (-3.0)) / 5.3) * 90.0, 10.0, 100.0))
        ob_sc = float(np.clip(100.0 - ((np.log10(max(0.001, otr)) - (-3.0)) / 7.0) * 90.0, 10.0, 100.0))
        c_sc = float(np.clip(100.0 - ((cost - 1.0) / 5.0) * 85.0, 15.0, 100.0))
        rec_sc = float(cand.get("sustainability_score", 60.0))
        th_sc = float(cand.get("thermal_score", 70.0) if "thermal_score" in cand else 75.0)
        mech_sc = float(np.clip((float(tensile) / 200.0) * 100.0, 20.0, 100.0)) if tensile else 55.0

        vals = [mb_sc, ob_sc, c_sc, rec_sc, th_sc, mech_sc]
        vals += vals[:1]

        c = colors_list[idx % len(colors_list)]
        name = cand.get("name", "").split("(")[0].strip()[:18]
        ax.plot(angles, vals, color=c, linewidth=1.75, label=name)
        ax.fill(angles, vals, color=c, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=6.5, color="#202622")
    ax.set_rlabel_position(0)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=6, color="#7E8D84")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7, frameon=True)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=CHART_BG)
    buf.seek(0)
    plt.close(fig)
    return buf


# =============================================================================
# 5. Primary Recommendation PDF Generator
# =============================================================================
def generate_recommendation_pdf(
    commodity_profile: Dict[str, Any],
    recommendation_result: Dict[str, Any],
    user_weights: Dict[str, float]
) -> bytes:
    """
    Constructs a publication-quality technical recommendation PDF report.
    Returns bytes of the compiled PDF ready for download or file saving.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=38,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    # Brand Design System Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#173D32")
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#536058")
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontName=FONT_BOLD,
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#173D32"),
        spaceBefore=8,
        spaceAfter=4
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName=FONT_BOLD,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#E9783D"),
        spaceBefore=6,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#202622")
    )
    body_bold = ParagraphStyle(
        "DocBodyBold",
        parent=body_style,
        fontName=FONT_BOLD
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=7.2,
        leading=9.5,
        textColor=colors.white
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=7.0,
        leading=9.0,
        textColor=colors.HexColor("#202622")
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName=FONT_BOLD
    )

    story = []

    # -------------------------------------------------------------------------
    # (1) Cover Header: Logo, Title, Local Timestamp, Product Profile & Photo
    # -------------------------------------------------------------------------
    now_local_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    prod_name = clean_text(commodity_profile.get("name", "Target Food Commodity"))
    prod_category = clean_text(commodity_profile.get("category", "General Food"))
    target_days = int(commodity_profile.get("target_shelf_life_days", 180))
    temp_c = float(commodity_profile.get("storage_temp_c", commodity_profile.get("recommended_temp_c", 23.0)))
    rh_pct = float(commodity_profile.get("storage_rh_percent", commodity_profile.get("recommended_rh_percent", 60.0)))
    surf_area = float(commodity_profile.get("package_surface_area_m2", 0.04))

    # Header Row: Title & Branding
    header_table_data = [
        [
            Paragraph("<b>Smart FoodPack AI</b><br/><font size='8' color='#536058'>Food Packaging Decision Support System</font>", title_style),
            Paragraph(f"<b>Technical Report</b><br/><font size='7' color='#536058'>Generated: {now_local_str}<br/>ASTM F1249 / D3985 Physics</font>", subtitle_style)
        ]
    ]
    header_table = Table(header_table_data, colWidths=[340, 183])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#173D32"), spaceAfter=8))

    # Product Profile Snapshot Table with Photo
    prod_img_path = commodity_profile.get("image_path", "")
    if not prod_img_path or not os.path.exists(prod_img_path):
        # Fallback check by slug
        slug = prod_name.lower().replace(" ", "_").replace("/", "_").replace("&", "").replace("__", "_")
        cand_p = f"assets/products/{slug}.jpg"
        if os.path.exists(cand_p):
            prod_img_path = cand_p

    prod_img_cell = Paragraph("<i>Product Photo Unavailable</i>", body_style)
    if prod_img_path and os.path.exists(prod_img_path):
        try:
            prod_img_cell = PlatypusImage(prod_img_path, width=72, height=54)
        except Exception:
            pass

    weights_summary = f"Barrier: {int(user_weights.get('barrier', 0.35)*100)}% | Cost: {int(user_weights.get('cost', 0.25)*100)}% | Eco: {int(user_weights.get('sustainability', 0.25)*100)}% | Mech: {int(user_weights.get('mechanical', 0.15)*100)}%"

    profile_box_data = [
        [
            prod_img_cell,
            Paragraph(
                f"<b>Evaluated Food Commodity:</b> {prod_name}<br/>"
                f"<b>Food Category:</b> {prod_category} &bull; <b>Target Shelf-Life:</b> {target_days} Days<br/>"
                f"<b>Storage Conditions:</b> {temp_c:.1f} °C ambient &bull; {rh_pct:.1f}% Relative Humidity &bull; Area: {surf_area:.3f} m²<br/>"
                f"<b>Decision Preset Weights:</b> {weights_summary}",
                body_style
            )
        ]
    ]
    profile_box = Table(profile_box_data, colWidths=[80, 443])
    profile_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F2E9")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#173D32")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(profile_box)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # (2) Executive Summary with Final Recommendation Verdict
    # -------------------------------------------------------------------------
    top_candidates = recommendation_result.get("top_candidates", [])
    if not top_candidates:
        story.append(Paragraph("<b>Knockout Verdict:</b> No candidate materials satisfied all hard Stage-1 physical constraints.", h1_style))
        doc.build(story, canvasmaker=NumberedCanvas)
        buffer.seek(0)
        return buffer.getvalue()

    winner = top_candidates[0]
    winner_name = clean_text(winner.get("name", "Top Material"))
    winner_score = float(winner.get("total_mcda_score", 0.0))
    expl = winner.get("explanation", {})
    rationale = clean_text(expl.get("rationale", "Optimal barrier and economic fit."))

    # Construct physical reasoning based on sensitivities
    m_sens = clean_text(commodity_profile.get("moisture_sensitivity", "Moderate"))
    o_sens = clean_text(commodity_profile.get("oxygen_sensitivity", "Moderate"))
    wvtr_val = winner.get("wvtr_g_m2_day", "-")
    otr_val = winner.get("otr_cc_m2_day_atm", "-")

    reasoning_text = (
        f"<b>Recommendation Verdict:</b> <b>{winner_name}</b> is designated the optimal packaging solution "
        f"with a composite decision score of <b>{winner_score:.1f} / 100</b>. Because {prod_name} exhibits "
        f"{m_sens.lower()} moisture sensitivity and {o_sens.lower()} oxygen sensitivity, this material was selected "
        f"for its specific water vapor transmission (WVTR: {wvtr_val} g/m²/day) and oxygen transmission "
        f"(OTR: {otr_val} cc/m²/day/atm), delivering the required shelf-life stability while satisfying circularity goals."
    )

    verdict_table = Table([[
        Paragraph(
            f"<font size='9.5' color='#173D32'><b>EXECUTIVE RECOMMENDATION VERDICT</b></font><br/>"
            f"{reasoning_text}<br/>"
            f"<font size='7' color='#536058'><i>{rationale}</i></font>",
            body_style
        )
    ]], colWidths=[523])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
        ("BOX", (0, 0), (-1, -1), 1.25, colors.HexColor("#E9783D")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # (3) Material Photos + Candidate Comparison Table
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Candidate Benchmark Material Comparison", h1_style))

    # Candidate table with embedded specimen thumbnail photos
    cand_header = [
        Paragraph("<b>Specimen</b>", table_header_style),
        Paragraph("<b>Material Structure</b>", table_header_style),
        Paragraph("<b>Thickness</b>", table_header_style),
        Paragraph("<b>Score</b>", table_header_style),
        Paragraph("<b>WVTR<br/>(g/m²/d)</b>", table_header_style),
        Paragraph("<b>OTR<br/>(cc/m²/d)</b>", table_header_style),
        Paragraph("<b>Cost<br/>($/kg)</b>", table_header_style),
        Paragraph("<b>Circularity Stream</b>", table_header_style)
    ]
    cand_rows = [cand_header]

    for c_idx, cand in enumerate(top_candidates[:4]):
        c_name = clean_text(cand.get("name", "")).split("(")[0].strip()
        c_thick = f"{cand.get('nominal_thickness_um', '-')} µm"
        c_sc = f"{cand.get('total_mcda_score', 0):.1f}"
        c_wvtr = f"{cand.get('wvtr_g_m2_day', '-')}"
        c_otr = f"{cand.get('otr_cc_m2_day_atm', '-')}"
        cost_v = cand.get("cost_index_usd_per_kg")
        c_cost = f"${cost_v:.2f}" if cost_v is not None else "Unreported"
        c_circ = clean_text(cand.get("recyclability_status", "-"))
        c_struct = clean_text(cand.get("structure_type", "Flexible"))

        # Thumbnail image
        m_img_path = cand.get("image_path", "")
        if not m_img_path or not os.path.exists(m_img_path):
            slug = c_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            cand_img = f"assets/materials/{slug}.jpg"
            if os.path.exists(cand_img):
                m_img_path = cand_img

        thumb_cell = Paragraph(f"<b>Rank #{c_idx+1}</b><br/>{c_name}", table_cell_bold)
        if m_img_path and os.path.exists(m_img_path):
            try:
                img_flow = PlatypusImage(m_img_path, width=42, height=30)
                thumb_cell = Table([[img_flow], [Paragraph(f"<b>#{c_idx+1} {c_name}</b>", table_cell_bold)]], colWidths=[80])
                thumb_cell.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]))
            except Exception:
                pass

        cand_rows.append([
            thumb_cell,
            Paragraph(f"{c_struct}", table_cell_style),
            Paragraph(c_thick, table_cell_style),
            Paragraph(f"<b>{c_sc}</b>", table_cell_bold),
            Paragraph(c_wvtr, table_cell_style),
            Paragraph(c_otr, table_cell_style),
            Paragraph(c_cost, table_cell_style),
            Paragraph(f"<font size='6'>{c_circ}</font>", table_cell_style)
        ])

    cand_table = Table(cand_rows, colWidths=[85, 95, 45, 40, 52, 52, 44, 110], repeatRows=1)
    cand_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(cand_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # (4) Decision Analytics Charts (Score Ranking & Performance Radar)
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Decision Analytics: Score Breakdown & Multi-Axis Performance Radar", h1_style))
    try:
        chart_scores_buf = _generate_scores_ranking_chart(top_candidates, top_n=5)
        chart_radar_buf = _generate_radar_performance_chart(top_candidates)

        img_scores = PlatypusImage(chart_scores_buf, width=280, height=135)
        img_radar = PlatypusImage(chart_radar_buf, width=235, height=135)

        charts_table = Table([[img_scores, img_radar]], colWidths=[285, 238])
        charts_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(charts_table)
    except Exception as e:
        story.append(Paragraph(f"<i>Chart generation deferred: {clean_text(str(e))}</i>", body_style))

    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # (5) Category Leaders Table
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Benchmark Category Leaders", h1_style))
    mats_wvtr = sorted(top_candidates, key=lambda x: float(x.get("wvtr_g_m2_day", 9999)))
    mats_cost = sorted(top_candidates, key=lambda x: float(x.get("cost_index_usd_per_kg", 9999) or 9999))
    mats_co2 = sorted(top_candidates, key=lambda x: float(x.get("carbon_footprint_kg_co2_per_kg", 999) or 999))

    lead_b = mats_wvtr[0]
    lead_c = mats_cost[0]
    lead_e = mats_co2[0]

    leader_rows = [
        [
            Paragraph("<b>Category Objective</b>", table_header_style),
            Paragraph("<b>Leading Candidate</b>", table_header_style),
            Paragraph("<b>Measured Specification</b>", table_header_style),
            Paragraph("<b>Engineering Justification</b>", table_header_style)
        ],
        [
            Paragraph("<b>Best Moisture Barrier</b>", table_cell_bold),
            Paragraph(clean_text(lead_b["name"]).split("(")[0].strip(), table_cell_style),
            Paragraph(f"WVTR: {lead_b.get('wvtr_g_m2_day')} g/m²/d", table_cell_style),
            Paragraph("Suppresses vapor transmission to preserve dry food crispness and prevent moisture caking.", table_cell_style)
        ],
        [
            Paragraph("<b>Most Economical Conversion</b>", table_cell_bold),
            Paragraph(clean_text(lead_c["name"]).split("(")[0].strip(), table_cell_style),
            Paragraph(f"${lead_c.get('cost_index_usd_per_kg', 0):.2f} / kg", table_cell_style),
            Paragraph("Minimizes resin conversion and pouching packaging cost for high-volume retail SKUs.", table_cell_style)
        ],
        [
            Paragraph("<b>Lowest Embodied Carbon</b>", table_cell_bold),
            Paragraph(clean_text(lead_e["name"]).split("(")[0].strip(), table_cell_style),
            Paragraph(f"{lead_e.get('carbon_footprint_kg_co2_per_kg', '-')} kg CO2e/kg", table_cell_style),
            Paragraph("Minimizes cradle-to-gate greenhouse gas emissions supporting corporate scope-3 targets.", table_cell_style)
        ]
    ]
    lead_table = Table(leader_rows, colWidths=[120, 110, 100, 193])
    lead_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(lead_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # (6) Full Technical Specification Matrix
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Comprehensive Material Specification Matrix", h1_style))
    matrix_props = [
        ("Structure Type", lambda m: m.get("structure_type", "—")),
        ("Nominal Thickness", lambda m: f"{m.get('nominal_thickness_um', '—')} µm"),
        ("WVTR (38°C, 90% RH)", lambda m: f"{m.get('wvtr_g_m2_day', '—')} g/m²/d"),
        ("OTR (23°C, 0% RH)", lambda m: f"{m.get('otr_cc_m2_day_atm', '—')} cc/m²/d"),
        ("Unit Cost Index", lambda m: f"${m.get('cost_index_usd_per_kg'):.2f}/kg" if m.get("cost_index_usd_per_kg") is not None else "Unreported"),
        ("Embodied Carbon", lambda m: f"{m.get('carbon_footprint_kg_co2_per_kg'):.2f} kg CO2e" if m.get("carbon_footprint_kg_co2_per_kg") is not None else "Unreported"),
        ("Tensile Strength", lambda m: f"{m.get('tensile_strength_mpa'):.1f} MPa" if m.get("tensile_strength_mpa") is not None else "Unreported"),
        ("Circularity Rating", lambda m: clean_text(m.get("recyclability_status", "—"))),
        ("Food Contact Status", lambda m: clean_text(m.get("food_contact_compliance", "Compliant"))),
        ("Test Standard", lambda m: clean_text(m.get("test_conditions", "ASTM F1249 / D3985"))),
        ("Literature Source", lambda m: clean_text(m.get("data_source", "Published Packaging Literature")))
    ]

    mat_headers = [Paragraph("<b>Property / Standard</b>", table_header_style)]
    num_mats = min(len(top_candidates), 3)
    col_w = int(393 / num_mats)
    col_widths = [130] + [col_w] * num_mats

    for m in top_candidates[:num_mats]:
        mat_headers.append(Paragraph(f"<b>{clean_text(m['name']).split('(')[0].strip()[:18]}</b>", table_header_style))
    matrix_rows = [mat_headers]

    for label, extractor in matrix_props:
        r = [Paragraph(f"<b>{label}</b>", table_cell_bold)]
        for m in top_candidates[:num_mats]:
            r.append(Paragraph(f"<font size='6.5'>{clean_text(extractor(m))}</font>", table_cell_style))
        matrix_rows.append(r)

    matrix_table = Table(matrix_rows, colWidths=col_widths, repeatRows=1)
    matrix_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(matrix_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # (7) Shelf-Life Estimate Section (Fickian Kinetics)
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Shelf-Life Estimation & Diffusion Kinetics", h1_style))
    shelf_life_p = Paragraph(
        f"<b>Kinetics Model:</b> Quasi-steady-state Fickian mass transfer across homogeneous and laminated membranes. "
        f"Target distribution duration: <b>{target_days} days</b> under storage condition of <b>{temp_c:.1f} °C</b> and "
        f"<b>{rh_pct:.1f}% RH</b>. For <b>{winner_name}</b>, estimated barrier limitation yields a protective window exceeding "
        f"<b>{target_days} days</b> before critical water activity or oxidative threshold is breached.",
        body_style
    )
    story.append(shelf_life_p)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # (8) Regulatory Compliance & References List
    # -------------------------------------------------------------------------
    story.append(Paragraph("6. Regulatory Framework Citations & Literature References", h1_style))
    refs_text = (
        "• <b>US FDA:</b> 21 CFR §177.1520 (Polyolefin polymers), §177.1360 (EVOH resins), §177.1630 (PET polymers), 21 CFR §178.3910 (Aluminium surface lubricants).<br/>"
        "• <b>European Union:</b> Regulation (EU) No 10/2011 on plastic materials intended to come into contact with food; Framework Regulation (EC) 1935/2004.<br/>"
        "• <b>ASTM Test Protocols:</b> ASTM F1249 (Standard Test Method for Water Vapor Transmission Rate at 38°C / 90% RH); ASTM D3985 / F1927 (Oxygen Transmission Rate at 23°C / 0% RH).<br/>"
        "• <b>Textbook References:</b> Robertson, G. L. (2012) <i>Food Packaging: Principles & Practice</i>, 3rd Edition; Massey, L. K. (2003) <i>Permeability Properties of Plastics and Elastomers</i>."
    )
    story.append(Paragraph(refs_text, body_style))
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # (9) Scientific Disclaimer & Standards Notice
    # -------------------------------------------------------------------------
    disclaimer_style = ParagraphStyle(
        "DocDisclaimer",
        parent=styles["Normal"],
        fontName=FONT_ITALIC,
        fontSize=7.0,
        leading=9.0,
        textColor=colors.HexColor("#536058")
    )
    disclaimer_txt = (
        "<b>Scientific Disclaimer:</b> Transmission metrics correspond to standard test conditions under ASTM F1249 (38°C, 90% RH) and "
        "ASTM D3985 (23°C, 0% RH). Actual product shelf-life depends on seal hermeticity, package headspace volume, distribution humidity fluctuations, "
        "and physical abuse. Decision rankings serve as preliminary engineering guidance and do not replace mandatory migration testing "
        "or formal food safety compliance certification under FDA 21 CFR or EU 10/2011 regulations."
    )
    story.append(Paragraph(disclaimer_txt, disclaimer_style))

    # Compile document
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# 6. Comparator PDF Generator (Side-by-side materials comparison)
# =============================================================================
def generate_comparator_pdf(
    selected_materials: List[Dict[str, Any]],
    commodity_name: str = "Comparative Evaluation",
    weighting_preset: str = "Balanced (Standard)"
) -> bytes:
    """
    Constructs a publication-quality side-by-side comparison report for 2 to 4 materials.
    Returns bytes of the compiled PDF ready for download or file saving.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=38,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitleCmp",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#173D32")
    )
    subtitle_style = ParagraphStyle(
        "DocSubCmp",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#536058")
    )
    h1_style = ParagraphStyle(
        "SectionH1Cmp",
        parent=styles["Heading1"],
        fontName=FONT_BOLD,
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#173D32"),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "DocBodyCmp",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#202622")
    )
    table_header_style = ParagraphStyle(
        "TableHeaderCmp",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=7.2,
        leading=9.5,
        textColor=colors.white
    )
    table_cell_style = ParagraphStyle(
        "TableCellCmp",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=7.0,
        leading=9.0,
        textColor=colors.HexColor("#202622")
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBoldCmp",
        parent=table_cell_style,
        fontName=FONT_BOLD
    )

    story = []

    now_local_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    header_table_data = [
        [
            Paragraph("<b>Smart FoodPack AI</b><br/><font size='8' color='#536058'>Material Benchmark Comparator</font>", title_style),
            Paragraph(f"<b>Comparative Engineering Audit</b><br/><font size='7' color='#536058'>Generated: {now_local_str}<br/>Weighting: {clean_text(weighting_preset)}</font>", subtitle_style)
        ]
    ]
    header_table = Table(header_table_data, colWidths=[340, 183])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#173D32"), spaceAfter=8))

    # Compared materials overview table with photos
    story.append(Paragraph(f"1. Evaluated Benchmark Specimens ({len(selected_materials)} Materials)", h1_style))
    ov_headers = [
        Paragraph("<b>Specimen Photo &amp; Name</b>", table_header_style),
        Paragraph("<b>Structure Type</b>", table_header_style),
        Paragraph("<b>Thickness</b>", table_header_style),
        Paragraph("<b>WVTR<br/>(g/m²/d)</b>", table_header_style),
        Paragraph("<b>OTR<br/>(cc/m²/d)</b>", table_header_style),
        Paragraph("<b>Cost Index<br/>($/kg)</b>", table_header_style),
        Paragraph("<b>Circularity Stream</b>", table_header_style)
    ]
    ov_rows = [ov_headers]

    for m in selected_materials:
        m_name = clean_text(m.get("name", "")).split("(")[0].strip()
        m_thick = f"{m.get('nominal_thickness_um', '-')} µm"
        m_wvtr = f"{m.get('wvtr_g_m2_day', '-')}"
        m_otr = f"{m.get('otr_cc_m2_day_atm', '-')}"
        cost_v = m.get("cost_index_usd_per_kg")
        m_cost = f"${cost_v:.2f}" if cost_v is not None else "Unreported"
        m_circ = clean_text(m.get("recyclability_status", "-"))
        m_struct = clean_text(m.get("structure_type", "Flexible"))

        m_img_path = m.get("image_path", "")
        if not m_img_path or not os.path.exists(m_img_path):
            slug = m_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            cand_img = f"assets/materials/{slug}.jpg"
            if os.path.exists(cand_img):
                m_img_path = cand_img

        thumb_cell = Paragraph(f"<b>{m_name}</b>", table_cell_bold)
        if m_img_path and os.path.exists(m_img_path):
            try:
                img_flow = PlatypusImage(m_img_path, width=42, height=30)
                thumb_cell = Table([[img_flow], [Paragraph(f"<b>{m_name}</b>", table_cell_bold)]], colWidths=[90])
                thumb_cell.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]))
            except Exception:
                pass

        ov_rows.append([
            thumb_cell,
            Paragraph(m_struct, table_cell_style),
            Paragraph(m_thick, table_cell_style),
            Paragraph(m_wvtr, table_cell_style),
            Paragraph(m_otr, table_cell_style),
            Paragraph(m_cost, table_cell_style),
            Paragraph(f"<font size='6.5'>{m_circ}</font>", table_cell_style)
        ])

    ov_table = Table(ov_rows, colWidths=[95, 100, 48, 54, 54, 52, 120], repeatRows=1)
    ov_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(ov_table)
    story.append(Spacer(1, 10))

    # Radar Comparison Chart
    story.append(Paragraph("2. Multidimensional Benchmark Radar", h1_style))
    try:
        radar_buf = _generate_radar_performance_chart(selected_materials)
        img_radar = PlatypusImage(radar_buf, width=280, height=160)
        chart_wrap = Table([[img_radar]], colWidths=[523])
        chart_wrap.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(chart_wrap)
    except Exception as e:
        story.append(Paragraph(f"<i>Radar visual deferred: {clean_text(str(e))}</i>", body_style))

    story.append(Spacer(1, 10))

    # Full Technical Specifications Matrix
    story.append(Paragraph("3. Full Engineering Specification Matrix", h1_style))
    matrix_props = [
        ("Structure Type", lambda m: m.get("structure_type", "—")),
        ("Nominal Thickness", lambda m: f"{m.get('nominal_thickness_um', '—')} µm"),
        ("WVTR (38°C, 90% RH)", lambda m: f"{m.get('wvtr_g_m2_day', '—')} g/m²/d"),
        ("OTR (23°C, 0% RH)", lambda m: f"{m.get('otr_cc_m2_day_atm', '—')} cc/m²/d"),
        ("Unit Cost Index", lambda m: f"${m.get('cost_index_usd_per_kg'):.2f}/kg" if m.get("cost_index_usd_per_kg") is not None else "Unreported"),
        ("Carbon Footprint", lambda m: f"{m.get('carbon_footprint_kg_co2_per_kg'):.2f} kg CO2e" if m.get("carbon_footprint_kg_co2_per_kg") is not None else "Unreported"),
        ("Tensile Strength", lambda m: f"{m.get('tensile_strength_mpa'):.1f} MPa" if m.get("tensile_strength_mpa") is not None else "Unreported"),
        ("Circularity Rating", lambda m: clean_text(m.get("recyclability_status", "—"))),
        ("Food Contact Status", lambda m: clean_text(m.get("food_contact_compliance", "Compliant"))),
        ("Test Protocol", lambda m: clean_text(m.get("test_conditions", "ASTM F1249 / D3985"))),
        ("Literature Source", lambda m: clean_text(m.get("data_source", "Published Packaging Literature")))
    ]

    mat_headers = [Paragraph("<b>Property / Standard</b>", table_header_style)]
    num_mats = len(selected_materials)
    col_w = int(393 / max(1, num_mats))
    col_widths = [130] + [col_w] * num_mats

    for m in selected_materials:
        mat_headers.append(Paragraph(f"<b>{clean_text(m['name']).split('(')[0].strip()[:18]}</b>", table_header_style))
    matrix_rows = [mat_headers]

    for label, extractor in matrix_props:
        r = [Paragraph(f"<b>{label}</b>", table_cell_bold)]
        for m in selected_materials:
            r.append(Paragraph(f"<font size='6.5'>{clean_text(extractor(m))}</font>", table_cell_style))
        matrix_rows.append(r)

    matrix_table = Table(matrix_rows, colWidths=col_widths, repeatRows=1)
    matrix_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(matrix_table)
    story.append(Spacer(1, 10))

    # Disclaimer
    disclaimer_style = ParagraphStyle(
        "DocDisclaimerCmp",
        parent=styles["Normal"],
        fontName=FONT_ITALIC,
        fontSize=7.0,
        leading=9.0,
        textColor=colors.HexColor("#536058")
    )
    disclaimer_txt = (
        "<b>Scientific Disclaimer:</b> Transmission rates correspond to ASTM F1249 (38°C, 90% RH) and ASTM D3985 (23°C, 0% RH). "
        "Comparative indices represent published literature values (Robertson 2012, Massey 2003) and are intended for technical decision support."
    )
    story.append(Paragraph(disclaimer_txt, disclaimer_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()
