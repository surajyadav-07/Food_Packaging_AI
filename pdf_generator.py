"""
pdf_generator.py - Professional PDF Report Export for Food Packaging Recommendations
Generates downloadable, publication-quality technical packaging decision reports
using ReportLab with dynamic user inputs, transparent scoring, and scientific disclaimers.
Follows a clean 2-page executive format:
  Page 1: Recommendation Summary (Food Profile, Priorities, Rank #1 Winner Card, Scores & Specs)
  Page 2: Comparison & Limitations (Side-by-side Table, Trade-offs, Limitations, Standards & Disclaimers)
"""

import io
from datetime import datetime, timezone
from typing import Dict, Any, List

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute total page count and render
    consistent running headers and footers on every page.
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
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 762, "Food Packaging AI — Technical Packaging Decision Report")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 38, 576, 38)

        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 26, footer_text)
        disclaimer_short = "Physics-Grounded Multi-Criteria Analysis • ASTM F1249 / D3985 Test Standards"
        self.drawString(36, 26, disclaimer_short)

        self.restoreState()


def generate_recommendation_pdf(
    commodity_profile: Dict[str, Any],
    recommendation_result: Dict[str, Any],
    user_weights: Dict[str, float]
) -> bytes:
    """
    Constructs an executive 2-page technical recommendation PDF from current analysis results.
    Returns:
        bytes: in-memory PDF binary ready for download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=44
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#0f172a")
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#0f766e")
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=7,
        spaceAfter=4
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=4,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1e293b")
    )

    body_bold = ParagraphStyle(
        "Body_Bold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=8,
        firstLineIndent=-5,
        spaceBefore=1,
        spaceAfter=1
    )

    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=body_style,
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#475569")
    )

    story = []

    # =========================================================================
    # PAGE 1: RECOMMENDATION SUMMARY
    # =========================================================================

    # Title & Metadata
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph("Food Packaging AI", title_style))
    story.append(Paragraph(f"Technical Packaging Recommendation & Material Decision Summary • Generated: {now_str}", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0d9488"), spaceAfter=8))

    # Section 1: Food Commodity Profile & Storage
    story.append(Paragraph("1. Food Commodity Profile & Storage Conditions", h1_style))

    comm_name = commodity_profile.get("name", "Custom Commodity")
    comm_cat = commodity_profile.get("category", "General Food")
    m_sens = commodity_profile.get("moisture_sensitivity", "Moderate")
    o_sens = commodity_profile.get("oxygen_sensitivity", "Moderate")
    fat_lvl = commodity_profile.get("fat_content_level", "Low")
    light_sens = commodity_profile.get("light_sensitivity", "Low")
    t_store = commodity_profile.get("storage_temp_c", 23.0)
    rh_store = commodity_profile.get("storage_rh_percent", 60)
    shelf_life_target = commodity_profile.get("target_shelf_life_days", 90)
    punc_risk = commodity_profile.get("puncture_risk", "Low")

    comm_table_data = [
        [
            Paragraph("<b>Commodity:</b>", body_style), Paragraph(str(comm_name), body_style),
            Paragraph("<b>Category:</b>", body_style), Paragraph(str(comm_cat), body_style),
            Paragraph("<b>Target Shelf Life:</b>", body_style), Paragraph(f"{shelf_life_target} Days", body_style)
        ],
        [
            Paragraph("<b>Moisture Sensitivity:</b>", body_style), Paragraph(str(m_sens), body_style),
            Paragraph("<b>Oxygen Sensitivity:</b>", body_style), Paragraph(str(o_sens), body_style),
            Paragraph("<b>Lipid / Fat Level:</b>", body_style), Paragraph(str(fat_lvl), body_style)
        ],
        [
            Paragraph("<b>Storage Temp:</b>", body_style), Paragraph(f"{t_store} °C", body_style),
            Paragraph("<b>Storage Humidity:</b>", body_style), Paragraph(f"{rh_store} % RH", body_style),
            Paragraph("<b>Puncture Hazard:</b>", body_style), Paragraph(str(punc_risk), body_style)
        ]
    ]

    comm_table = Table(comm_table_data, colWidths=[90, 90, 90, 90, 90, 90])
    comm_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(comm_table)
    story.append(Spacer(1, 6))

    # Section 2: User Decision Weights & Filter Audit
    story.append(Paragraph("2. Decision Preferences & Evaluation Scope", h1_style))

    w_b = user_weights.get("barrier", 0.40)
    w_c = user_weights.get("cost", 0.25)
    w_s = user_weights.get("sustainability", 0.25)
    w_m = user_weights.get("mechanical", 0.10)
    w_sum = w_b + w_c + w_s + w_m or 1.0

    tot_eval = recommendation_result.get("total_evaluated", 0)
    tot_qual = recommendation_result.get("total_qualified", 0)
    tot_knock = recommendation_result.get("total_knocked_out", 0)

    weights_data = [
        [
            Paragraph(f"<b>Barrier:</b> {w_b/w_sum*100:.0f}%", body_style),
            Paragraph(f"<b>Cost:</b> {w_c/w_sum*100:.0f}%", body_style),
            Paragraph(f"<b>Circularity:</b> {w_s/w_sum*100:.0f}%", body_style),
            Paragraph(f"<b>Mechanical:</b> {w_m/w_sum*100:.0f}%", body_style),
            Paragraph(f"<b>Feasible:</b> {tot_qual} / {tot_eval}", body_style),
            Paragraph(f"<b>Knocked Out:</b> {tot_knock}", body_style)
        ]
    ]
    weights_table = Table(weights_data, colWidths=[90, 90, 90, 90, 90, 90])
    weights_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdfa")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#99f6e4")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccfbf1")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(weights_table)
    story.append(Spacer(1, 8))

    # Section 3: Recommended Material (Rank #1)
    story.append(Paragraph("3. Primary Recommended Material (Top Ranked Candidate)", h1_style))

    top_candidates = recommendation_result.get("top_candidates", [])
    if not top_candidates:
        story.append(Paragraph(
            "<b>No materials met all physical barrier and safety constraints.</b><br/>"
            "All candidate materials in the database were disqualified during Stage-1 physics knockout filters "
            "(e.g., WVTR/OTR permeation threshold exceeded for target shelf-life or temperature limit exceeded). "
            "Consider reducing target shelf-life or lowering storage temperature.",
            body_style
        ))
    else:
        top_cand = top_candidates[0]
        c_name = top_cand.get("name", "Unknown Material")
        mcda = top_cand.get("total_mcda_score", 0.0)
        b_sc = top_cand.get("barrier_score", 0.0)
        c_sc = top_cand.get("cost_score", 0.0)
        s_sc = top_cand.get("sustainability_score", 0.0)
        m_sc = top_cand.get("mechanical_score", 0.0)
        cost_v = top_cand.get("cost_index_usd_per_kg", 0.0)
        co2_fp = top_cand.get("carbon_footprint_kg_co2_per_kg")
        co2_disp = f"{co2_fp:.2f} kg CO2e/kg" if co2_fp is not None else "Unreported in literature"
        recyc = top_cand.get("recyclability_status", "Unspecified")
        compl = top_cand.get("food_contact_compliance", "Compliant")
        expl = top_cand.get("explanation", {})

        # Winner Headline Box
        winner_card_data = [
            [
                Paragraph(f"<b>RECOMMENDED: {c_name}</b>", ParagraphStyle("WinTitle", parent=h1_style, fontSize=12, textColor=colors.HexColor("#0f766e"))),
                Paragraph(f"<b>Overall Score: {mcda:.1f} / 100</b>", ParagraphStyle("WinScore", parent=h1_style, fontSize=12, alignment=2, textColor=colors.HexColor("#0f172a")))
            ]
        ]
        winner_card = Table(winner_card_data, colWidths=[360, 180])
        winner_card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ecfdf5")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#059669")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(winner_card)
        story.append(Spacer(1, 4))

        # Score Breakdown Grid
        score_grid_data = [
            [
                Paragraph("<b>Barrier Score</b>", body_style),
                Paragraph("<b>Cost Economy</b>", body_style),
                Paragraph("<b>Circularity</b>", body_style),
                Paragraph("<b>Mechanical Resilience</b>", body_style)
            ],
            [
                Paragraph(f"<b>{b_sc:.1f}</b> / 100", body_style),
                Paragraph(f"<b>{c_sc:.1f}</b> / 100", body_style),
                Paragraph(f"<b>{s_sc:.1f}</b> / 100", body_style),
                Paragraph(f"<b>{m_sc:.1f}</b> / 100", body_style)
            ]
        ]
        score_grid = Table(score_grid_data, colWidths=[135, 135, 135, 135])
        score_grid.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#ffffff")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story.append(score_grid)
        story.append(Spacer(1, 5))

        # Material Specifications Table
        spec_data = [
            [
                Paragraph("<b>Structure:</b>", body_style), Paragraph(str(top_cand.get("structure_type", "Flexible")), body_style),
                Paragraph("<b>Nominal Gauge:</b>", body_style), Paragraph(f"{top_cand.get('nominal_thickness_um', '-')} µm", body_style),
                Paragraph("<b>WVTR (38°C/90%RH):</b>", body_style), Paragraph(f"{top_cand.get('wvtr_g_m2_day', '-')} g/m²/d", body_style)
            ],
            [
                Paragraph("<b>OTR (23°C/0%RH):</b>", body_style), Paragraph(f"{top_cand.get('otr_cc_m2_day_atm', '-')} cc/m²/d", body_style),
                Paragraph("<b>Cost Index:</b>", body_style), Paragraph(f"${cost_v:.2f} / kg", body_style),
                Paragraph("<b>Carbon Footprint:</b>", body_style), Paragraph(co2_disp, body_style)
            ],
            [
                Paragraph("<b>Recycling Stream:</b>", body_style), Paragraph(str(recyc), body_style),
                Paragraph("<b>Food Contact:</b>", body_style), Paragraph(str(compl), body_style),
                Paragraph("<b>Standard Method:</b>", body_style), Paragraph(str(top_cand.get("test_conditions", "ASTM F1249/D3985")), body_style)
            ]
        ]
        spec_table = Table(spec_data, colWidths=[90, 90, 90, 90, 90, 90])
        spec_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story.append(spec_table)
        story.append(Spacer(1, 6))

        # Concise Recommendation Rationale
        rationale_text = expl.get("rationale", "Optimal candidate satisfying barrier, cost, and circularity criteria.")
        story.append(Paragraph("<b>Recommendation Rationale:</b>", h2_style))
        story.append(Paragraph(f"<i>{rationale_text}</i>", body_style))
        story.append(Spacer(1, 4))

        # Requirements Satisfied
        reqs = expl.get("requirements_met", [])
        if reqs:
            story.append(Paragraph("<b>Key Requirements Satisfied:</b>", h2_style))
            for r in reqs[:4]:  # Show top 4 key requirements to keep page 1 clean
                story.append(Paragraph(f"• {r}", bullet_style))

    # =========================================================================
    # PAGE 2: COMPARISON, TRADEOFFS, AND TECHNICAL LIMITATIONS
    # =========================================================================
    if top_candidates:
        story.append(PageBreak())

        story.append(Paragraph("4. Side-by-Side Comparison of Eligible Candidates", h1_style))

        comp_headers = [
            Paragraph("<b>Rank</b>", body_bold),
            Paragraph("<b>Candidate Material</b>", body_bold),
            Paragraph("<b>Score</b>", body_bold),
            Paragraph("<b>Gauge</b>", body_bold),
            Paragraph("<b>WVTR</b><br/><font size='6.5'>g/m²/d</font>", body_bold),
            Paragraph("<b>OTR</b><br/><font size='6.5'>cc/m²/d</font>", body_bold),
            Paragraph("<b>Cost</b><br/><font size='6.5'>$/kg</font>", body_bold),
            Paragraph("<b>Circularity</b>", body_bold),
            Paragraph("<b>Compliance</b>", body_bold)
        ]
        comp_table_rows = [comp_headers]

        for cand in top_candidates[:5]:  # Top 5 candidates
            c_cost = cand.get("cost_index_usd_per_kg")
            c_cost_str = f"${c_cost:.2f}" if c_cost is not None else "Unreported"
            comp_table_rows.append([
                Paragraph(f"#{cand.get('rank', '-')}", body_bold),
                Paragraph(cand.get("name", ""), body_style),
                Paragraph(f"<b>{cand.get('total_mcda_score', 0):.1f}</b>", body_style),
                Paragraph(f"{cand.get('nominal_thickness_um', '-')} µm", body_style),
                Paragraph(str(cand.get("wvtr_g_m2_day", "-")), body_style),
                Paragraph(str(cand.get("otr_cc_m2_day_atm", "-")), body_style),
                Paragraph(c_cost_str, body_style),
                Paragraph(f"<font size='7'>{cand.get('recyclability_status', '-')}</font>", body_style),
                Paragraph(f"<font size='6.5'>{cand.get('food_contact_compliance', '-')}</font>", body_style)
            ])

        comp_summary_table = Table(comp_table_rows, colWidths=[28, 120, 36, 40, 42, 42, 42, 95, 95])
        comp_summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story.append(comp_summary_table)
        story.append(Spacer(1, 8))

        # Comparative Trade-Off Analysis ("Why #1 Won")
        tradeoffs = recommendation_result.get("leading_tradeoffs", {})
        if tradeoffs and tradeoffs.get("headline"):
            story.append(Paragraph(f"5. Comparative Trade-Off Analysis: {tradeoffs.get('headline')}", h1_style))
            if tradeoffs.get("verdict"):
                story.append(Paragraph(f"<b>Decision Verdict:</b> {tradeoffs.get('verdict')}", body_style))
                story.append(Spacer(1, 3))

            for adv in tradeoffs.get("advantages", []):
                story.append(Paragraph(f"• <b>Advantage vs. Runner-Up:</b> {adv}", bullet_style))
            for conc in tradeoffs.get("concessions", []):
                story.append(Paragraph(f"• <b>Concession / Trade-Off:</b> {conc}", bullet_style))
            story.append(Spacer(1, 8))

        # Technical Limitations & Missing Data
        story.append(Paragraph("6. Operational Limitations & Missing Literature Metrics", h1_style))

        top_cand = top_candidates[0]
        expl = top_cand.get("explanation", {})
        limits = expl.get("limitations", [])
        missing = expl.get("missing_data", [])

        if limits:
            for l in limits:
                story.append(Paragraph(f"• <b>Known Operational Limitation:</b> {l}", bullet_style))
        else:
            story.append(Paragraph("• Standard commercial handling constraints apply.", bullet_style))

        if missing:
            for m in missing:
                story.append(Paragraph(f"• <b>Missing Literature Metric:</b> {m} (not reported in published database entry)", bullet_style))
        story.append(Spacer(1, 8))

        # Scientific Standards & Engineering Disclaimers
        story.append(Paragraph("7. Scientific Basis, Standard Test Methods & Engineering Disclaimers", h1_style))
        disclaimer_text = (
            "<b>Standard Test Reference Methods:</b> Water Vapor Transmission Rates (WVTR) correspond to ASTM F1249 "
            "modulated infrared sensor testing conducted at standard 38.0°C and 90% RH. Oxygen Transmission Rates (OTR) "
            "correspond to ASTM D3985 / ASTM F1927 coulometric sensor testing conducted at 23.0°C and 0% RH.<br/><br/>"
            "<b>Accelerated Testing & Shelf-Life Verification:</b> Permeation rates cited represent pristine flat-sheet films "
            "under constant laboratory conditions. Commercial food stability depends on initial product moisture/water activity "
            "(a<sub>w</sub>), microbial inoculation, seal hermeticity, fold/flex crack fatigue, headspace gas, and distribution "
            "temperature abuse. This report provides scientific decision support; empirical Accelerated Shelf-Life Testing (ASLT) "
            "and formal migration testing under FDA 21 CFR §177 and EU Regulation 10/2011 are required prior to commercial release.<br/><br/>"
            "<b>Academic Honesty & Data Provenance:</b> Gaps in published literature entries are explicitly acknowledged "
            "and are never replaced with fabricated numbers. Literature citations: Robertson (2012) <i>Food Packaging</i>; "
            "Massey (2003) <i>Permeability Properties of Plastics and Elastomers</i>."
        )
        story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()


def generate_comparator_pdf(
    selected_materials: List[Dict[str, Any]],
    commodity_name: str = "Comparative Analysis",
    weighting_preset: str = "Balanced (Standard)"
) -> bytes:
    """
    Generates an executive PDF report comparing 2 to 4 packaging materials side-by-side,
    including technical specifications, trade-off synthesis, and ASTM standards.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=44,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CompTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f766e")
    )
    subtitle_style = ParagraphStyle(
        "CompSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b")
    )
    h1_style = ParagraphStyle(
        "CompH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "CompBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1e293b")
    )
    header_style = ParagraphStyle(
        "CompHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    )
    badge_style = ParagraphStyle(
        "CompBadge",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f766e")
    )

    story = []

    # Title & Metadata
    now_str = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    story.append(Paragraph("Smart FoodPack AI — Benchmark Comparison Report", title_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        f"Side-by-Side Material Permeation &amp; Sustainability Audit &bull; Generated: {now_str} &bull; Scope: {commodity_name} &bull; Preset: {weighting_preset}",
        subtitle_style
    ))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f766e"), spaceAfter=10))

    # Materials Overview
    story.append(Paragraph(f"1. Compared Benchmark Specimens ({len(selected_materials)} Materials)", h1_style))
    ov_rows = [[
        Paragraph("<b>Specimen</b>", header_style),
        Paragraph("<b>Polymer &amp; Structure</b>", header_style),
        Paragraph("<b>Thickness</b>", header_style),
        Paragraph("<b>WVTR (g/m²/d)</b>", header_style),
        Paragraph("<b>OTR (cc/m²/d)</b>", header_style),
        Paragraph("<b>Cost ($/kg)</b>", header_style),
        Paragraph("<b>Circularity</b>", header_style)
    ]]
    for m in selected_materials:
        cost_v = m.get("cost_index_usd_per_kg")
        cost_s = f"${cost_v:.2f}" if cost_v is not None else "Unreported"
        ov_rows.append([
            Paragraph(f"<b>{m['name'].split('(')[0].strip()}</b>", body_style),
            Paragraph(f"{m.get('polymer_code', '')} &bull; {m.get('structure_type', '')}", body_style),
            Paragraph(f"{m.get('nominal_thickness_um', '-')} µm", body_style),
            Paragraph(str(m.get("wvtr_g_m2_day", "-")), body_style),
            Paragraph(str(m.get("otr_cc_m2_day_atm", "-")), body_style),
            Paragraph(cost_s, body_style),
            Paragraph(f"<font size='6.5'>{m.get('recyclability_status', '-')}</font>", body_style)
        ])

    ov_table = Table(ov_rows, colWidths=[120, 110, 48, 56, 56, 48, 102])
    ov_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(ov_table)
    story.append(Spacer(1, 12))

    # Winner Highlights Summary
    story.append(Paragraph("2. Performance Category Leaders", h1_style))
    mats_wvtr = sorted(selected_materials, key=lambda x: float(x.get("wvtr_g_m2_day", 9999)))
    mats_cost = sorted(selected_materials, key=lambda x: float(x.get("cost_index_usd_per_kg", 9999) or 9999))
    mats_co2 = sorted(selected_materials, key=lambda x: float(x.get("carbon_footprint_kg_co2_per_kg", 999) or 999))

    best_b = mats_wvtr[0]
    cheapest = mats_cost[0]
    greenest = mats_co2[0]

    leader_data = [
        [
            Paragraph("<b>Category</b>", header_style),
            Paragraph("<b>Top Benchmark</b>", header_style),
            Paragraph("<b>Measured Metric</b>", header_style),
            Paragraph("<b>Engineering Significance</b>", header_style)
        ],
        [
            Paragraph("<b>🛡️ Best Moisture Barrier</b>", badge_style),
            Paragraph(best_b["name"].split("(")[0].strip(), body_style),
            Paragraph(f"WVTR: {best_b.get('wvtr_g_m2_day')} g/m²/d", body_style),
            Paragraph("Minimizes water vapor ingress; preserves crispness and prevents caking.", body_style)
        ],
        [
            Paragraph("<b>💰 Most Economical</b>", badge_style),
            Paragraph(cheapest["name"].split("(")[0].strip(), body_style),
            Paragraph(f"${cheapest.get('cost_index_usd_per_kg', 0):.2f} / kg", body_style),
            Paragraph("Minimizes unit packaging conversion costs for high-volume retail commodities.", body_style)
        ],
        [
            Paragraph("<b>🌱 Lowest Embodied Carbon</b>", badge_style),
            Paragraph(greenest["name"].split("(")[0].strip(), body_style),
            Paragraph(f"{greenest.get('carbon_footprint_kg_co2_per_kg', '-')} kg CO2e/kg", body_style),
            Paragraph("Reduces cradle-to-gate GHG emissions and aligns with corporate ESG targets.", body_style)
        ]
    ]
    leader_table = Table(leader_data, colWidths=[110, 110, 90, 230])
    leader_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(leader_table)
    story.append(Spacer(1, 12))

    # Detailed Specifications Matrix
    story.append(Paragraph("3. Full Specification Matrix & Regulatory Compliance", h1_style))
    matrix_props = [
        ("Structure Type", lambda m: m.get("structure_type", "—")),
        ("Nominal Thickness", lambda m: f"{m.get('nominal_thickness_um', '—')} µm"),
        ("WVTR (38°C, 90% RH)", lambda m: f"{m.get('wvtr_g_m2_day', '—')} g/m²/d"),
        ("OTR (23°C, 0% RH)", lambda m: f"{m.get('otr_cc_m2_day_atm', '—')} cc/m²/d"),
        ("Cost Index", lambda m: f"${m.get('cost_index_usd_per_kg'):.2f}/kg" if m.get("cost_index_usd_per_kg") is not None else "Unreported"),
        ("Carbon Footprint", lambda m: f"{m.get('carbon_footprint_kg_co2_per_kg'):.2f} kg CO2e" if m.get("carbon_footprint_kg_co2_per_kg") is not None else "Unreported"),
        ("Tensile Strength", lambda m: f"{m.get('tensile_strength_mpa'):.1f} MPa" if m.get("tensile_strength_mpa") is not None else "Unreported"),
        ("Food Contact Status", lambda m: m.get("food_contact_compliance", "Compliant")),
        ("Verification Tier", lambda m: "Lab Verified" if m.get("is_verified_experimental") == 1 else "Published Literature"),
        ("Literature Source", lambda m: m.get("data_source", "Published Packaging Literature"))
    ]

    mat_headers = [Paragraph("<b>Property / Standard</b>", header_style)]
    col_w = int(410 / len(selected_materials))
    col_widths = [130] + [col_w] * len(selected_materials)

    for m in selected_materials:
        mat_headers.append(Paragraph(f"<b>{m['name'].split('(')[0].strip()[:18]}</b>", header_style))
    matrix_rows = [mat_headers]

    for label, extractor in matrix_props:
        r = [Paragraph(f"<b>{label}</b>", body_style)]
        for m in selected_materials:
            r.append(Paragraph(f"<font size='7'>{extractor(m)}</font>", body_style))
        matrix_rows.append(r)

    matrix_table = Table(matrix_rows, colWidths=col_widths)
    matrix_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(matrix_table)
    story.append(Spacer(1, 10))

    # Regulatory & ASTM Disclaimers
    disclaimer_style = ParagraphStyle(
        "CompDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor("#64748b")
    )
    disclaimer_txt = (
        "<b>Scientific Disclaimer & ASTM Test Standard Conditions:</b> Water Vapor Transmission Rates (WVTR) correspond to ASTM F1249 "
        "modulated infrared sensor testing conducted at 38.0°C and 90% RH. Oxygen Transmission Rates (OTR) correspond to ASTM D3985 / ASTM F1927 "
        "coulometric sensor testing at 23.0°C and 0% RH. All citations represent published packaging literature (Robertson 2012, Massey 2003). "
        "Actual packaging shelf-life depends on seal integrity, environmental storage, and food water activity. Formal migration testing "
        "under FDA 21 CFR §177 and EU 10/2011 is advised prior to commercial manufacturing."
    )
    story.append(Paragraph(disclaimer_txt, disclaimer_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()

