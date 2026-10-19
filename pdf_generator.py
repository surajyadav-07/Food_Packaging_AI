"""
pdf_generator.py - Professional PDF Report Export for Food Packaging Recommendations
Generates downloadable, publication-quality technical packaging decision reports
using ReportLab with dynamic user inputs, transparent scoring, and scientific disclaimers.
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
    KeepTogether,
    HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute total page count and render
    running headers and footers on every page.
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
            self.drawString(36, 762, "FoodPackAI — Technical Recommendation & Material Decision Report")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 42, 576, 42)

        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 30, footer_text)
        disclaimer_short = "Physics-Grounded Multi-Criteria Analysis • Standard Reference Conditions (ASTM F1249 / D3985)"
        self.drawString(36, 30, disclaimer_short)

        self.restoreState()


def generate_recommendation_pdf(
    commodity_profile: Dict[str, Any],
    recommendation_result: Dict[str, Any],
    user_weights: Dict[str, float]
) -> bytes:
    """
    Constructs a complete technical recommendation PDF document from current user analysis.
    Returns:
        bytes: in-memory PDF binary data ready for download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a")
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0f766e")
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=10,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=6,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1e293b")
    )

    body_bold = ParagraphStyle(
        "Body_Bold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    body_muted = ParagraphStyle(
        "Body_Muted",
        parent=body_style,
        fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#64748b")
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-6,
        spaceBefore=1,
        spaceAfter=1
    )

    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=body_style,
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#334155")
    )

    story = []

    # 1. Header Banner
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph("AI-Based Intelligent Food Packaging Recommendation System", title_style))
    story.append(Paragraph(f"Material Suitability & Technical Decision Report • Generated: {now_str}", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0d9488"), spaceAfter=12))

    # 2. Section: Food Commodity Specification & User Constraints
    story.append(Paragraph("1. Food Commodity Profile & Spoilage Sensitivities", h1_style))

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
            Paragraph("<b>Commodity Name:</b>", body_style), Paragraph(str(comm_name), body_style),
            Paragraph("<b>Category:</b>", body_style), Paragraph(str(comm_cat), body_style)
        ],
        [
            Paragraph("<b>Moisture Sensitivity:</b>", body_style), Paragraph(str(m_sens), body_style),
            Paragraph("<b>Oxygen Sensitivity:</b>", body_style), Paragraph(str(o_sens), body_style)
        ],
        [
            Paragraph("<b>Lipid / Fat Level:</b>", body_style), Paragraph(str(fat_lvl), body_style),
            Paragraph("<b>Light Sensitivity:</b>", body_style), Paragraph(str(light_sens), body_style)
        ],
        [
            Paragraph("<b>Storage Temperature:</b>", body_style), Paragraph(f"{t_store} °C", body_style),
            Paragraph("<b>Ambient Humidity:</b>", body_style), Paragraph(f"{rh_store} % RH", body_style)
        ],
        [
            Paragraph("<b>Target Shelf Life:</b>", body_style), Paragraph(f"{shelf_life_target} Days", body_style),
            Paragraph("<b>Puncture Hazard:</b>", body_style), Paragraph(str(punc_risk), body_style)
        ],
        [
            Paragraph("<b>Food Contact Rule:</b>", body_style), Paragraph("Mandatory Compliance (FDA 21 CFR / EU 10/2011)", body_style),
            Paragraph("<b>Evaluation Scope:</b>", body_style), Paragraph("Stage-1 Knockouts + MCDA Scoring", body_style)
        ]
    ]

    comm_table = Table(comm_table_data, colWidths=[110, 150, 110, 150])
    comm_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(comm_table)
    story.append(Spacer(1, 8))

    # 3. Section: Decision Priority Weights & Filtering Audit
    story.append(Paragraph("2. Decision Preferences & Knockout Filter Audit", h1_style))

    w_b = user_weights.get("barrier", 0.45)
    w_c = user_weights.get("cost", 0.25)
    w_s = user_weights.get("sustainability", 0.30)
    w_sum = w_b + w_c + w_s or 1.0

    tot_eval = recommendation_result.get("total_evaluated", 0)
    tot_qual = recommendation_result.get("total_qualified", 0)
    tot_knock = recommendation_result.get("total_knocked_out", 0)

    weights_data = [
        [
            Paragraph(f"<b>Barrier Protection Weight:</b> {w_b/w_sum*100:.1f}%", body_style),
            Paragraph(f"<b>Cost Economy Weight:</b> {w_c/w_sum*100:.1f}%", body_style),
            Paragraph(f"<b>Circularity Weight:</b> {w_s/w_sum*100:.1f}%", body_style)
        ],
        [
            Paragraph(f"<b>Total Evaluated in DB:</b> {tot_eval}", body_style),
            Paragraph(f"<b>Stage-1 Feasible:</b> {tot_qual}", body_style),
            Paragraph(f"<b>Disqualified by Physics/Safety:</b> {tot_knock}", body_style)
        ]
    ]
    weights_table = Table(weights_data, colWidths=[173, 173, 174])
    weights_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdfa")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#99f6e4")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccfbf1")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(weights_table)
    story.append(Spacer(1, 10))

    # 4. Section: Top Ranked Candidates & Explanations
    story.append(Paragraph("3. Top Recommended Packaging Candidates", h1_style))

    top_candidates = recommendation_result.get("top_candidates", [])
    if not top_candidates:
        story.append(Paragraph("No materials met all physical and food-safety constraints.", body_style))
    else:
        rank_titles = [
            "PRIMARY RECOMMENDED SOLUTION",
            "HIGH-PERFORMANCE VIABLE ALTERNATIVE",
            "BUDGET / CIRCULARITY ALTERNATIVE"
        ]

        for idx, cand in enumerate(top_candidates):
            rank_label = rank_titles[idx] if idx < len(rank_titles) else f"CANDIDATE #{cand.get('rank', idx+1)}"
            cand_name = cand.get("name", "Unknown Material")
            mcda_score = cand.get("total_mcda_score", 0.0)
            b_score = cand.get("barrier_score", 0.0)
            c_score = cand.get("cost_score", 0.0)
            s_score = cand.get("sustainability_score", 0.0)
            m_score = cand.get("mechanical_score", 0.0)
            cost_val = cand.get("cost_index_usd_per_kg", 0.0)
            recyclability = cand.get("recyclability_status", "Unspecified")
            compliance = cand.get("food_contact_compliance", "Compliant")
            data_source = cand.get("data_source", "Literature Citation")
            test_conditions = cand.get("test_conditions", "ASTM F1249 / D3985")
            co2_footprint = cand.get("carbon_footprint_kg_co2_per_kg")
            co2_str = f"{co2_footprint:.2f} kg CO2e/kg" if co2_footprint is not None else "Unreported in literature"

            expl = cand.get("explanation", {})
            rationale = expl.get("rationale", "")
            reqs = expl.get("requirements_met", [])
            limits = expl.get("limitations", [])
            missing = expl.get("missing_data", [])

            candidate_elements = []

            # Candidate Header
            candidate_elements.append(Paragraph(
                f"<b>RANK #{cand.get('rank', idx+1)} — {cand_name}</b> "
                f"<font color='#0d9488'>[MCDA Score: {mcda_score}/100 | {rank_label}]</font>",
                h2_style
            ))
            candidate_elements.append(Paragraph(f"<i>{rationale}</i>", body_style))
            candidate_elements.append(Spacer(1, 3))

            # Pillar Breakdown & Properties Grid
            spec_rows = [
                [
                    Paragraph("<b>Barrier:</b>", body_style), Paragraph(f"{b_score}/100", body_style),
                    Paragraph("<b>Cost:</b>", body_style), Paragraph(f"{c_score}/100", body_style),
                    Paragraph("<b>Circularity:</b>", body_style), Paragraph(f"{s_score}/100", body_style),
                    Paragraph("<b>Mechanical:</b>", body_style), Paragraph(f"{m_score}/100", body_style)
                ],
                [
                    Paragraph("<b>Structure:</b>", body_style), Paragraph(str(cand.get("structure_type", "Flexible")), body_style),
                    Paragraph("<b>Thickness:</b>", body_style), Paragraph(f"{cand.get('nominal_thickness_um', 0)} µm", body_style),
                    Paragraph("<b>WVTR:</b>", body_style), Paragraph(f"{cand.get('wvtr_g_m2_day')} g/m²/d", body_style),
                    Paragraph("<b>OTR:</b>", body_style), Paragraph(f"{cand.get('otr_cc_m2_day_atm')} cc/m²/d", body_style)
                ],
                [
                    Paragraph("<b>Cost Index:</b>", body_style), Paragraph(f"${cost_val:.2f}/kg", body_style),
                    Paragraph("<b>CO2e:</b>", body_style), Paragraph(co2_str, body_style),
                    Paragraph("<b>Stream:</b>", body_style), Paragraph(f"<font size='7'>{recyclability}</font>", body_style),
                    Paragraph("<b>Food Contact:</b>", body_style), Paragraph(f"<font size='7'>{compliance}</font>", body_style)
                ],
                [
                    Paragraph("<b>Data Source:</b>", body_style),
                    Paragraph(f"<font size='6.5'>{data_source}</font>", body_style),
                    Paragraph("<b>Test Standards:</b>", body_style),
                    Paragraph(f"<font size='6.5'>{test_conditions}</font>", body_style),
                    Paragraph("<b>Polymer Code:</b>", body_style),
                    Paragraph(str(cand.get("polymer_code", "N/A")), body_style),
                    Paragraph("<b>Verification:</b>", body_style),
                    Paragraph("Literature" if cand.get("is_verified_experimental") == 0 else "Lab Verified", body_style)
                ]
            ]
            spec_table = Table(spec_rows, colWidths=[65, 65, 65, 65, 65, 65, 65, 65])
            spec_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            candidate_elements.append(spec_table)
            candidate_elements.append(Spacer(1, 3))

            # Explainability Lists
            candidate_elements.append(Paragraph("<b>Requirements Satisfied:</b>", body_bold))
            for r in reqs:
                candidate_elements.append(Paragraph(f"• {r}", bullet_style))

            candidate_elements.append(Paragraph("<b>Technical Limitations & Operational Trade-offs:</b>", body_bold))
            for l in limits:
                candidate_elements.append(Paragraph(f"• {l}", bullet_style))

            if missing:
                candidate_elements.append(Paragraph("<b>Missing / Unverified Literature Metrics:</b>", body_bold))
                for m in missing:
                    candidate_elements.append(Paragraph(f"• <font color='#64748b'>{m}</font>", bullet_style))

            candidate_elements.append(Spacer(1, 6))
            candidate_elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=6))

            story.append(KeepTogether(candidate_elements))

    # 5. Section: Side-by-Side Comparison Table
    story.append(Paragraph("4. Side-by-Side Material Properties Comparison", h1_style))

    comp_headers = [
        Paragraph("<b>Rank</b>", body_style),
        Paragraph("<b>Candidate Material</b>", body_style),
        Paragraph("<b>Thickness</b>", body_style),
        Paragraph("<b>WVTR</b><br/><font size='6.5'>g/m²/d</font>", body_style),
        Paragraph("<b>OTR</b><br/><font size='6.5'>cc/m²/d</font>", body_style),
        Paragraph("<b>Cost</b><br/><font size='6.5'>$/kg</font>", body_style),
        Paragraph("<b>Circularity</b>", body_style),
        Paragraph("<b>Compliance</b>", body_style)
    ]
    comp_table_rows = [comp_headers]

    for cand in top_candidates:
        comp_table_rows.append([
            Paragraph(f"#{cand.get('rank')}", body_bold),
            Paragraph(cand.get("name", ""), body_style),
            Paragraph(f"{cand.get('nominal_thickness_um')} µm", body_style),
            Paragraph(str(cand.get("wvtr_g_m2_day")), body_style),
            Paragraph(str(cand.get("otr_cc_m2_day_atm")), body_style),
            Paragraph(f"${cand.get('cost_index_usd_per_kg'):.2f}", body_style),
            Paragraph(f"<font size='7'>{cand.get('recyclability_status')}</font>", body_style),
            Paragraph(f"<font size='6.5'>{cand.get('food_contact_compliance')}</font>", body_style)
        ])

    comp_summary_table = Table(comp_table_rows, colWidths=[30, 120, 45, 45, 45, 45, 95, 95])
    comp_summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(comp_summary_table)
    story.append(Spacer(1, 10))

    # 6. Section: Scientific Disclaimers, ASTM Standards & References
    story.append(Paragraph("5. Scientific Basis, Standard Test Conditions & Disclaimers", h1_style))
    disclaimer_text = (
        "<b>Standard Reference Test Conditions:</b> Water Vapor Transmission Rates (WVTR) cited herein correspond to "
        "ASTM F1249 standard modulated infrared sensor testing conducted at 38.0°C and 90% Relative Humidity. "
        "Oxygen Transmission Rates (OTR) correspond to ASTM D3985 / ASTM F1927 coulometric sensor testing conducted at "
        "23.0°C and 0% Relative Humidity.<br/><br/>"
        "<b>Shelf-Life & Degradation Determinants:</b> Packaging material barrier properties represent only one factor "
        "determining food stability. Actual commercial shelf life depends on initial microbial contamination, product water activity "
        "(a<sub>w</sub>), active respiration kinetics (for fresh produce), package seal integrity, pinhole flex cracks, "
        "headspace volume, and supply chain temperature abuse. This decision report provides first-principles physical "
        "permeation and multi-attribute decision support; it does not constitute a certified empirical shelf-life guarantee.<br/><br/>"
        "<b>Missing Data & Academic Honesty:</b> Unreported values (such as carbon footprint or CO2TR) reflect gaps in standard "
        "published literature entries and have NOT been replaced with fabricated numbers.<br/><br/>"
        "<b>Primary Reference Authorities:</b> Robertson, G. L. (2012) <i>Food Packaging: Principles and Practice</i>, 3rd ed., CRC Press; "
        "Massey, L. K. (2003) <i>Permeability Properties of Plastics and Elastomers</i>, William Andrew; US FDA 21 CFR §177; EU Regulation (EU) No 10/2011."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()
