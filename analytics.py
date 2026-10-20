"""
analytics.py - Interactive Visualization and Multi-Material Analytics
Generates Plotly figures for radar charts, barrier maps, sustainability trade-offs,
and sensitivity analyses, with honest missing-value disclosures and responsive dark styling.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from recommender import RECYCLABILITY_SCORE_MAP


# Modern FoodTech 3D Palette: Vibrant Orange, Amber Gold, Eco Green, Sky Blue, Deep Crimson
PALETTE = [
    "#FF8A00",  # Primary Vibrant Orange
    "#FFB300",  # Warm Amber Gold
    "#2E7D32",  # Eco Forest Green
    "#0288D1",  # Deep Sky Blue
    "#E65100",  # Deep Orange
    "#8D6E63",  # Warm Brown
    "#7B1FA2",  # Rich Purple
    "#455A64",  # Muted Slate
]


def create_radar_comparison_chart(selected_materials: List[Dict[str, Any]]) -> Tuple[go.Figure, List[str]]:
    """
    Builds a normalized 6-axis radar chart comparing materials.
    Returns (figure, list_of_missing_data_warnings).
    """
    categories = [
        "Moisture Barrier",
        "Oxygen Barrier",
        "Cost Economy",
        "Recyclability",
        "Thermal Tolerance",
        "Mechanical Strength"
    ]

    fig = go.Figure()
    warnings = []

    for idx, mat in enumerate(selected_materials):
        color = PALETTE[idx % len(PALETTE)]
        name = mat.get("name", f"Material {idx+1}")

        # 1. Moisture Barrier (Invert log scale: lower WVTR = higher barrier)
        wvtr = float(mat.get("wvtr_g_m2_day", 10.0))
        log_wvtr = np.log10(max(0.001, wvtr))
        moisture_barrier_score = float(np.clip(100.0 - ((log_wvtr - (-3.0)) / (2.3 - (-3.0))) * 90.0, 10.0, 100.0))

        # 2. Oxygen Barrier (Invert log scale: lower OTR = higher barrier)
        otr = float(mat.get("otr_cc_m2_day_atm", 100.0))
        log_otr = np.log10(max(0.001, otr))
        oxygen_barrier_score = float(np.clip(100.0 - ((log_otr - (-3.0)) / (4.0 - (-3.0))) * 90.0, 10.0, 100.0))

        # 3. Cost Economy (Lower cost = higher score)
        cost = float(mat.get("cost_index_usd_per_kg", 3.0))
        cost_score = float(np.clip(100.0 - ((cost - 1.0) / (6.0 - 1.0)) * 85.0, 15.0, 100.0))

        # 4. Recyclability
        recyclability_status = mat.get("recyclability_status", "")
        recyclability_score = RECYCLABILITY_SCORE_MAP.get(recyclability_status, 40.0)

        # 5. Thermal Window
        t_min = float(mat.get("temp_range_min_c", -20.0))
        t_max = float(mat.get("temp_range_max_c", 80.0))
        t_range = t_max - t_min
        thermal_score = float(np.clip((t_range / 200.0) * 100.0, 20.0, 100.0))

        # 6. Mechanical Strength
        tensile = mat.get("tensile_strength_mpa")
        if tensile is not None and not np.isnan(float(tensile)):
            mech_score = float(np.clip((float(tensile) / 200.0) * 100.0, 15.0, 100.0))
        else:
            mech_score = 50.0  # Plotted neutrally with explicit disclosure
            warnings.append(f"'{name}': Tensile strength data is unmeasured in literature; plotted at default neutral reference.")

        values = [
            round(moisture_barrier_score, 1),
            round(oxygen_barrier_score, 1),
            round(cost_score, 1),
            round(recyclability_score, 1),
            round(thermal_score, 1),
            round(mech_score, 1)
        ]
        values_closed = values + [values[0]]
        cat_closed = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=cat_closed,
            fill='toself',
            fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}",
            line=dict(color=color, width=2.5),
            name=name
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color="#8D6E63"),
                gridcolor="rgba(141, 110, 99, 0.2)"
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#3E2723", family="sans-serif"),
                rotation=90,
                direction="clockwise",
                gridcolor="rgba(141, 110, 99, 0.2)"
            ),
            bgcolor="rgba(255, 253, 245, 0.6)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(color="#3E2723", size=11)
        ),
        margin=dict(l=40, r=40, t=30, b=60),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig, warnings


def create_barrier_tradeoff_map(materials_df: pd.DataFrame, highlight_mats: List[str] = None) -> go.Figure:
    """
    Log-Log Scatter plot of WVTR vs OTR.
    Highlights specific materials and food packaging requirement zones.
    """
    df = materials_df.copy()
    df["log_wvtr"] = np.log10(np.clip(df["wvtr_g_m2_day"], 0.001, 1000.0))
    df["log_otr"] = np.log10(np.clip(df["otr_cc_m2_day_atm"], 0.001, 50000.0))

    if highlight_mats:
        df["is_highlighted"] = df["name"].apply(lambda x: "Recommended" if x in highlight_mats else "Other Materials")
    else:
        df["is_highlighted"] = "Standard Material"

    fig = px.scatter(
        df,
        x="wvtr_g_m2_day",
        y="otr_cc_m2_day_atm",
        log_x=True,
        log_y=True,
        color="structure_type",
        hover_name="name",
        hover_data={
            "nominal_thickness_um": True,
            "cost_index_usd_per_kg": ":$.2f",
            "recyclability_status": True,
            "wvtr_g_m2_day": ":.2f",
            "otr_cc_m2_day_atm": ":.1f"
        },
        size=[18 if x in (highlight_mats or []) else 10 for x in df["name"]],
        title="Physical Barrier Map: Water Vapor (WVTR) vs. Oxygen Transmission (OTR)",
        labels={
            "wvtr_g_m2_day": "Water Vapor Transmission Rate (g/m²/day @ 38°C, 90% RH) [Log Scale]",
            "otr_cc_m2_day_atm": "Oxygen Transmission Rate (cc/m²/day/atm @ 23°C, 0% RH) [Log Scale]",
            "structure_type": "Structure Type"
        }
    )

    # Add Food Packaging Application Quadrant Annotations with Warm FoodTech accents
    fig.add_annotation(
        x=np.log10(0.1), y=np.log10(1.0),
        text="<b>Ultra-Barrier Zone</b><br>(Milk Powder, Retort, Sensitive Snacks)",
        showarrow=False,
        xref="x", yref="y",
        font=dict(size=10, color="#E65100"),
        bgcolor="rgba(255, 243, 224, 0.85)",
        bordercolor="#FF8A00",
        borderwidth=1.5
    )

    fig.add_annotation(
        x=np.log10(100.0), y=np.log10(6000.0),
        text="<b>Breathable / EMAP Zone</b><br>(Fresh Cut Fruits & Vegetables)",
        showarrow=False,
        xref="x", yref="y",
        font=dict(size=10, color="#0288D1"),
        bgcolor="rgba(225, 245, 254, 0.85)",
        bordercolor="#0288D1",
        borderwidth=1.5
    )

    fig.update_layout(
        height=520,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3E2723"),
        xaxis=dict(gridcolor="rgba(141, 110, 99, 0.15)", linecolor="rgba(141, 110, 99, 0.3)"),
        yaxis=dict(gridcolor="rgba(141, 110, 99, 0.15)", linecolor="rgba(141, 110, 99, 0.3)"),
        legend=dict(font=dict(color="#6D4C41"))
    )
    return fig


def create_cost_sustainability_scatter(materials_df: pd.DataFrame) -> Tuple[go.Figure, List[str]]:
    """
    Scatter plot comparing Cost Index ($/kg) vs Carbon Footprint (kg CO2e/kg).
    Handles missing carbon footprint without silent fabrication.
    """
    df = materials_df.copy()
    missing_co2_count = df["carbon_footprint_kg_co2_per_kg"].isna().sum()
    warnings = []
    if missing_co2_count > 0:
        warnings.append(f"{missing_co2_count} material(s) have unmeasured cradle-to-gate carbon footprint data and are excluded from the carbon plot.")

    df_valid = df.dropna(subset=["carbon_footprint_kg_co2_per_kg"])

    fig = px.scatter(
        df_valid,
        x="cost_index_usd_per_kg",
        y="carbon_footprint_kg_co2_per_kg",
        color="recyclability_status",
        hover_name="name",
        hover_data={"nominal_thickness_um": True, "structure_type": True},
        size="nominal_thickness_um",
        title="Economic & Environmental Trade-offs: Cost Index vs. Cradle-to-Gate Carbon Footprint",
        labels={
            "cost_index_usd_per_kg": "Estimated Cost Index ($/kg)",
            "carbon_footprint_kg_co2_per_kg": "Carbon Footprint (kg CO₂-eq / kg material)",
            "recyclability_status": "Circularity & Recyclability"
        }
    )

    fig.update_layout(
        height=480,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3E2723"),
        xaxis=dict(gridcolor="rgba(141, 110, 99, 0.15)", linecolor="rgba(141, 110, 99, 0.3)"),
        yaxis=dict(gridcolor="rgba(141, 110, 99, 0.15)", linecolor="rgba(141, 110, 99, 0.3)"),
        legend=dict(font=dict(color="#6D4C41"))
    )
    return fig, warnings


def create_sensitivity_breakdown_chart(top_candidates: List[Dict[str, Any]]) -> go.Figure:
    """
    Grouped bar chart showing the breakdown of Barrier, Cost, Sustainability,
    and Mechanical score contributions for each top recommended candidate.
    """
    if not top_candidates:
        fig = go.Figure()
        fig.update_layout(title="No candidate materials to display")
        return fig

    names = [c["name"].split("(")[0].strip() for c in top_candidates]
    barrier_pts = [c.get("barrier_score", 0.0) for c in top_candidates]
    cost_pts = [c.get("cost_score", 0.0) or 0.0 for c in top_candidates]
    sust_pts = [c.get("sustainability_score", 0.0) for c in top_candidates]
    mech_pts = [c.get("mechanical_score", 0.0) for c in top_candidates]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Barrier Protection",
        x=names,
        y=barrier_pts,
        marker_color="#FF8A00"
    ))
    fig.add_trace(go.Bar(
        name="Cost Affordability",
        x=names,
        y=cost_pts,
        marker_color="#FFB300"
    ))
    fig.add_trace(go.Bar(
        name="Circularity & Sustainability",
        x=names,
        y=sust_pts,
        marker_color="#2E7D32"
    ))
    fig.add_trace(go.Bar(
        name="Mechanical Resilience",
        x=names,
        y=mech_pts,
        marker_color="#0288D1"
    ))

    fig.update_layout(
        barmode="group",
        title="Score Breakdown by Evaluation Pillar",
        yaxis=dict(title="Pillar Utility Score (0-100)", range=[0, 105], gridcolor="rgba(141, 110, 99, 0.15)", tickfont=dict(color="#6D4C41")),
        xaxis=dict(title="Candidate Material", tickfont=dict(color="#3E2723")),
        height=340,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3E2723"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5, font=dict(color="#6D4C41"))
    )
    return fig


def create_ranking_horizontal_bar_chart(candidates: List[Dict[str, Any]], top_n: int = 5) -> go.Figure:
    """
    Builds a clean, readable horizontal bar chart ranking candidate materials by overall score.
    Top candidate is prominently highlighted.
    """
    if not candidates:
        fig = go.Figure()
        fig.update_layout(title="No eligible materials to display")
        return fig

    # Sort ascending for horizontal bar (highest score appears at top)
    sorted_cands = sorted(candidates[:top_n], key=lambda x: x.get("total_mcda_score", 0.0), reverse=False)
    names = [c["name"].split("(")[0].strip() for c in sorted_cands]
    scores = [round(float(c.get("total_mcda_score", 0.0)), 1) for c in sorted_cands]

    # Highest score receives vibrant orange, runner-ups receive amber gold
    bar_colors = ["#FF8A00" if i == len(sorted_cands) - 1 else "#FFB300" for i in range(len(sorted_cands))]

    fig = go.Figure(go.Bar(
        x=scores,
        y=names,
        orientation="h",
        text=[f"{s}/100" for s in scores],
        textposition="outside",
        textfont=dict(color="#3E2723", size=11, family="Poppins, sans-serif"),
        marker=dict(color=bar_colors),
        hovertemplate="<b>%{y}</b><br>Overall Score: %{x} / 100<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="Candidate Ranking: Overall Multi-Criteria Match Score", font=dict(color="#3E2723", size=13)),
        xaxis=dict(title="Match Score (0-100)", range=[0, 115], gridcolor="rgba(141, 110, 99, 0.15)", tickfont=dict(color="#6D4C41")),
        yaxis=dict(autorange=True, tickfont=dict(color="#3E2723", size=11)),
        height=270,
        margin=dict(l=20, r=40, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3E2723")
    )
    return fig


def create_comparator_subplots(selected_materials: List[Dict[str, Any]], normalized: bool = False) -> go.Figure:
    """
    Builds 4 distinct comparative subplots across key engineering dimensions:
    1. Water Vapor Transmission (WVTR) - Lower is better
    2. Oxygen Transmission (OTR) - Lower is better
    3. Unit Cost ($/kg) - Lower is better
    4. Circularity / Recyclability Score (0-100) - Higher is better
    Best value per pillar highlighted with 🏆 Green, worst highlighted with subtle Red.
    """
    if not selected_materials:
        return go.Figure()

    short_names = [m["name"].split("(")[0].strip()[:18] for m in selected_materials]
    full_names = [m["name"] for m in selected_materials]

    # Metrics computation
    wvtrs = [float(m.get("wvtr_g_m2_day", 10.0)) for m in selected_materials]
    otrs = [float(m.get("otr_cc_m2_day_atm", 100.0)) for m in selected_materials]
    costs = [float(m.get("cost_index_usd_per_kg", 3.0) or 3.0) for m in selected_materials]
    circs = [float(RECYCLABILITY_SCORE_MAP.get(m.get("recyclability_status", ""), 40.0)) for m in selected_materials]

    # Subplot titles with unit and optimization direction
    titles = [
        "💧 Moisture (WVTR) ↓" if not normalized else "💧 Moisture Barrier (0-100) ↑",
        "💨 Oxygen (OTR) ↓" if not normalized else "💨 Oxygen Barrier (0-100) ↑",
        "💰 Cost ($/kg) ↓" if not normalized else "💰 Cost Economy (0-100) ↑",
        "🌱 Circularity Score ↑"
    ]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=titles,
        horizontal_spacing=0.08
    )

    def get_bar_colors(vals, lower_is_better=True):
        if len(vals) <= 1:
            return ["#2E7D32"] * len(vals)
        best_idx = int(np.argmin(vals)) if lower_is_better else int(np.argmax(vals))
        worst_idx = int(np.argmax(vals)) if lower_is_better else int(np.argmin(vals))
        colors = []
        for i in range(len(vals)):
            if i == best_idx:
                colors.append("#2E7D32")  # Best (Forest Green)
            elif i == worst_idx:
                colors.append("#E53935")  # Worst (Muted Crimson)
            else:
                colors.append("#FF8A00")  # Middle (Vibrant Orange)
        return colors

    # Normalized scores computation (0-100)
    if normalized:
        # Lower WVTR -> higher score
        min_w, max_w = min(wvtrs), max(wvtrs)
        w_scores = [round(100.0 if max_w == min_w else 100.0 - ((w - min_w) / (max_w - min_w)) * 80.0, 1) for w in wvtrs]
        # Lower OTR -> higher score
        min_o, max_o = min(otrs), max(otrs)
        o_scores = [round(100.0 if max_o == min_o else 100.0 - ((o - min_o) / (max_o - min_o)) * 80.0, 1) for o in otrs]
        # Lower Cost -> higher score
        min_c, max_c = min(costs), max(costs)
        c_scores = [round(100.0 if max_c == min_c else 100.0 - ((c - min_c) / (max_c - min_c)) * 80.0, 1) for c in costs]
        
        y_w, y_o, y_c, y_s = w_scores, o_scores, c_scores, circs
        c_w = get_bar_colors(w_scores, lower_is_better=False)
        c_o = get_bar_colors(o_scores, lower_is_better=False)
        c_c = get_bar_colors(c_scores, lower_is_better=False)
        c_s = get_bar_colors(circs, lower_is_better=False)
        
        txt_w = [f"🏆 {v:.0f}" if c == "#2E7D32" else f"{v:.0f}" for v, c in zip(y_w, c_w)]
        txt_o = [f"🏆 {v:.0f}" if c == "#2E7D32" else f"{v:.0f}" for v, c in zip(y_o, c_o)]
        txt_c = [f"🏆 {v:.0f}" if c == "#2E7D32" else f"{v:.0f}" for v, c in zip(y_c, c_c)]
        txt_s = [f"🏆 {v:.0f}" if c == "#2E7D32" else f"{v:.0f}" for v, c in zip(y_s, c_s)]
    else:
        y_w, y_o, y_c, y_s = wvtrs, otrs, costs, circs
        c_w = get_bar_colors(wvtrs, lower_is_better=True)
        c_o = get_bar_colors(otrs, lower_is_better=True)
        c_c = get_bar_colors(costs, lower_is_better=True)
        c_s = get_bar_colors(circs, lower_is_better=False)
        
        txt_w = [f"🏆 {v:.2f}" if c == "#2E7D32" else f"{v:.2f}" for v, c in zip(y_w, c_w)]
        txt_o = [f"🏆 {v:.1f}" if c == "#2E7D32" else f"{v:.1f}" for v, c in zip(y_o, c_o)]
        txt_c = [f"🏆 ${v:.2f}" if c == "#2E7D32" else f"${v:.2f}" for v, c in zip(y_c, c_c)]
        txt_s = [f"🏆 {v:.0f}" if c == "#2E7D32" else f"{v:.0f}" for v, c in zip(y_s, c_s)]

    # 1. WVTR Bar
    fig.add_trace(go.Bar(
        x=short_names, y=y_w, text=txt_w, textposition="outside",
        marker=dict(color=c_w), name="WVTR", hovertext=full_names,
        showlegend=False
    ), row=1, col=1)

    # 2. OTR Bar
    fig.add_trace(go.Bar(
        x=short_names, y=y_o, text=txt_o, textposition="outside",
        marker=dict(color=c_o), name="OTR", hovertext=full_names,
        showlegend=False
    ), row=1, col=2)

    # 3. Cost Bar
    fig.add_trace(go.Bar(
        x=short_names, y=y_c, text=txt_c, textposition="outside",
        marker=dict(color=c_c), name="Cost", hovertext=full_names,
        showlegend=False
    ), row=1, col=3)

    # 4. Circularity Bar
    fig.add_trace(go.Bar(
        x=short_names, y=y_s, text=txt_s, textposition="outside",
        marker=dict(color=c_s), name="Circularity", hovertext=full_names,
        showlegend=False
    ), row=1, col=4)

    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3E2723", family="Inter, sans-serif")
    )
    for col_idx in range(1, 5):
        fig.update_xaxes(showgrid=False, linecolor="rgba(62, 39, 35, 0.2)", tickfont=dict(size=10, color="#3E2723"), row=1, col=col_idx)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(62, 39, 35, 0.08)", linecolor="rgba(62, 39, 35, 0.2)", tickfont=dict(size=9, color="#6D4C41"), row=1, col=col_idx)

    # If raw values and OTR range is large, apply log scale to col 2
    if not normalized and max(otrs) / max(0.01, min(otrs)) > 40:
        fig.update_yaxes(type="log", row=1, col=2)
    if not normalized and max(wvtrs) / max(0.01, min(wvtrs)) > 40:
        fig.update_yaxes(type="log", row=1, col=1)

    return fig


def create_comparator_heatmap(selected_materials: List[Dict[str, Any]]) -> go.Figure:
    """
    Builds a normalized 0-100 property comparison Heatmap across all selected materials.
    """
    if not selected_materials:
        return go.Figure()

    short_names = [m["name"].split("(")[0].strip() for m in selected_materials]
    metrics = [
        "Moisture Barrier",
        "Oxygen Barrier",
        "Cost Economy",
        "Circularity & Eco",
        "Mechanical Strength",
        "Thermal Tolerance"
    ]

    z_matrix = []
    for m in selected_materials:
        # 1. Moisture Barrier (Invert log scale: lower WVTR = higher barrier)
        wvtr = float(m.get("wvtr_g_m2_day", 10.0))
        log_wvtr = np.log10(max(0.001, wvtr))
        mb_score = float(np.clip(100.0 - ((log_wvtr - (-3.0)) / (2.3 - (-3.0))) * 90.0, 10.0, 100.0))

        # 2. Oxygen Barrier (Invert log scale: lower OTR = higher barrier)
        otr = float(m.get("otr_cc_m2_day_atm", 100.0))
        log_otr = np.log10(max(0.001, otr))
        ob_score = float(np.clip(100.0 - ((log_otr - (-3.0)) / (4.0 - (-3.0))) * 90.0, 10.0, 100.0))

        # 3. Cost Economy (Lower cost = higher score)
        cost = float(m.get("cost_index_usd_per_kg", 3.0) or 3.0)
        c_score = float(np.clip(100.0 - ((cost - 1.0) / (6.0 - 1.0)) * 85.0, 15.0, 100.0))

        # 4. Circularity
        circ_score = float(RECYCLABILITY_SCORE_MAP.get(m.get("recyclability_status", ""), 40.0))

        # 5. Mechanical
        tensile = m.get("tensile_strength_mpa")
        mech_score = float(np.clip((float(tensile) / 200.0) * 100.0, 15.0, 100.0)) if tensile is not None else 50.0

        # 6. Thermal
        t_min = float(m.get("temp_range_min_c", -20.0))
        t_max = float(m.get("temp_range_max_c", 80.0))
        therm_score = float(np.clip(((t_max - t_min) / 200.0) * 100.0, 20.0, 100.0))

        z_matrix.append([
            round(mb_score, 0),
            round(ob_score, 0),
            round(c_score, 0),
            round(circ_score, 0),
            round(mech_score, 0),
            round(therm_score, 0)
        ])

    z_matrix = np.array(z_matrix).T  # Shape: (len(metrics), len(materials))

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=short_names,
        y=metrics,
        colorscale=[
            [0.0, "#FFCDD2"],   # Low score (Soft Red)
            [0.4, "#FFE082"],   # Moderate score (Warm Amber)
            [0.7, "#A5D6A7"],   # Good score (Soft Green)
            [1.0, "#2E7D32"]    # Top score (Deep Forest Green)
        ],
        text=z_matrix,
        texttemplate="%{text:.0f}",
        textfont={"size": 13, "color": "#1E293B", "family": "Poppins, sans-serif"},
        colorbar=dict(title="Index (0-100)", tickfont=dict(color="#3E2723"))
    ))

    fig.update_layout(
        title=dict(text="Benchmark Property Matrix Heatmap (Normalized Utility 0-100)", font=dict(size=13, color="#3E2723")),
        height=320,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3E2723")
    )
    return fig


def create_pareto_barrier_cost(all_materials: List[Dict[str, Any]], selected_materials: List[Dict[str, Any]]) -> go.Figure:
    """
    Scatter plot comparing Water Vapor Transmission (WVTR, log scale) vs. Unit Cost ($/kg).
    Identifies and highlights the Pareto Frontier (materials where no alternative has both lower WVTR and lower cost).
    """
    df = pd.DataFrame(all_materials)
    selected_names = [m["name"] for m in selected_materials]

    # Clean numeric columns
    df["wvtr"] = pd.to_numeric(df["wvtr_g_m2_day"], errors="coerce").clip(lower=0.001)
    df["cost"] = pd.to_numeric(df["cost_index_usd_per_kg"], errors="coerce").fillna(3.5)
    df["carbon"] = pd.to_numeric(df["carbon_footprint_kg_co2_per_kg"], errors="coerce").fillna(2.0)

    # Compute 2D Pareto frontier: lower WVTR is better, lower Cost is better
    sorted_df = df.sort_values(by=["wvtr", "cost"], ascending=[True, True])
    pareto_mask = []
    min_cost_so_far = float('inf')
    for _, row in sorted_df.iterrows():
        if row["cost"] < min_cost_so_far:
            pareto_mask.append(True)
            min_cost_so_far = row["cost"]
        else:
            pareto_mask.append(False)
    sorted_df["is_pareto"] = pareto_mask
    pareto_pts = sorted_df[sorted_df["is_pareto"]].sort_values("wvtr")

    fig = go.Figure()

    # 1. Non-Pareto Benchmark points
    non_pareto = sorted_df[~sorted_df["is_pareto"]]
    fig.add_trace(go.Scatter(
        x=non_pareto["wvtr"],
        y=non_pareto["cost"],
        mode="markers",
        marker=dict(size=9, color="#B0BEC5", opacity=0.7),
        name="Other Materials",
        text=non_pareto["name"],
        hovertemplate="<b>%{text}</b><br>WVTR: %{x:.2f} g/m²/d<br>Cost: $%{y:.2f}/kg<extra></extra>"
    ))

    # 2. Pareto Optimal Line
    fig.add_trace(go.Scatter(
        x=pareto_pts["wvtr"],
        y=pareto_pts["cost"],
        mode="lines+markers",
        line=dict(color="#2E7D32", width=2.5, dash="dash"),
        marker=dict(size=12, color="#2E7D32", symbol="star"),
        name="Pareto Frontier (Optimal Trade-off)",
        text=pareto_pts["name"],
        hovertemplate="<b>⭐ Pareto Benchmark: %{text}</b><br>WVTR: %{x:.2f} g/m²/d<br>Cost: $%{y:.2f}/kg<extra></extra>"
    ))

    # 3. Selected Materials Highlighted with Vibrant Orange Rings
    sel_df = df[df["name"].isin(selected_names)]
    if not sel_df.empty:
        fig.add_trace(go.Scatter(
            x=sel_df["wvtr"],
            y=sel_df["cost"],
            mode="markers+text",
            marker=dict(size=18, color="#FF8A00", line=dict(color="#D84315", width=2.5)),
            text=[name.split("(")[0].strip()[:14] for name in sel_df["name"]],
            textposition="top center",
            textfont=dict(color="#3E2723", size=10, family="Poppins, sans-serif"),
            name="Selected in Comparator",
            hovertemplate="<b>🔍 Selected: %{text}</b><br>WVTR: %{x:.2f} g/m²/d<br>Cost: $%{y:.2f}/kg<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text="Cost vs. Moisture Barrier Trade-off (Pareto Frontier)", font=dict(size=13, color="#3E2723")),
        xaxis=dict(title="Water Vapor Transmission (WVTR, g/m²/day) [Log Scale - Lower is Better]", type="log", gridcolor="rgba(141, 110, 99, 0.15)", tickfont=dict(color="#6D4C41")),
        yaxis=dict(title="Unit Cost ($/kg) [Lower is Better]", gridcolor="rgba(141, 110, 99, 0.15)", tickfont=dict(color="#6D4C41")),
        height=360,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3E2723"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(color="#3E2723", size=10))
    )
    return fig


def create_shelf_life_comparison_bar(
    selected_materials: List[Dict[str, Any]],
    shelf_life_days_map: Dict[str, float],
    target_shelf_life_days: int = 180,
    food_name: str = "Food Product"
) -> go.Figure:
    """
    Builds a comparison bar chart of estimated shelf life across selected materials for a food product.
    """
    if not selected_materials:
        return go.Figure()

    names = [m["name"].split("(")[0].strip() for m in selected_materials]
    days = [round(float(shelf_life_days_map.get(m["name"], 0.0)), 1) for m in selected_materials]

    colors = []
    for d in days:
        if d >= target_shelf_life_days:
            colors.append("#2E7D32")  # Meets or exceeds target (Green)
        elif d >= target_shelf_life_days * 0.75:
            colors.append("#FF8A00")  # Moderate shelf-life (Orange)
        else:
            colors.append("#E53935")  # Deficient shelf-life (Red)

    fig = go.Figure(go.Bar(
        x=days,
        y=names,
        orientation="h",
        text=[f"{d:.0f} Days" for d in days],
        textposition="outside",
        textfont=dict(color="#3E2723", size=11, family="Poppins, sans-serif"),
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b><br>Predicted Shelf-Life: %{x:.0f} Days<extra></extra>"
    ))

    # Add Target Shelf-Life Reference Line
    fig.add_vline(
        x=target_shelf_life_days,
        line_dash="dash",
        line_color="#D84315",
        line_width=2,
        annotation_text=f"Target: {target_shelf_life_days} Days",
        annotation_position="top right",
        annotation_font=dict(color="#D84315", size=10)
    )

    max_x = max(max(days, default=100), target_shelf_life_days) * 1.25

    fig.update_layout(
        title=dict(text=f"Estimated Shelf-Life Comparison for {food_name} (Fickian Kinetics)", font=dict(color="#3E2723", size=13)),
        xaxis=dict(title="Predicted Shelf Life (Days)", range=[0, max_x], gridcolor="rgba(141, 110, 99, 0.15)", tickfont=dict(color="#6D4C41")),
        yaxis=dict(autorange=True, tickfont=dict(color="#3E2723", size=11)),
        height=260,
        margin=dict(l=20, r=40, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3E2723")
    )
    return fig

