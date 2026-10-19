"""
analytics.py - Interactive Visualization and Multi-Material Analytics
Generates Plotly figures for radar charts, barrier maps, sustainability trade-offs,
and sensitivity analyses, with honest missing-value disclosures.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from recommender import RECYCLABILITY_SCORE_MAP


# Brand Color Palette: Professional Teal, Navy, Emerald, Slate
PALETTE = [
    "#0d9488",  # Teal 600
    "#0284c7",  # Sky 600
    "#10b981",  # Emerald 500
    "#f59e0b",  # Amber 500
    "#6366f1",  # Indigo 500
    "#ec4899",  # Pink 500
    "#8b5cf6",  # Violet 500
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
        # Range: 0.001 to 200. Log10 range: -3 to 2.3
        # Score = 100 at 0.01, 10 at 200
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
        # Close radar loop
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
                tickfont=dict(size=10, color="#64748b"),
                gridcolor="#e2e8f0"
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#1e293b", family="sans-serif"),
                rotation=90,
                direction="clockwise"
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
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

    # Add Food Packaging Application Quadrant Annotations
    fig.add_annotation(
        x=np.log10(0.1), y=np.log10(1.0),
        text="<b>Ultra-Barrier Zone</b><br>(Milk Powder, Retort, Sensitive Snacks)",
        showarrow=False,
        xref="x", yref="y",
        font=dict(size=10, color="#0f766e"),
        bgcolor="rgba(204, 251, 241, 0.7)",
        bordercolor="#0d9488",
        borderwidth=1
    )

    fig.add_annotation(
        x=np.log10(100.0), y=np.log10(6000.0),
        text="<b>Breathable / EMAP Zone</b><br>(Fresh Cut Fruits & Vegetables)",
        showarrow=False,
        xref="x", yref="y",
        font=dict(size=10, color="#b45309"),
        bgcolor="rgba(254, 243, 199, 0.7)",
        bordercolor="#f59e0b",
        borderwidth=1
    )

    fig.update_layout(
        height=520,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc"
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
        plot_bgcolor="#f8fafc"
    )
    return fig, warnings


def create_sensitivity_breakdown_chart(top_candidates: List[Dict[str, Any]]) -> go.Figure:
    """
    Stacked bar chart showing the breakdown of Barrier, Cost, and Sustainability
    score contributions for each top recommended candidate.
    """
    names = [c["name"] for c in top_candidates]
    barrier_pts = [c["barrier_score"] for c in top_candidates]
    cost_pts = [c["cost_score"] for c in top_candidates]
    sust_pts = [c["sustainability_score"] for c in top_candidates]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Barrier Protection Score",
        x=names,
        y=barrier_pts,
        marker_color="#0d9488"
    ))
    fig.add_trace(go.Bar(
        name="Cost Economy Score",
        x=names,
        y=cost_pts,
        marker_color="#0284c7"
    ))
    fig.add_trace(go.Bar(
        name="Sustainability / Circularity Score",
        x=names,
        y=sust_pts,
        marker_color="#10b981"
    ))

    fig.update_layout(
        barmode="group",
        title="Score Breakdown by Evaluation Pillar",
        yaxis=dict(title="Pillar Utility Score (0-100)", range=[0, 105]),
        xaxis=dict(title="Candidate Material"),
        height=380,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    return fig
