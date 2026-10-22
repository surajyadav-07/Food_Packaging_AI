"""
app.py - Smart FoodPack AI (Food Packaging Intelligence & Decision Support Platform)
A clean, elegant, professional decision-support system for food packaging material recommendation,
barrier performance evaluation, shelf-life simulation, and sustainability assessment.

Designed with a modern scientific FoodTech aesthetic, stable selection controls, and physics-grounded logic.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
import json
import base64
import textwrap
from typing import Any, Dict, List, Optional

# Local system modules
import database
import recommender
import shelf_life
import data_loader
import analytics
import pdf_generator

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart FoodPack AI — Packaging Intelligence Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database schema if not present
database.init_database()


# -----------------------------------------------------------------------------
# Image Asset Helpers (Robust local path resolution with graceful fallback)
# -----------------------------------------------------------------------------
FOOD_PRODUCT_IMAGE_MAP = {
    "Fried Potato Chips & Crisps": "assets/foods/potato_chips.jpg",
    "Roasted Cashews & Almonds": "assets/foods/roasted_cashews.jpg",
    "Strawberries & Soft Berries": "assets/foods/strawberries.jpg",
    "Cheddar Cheese Block": "assets/foods/cheddar_cheese.jpg",
    "White Sandwich Bread": "assets/foods/sandwich_bread.jpg",
    "Dry Digestive Biscuits / Cookies": "assets/foods/digestive_biscuits.jpg",
    "Polished Basmati Rice": "assets/foods/basmati_rice.svg",
    "Whole Wheat Flour (Atta)": "assets/foods/wheat_flour.svg",
    "Salted Butter": "assets/foods/salted_butter.svg",
    "Whole Milk Powder (Full Cream)": "assets/foods/milk_powder.svg",
    "Fresh Cut Apple Slices": "assets/foods/apple_slices.svg",
    "Green Lentils & Chickpeas": "assets/foods/green_lentils.svg",
}

FOOD_CATEGORY_IMAGE_MAP = {
    "Snacks": "assets/foods/snacks.svg",
    "Bakery products": "assets/foods/bakery.svg",
    "Dairy products": "assets/foods/dairy.svg",
    "Fruits and vegetables": "assets/foods/fruits_vegetables.svg",
    "Cereals and grains": "assets/foods/cereals_grains.svg",
    "Pulses": "assets/foods/pulses.svg",
}

FOOD_EMOJI_MAP = {
    # Bakery products
    "White Sandwich Bread": "🍞",
    "Dry Digestive Biscuits / Cookies": "🍪",
    "Butter Croissants & Pastries": "🥐",
    "Sliced Plain Bagels": "🥯",
    "Chocolate Chip Cookies": "🍪",
    "Soft Flour Tortilla Wraps": "🫓",
    # Cereals and grains
    "Whole Wheat Flour (Atta)": "🌾",
    "Polished Basmati Rice": "🍚",
    "Rolled Oats & Oatmeal": "🥣",
    "Corn Flakes Breakfast Cereal": "🥣",
    "Pearl Millet Flour (Bajra)": "🌾",
    "Dry Durum Wheat Pasta": "🍝",
    # Dairy products
    "Whole Milk Powder (Full Cream)": "🥛",
    "Cheddar Cheese Block": "🧀",
    "Salted Butter": "🧈",
    "Whey Protein Isolate Powder": "🥛",
    "Shredded Mozzarella Cheese": "🧀",
    "Fresh Paneer (Cottage Cheese)": "🧀",
    # Fruits and vegetables
    "Fresh Cut Apple Slices": "🍎",
    "Strawberries & Soft Berries": "🍓",
    "Fresh Baby Spinach Leaves": "🥬",
    "Fresh Button Mushrooms": "🍄",
    "Cherry Tomatoes": "🍅",
    "Seedless Table Grapes": "🍇",
    # Pulses
    "Green Lentils & Chickpeas": "🫘",
    "Split Red Lentils (Masoor Dal)": "🫘",
    "Whole Black Gram (Urad Dal)": "🫘",
    "Dried Red Kidney Beans (Rajma)": "🫘",
    "Roasted Salted Peanuts": "🥜",
    "Split Yellow Pigeon Peas (Toor Dal)": "🫘",
    # Snacks
    "Fried Potato Chips & Crisps": "🥔",
    "Roasted Cashews & Almonds": "🥜",
    "Extruded Corn Puffs": "🍿",
    "Spiced Banana Chips": "🍌",
    "Granola & Seed Snack Bar": "🍫",
    "Rice Crackers & Rice Cakes": "🍘",
}


def get_food_image_path(category_or_name: str, product_name: str = "", image_path: str = "") -> str:
    """Return local image path for food category or preset product with dedicated dictionary mapping and graceful fallback."""
    # 0. Direct image path if specified and exists
    if image_path and os.path.exists(image_path):
        return image_path

    # 1. Product slug check in assets/products/
    if product_name:
        import re
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", product_name).strip("_").lower()
        p_slug = os.path.join("assets", "products", f"{slug}.jpg")
        if os.path.exists(p_slug):
            return p_slug

    # 2. Exact match by product name in FOOD_PRODUCT_IMAGE_MAP
    if product_name and product_name in FOOD_PRODUCT_IMAGE_MAP:
        p = FOOD_PRODUCT_IMAGE_MAP[product_name]
        if os.path.exists(p):
            return p

    # 3. Exact match by category_or_name
    if category_or_name in FOOD_PRODUCT_IMAGE_MAP:
        p = FOOD_PRODUCT_IMAGE_MAP[category_or_name]
        if os.path.exists(p):
            return p

    search_term = (str(product_name) + " " + str(category_or_name)).lower()

    # 4. Product keyword-based matching
    if "chip" in search_term or "crisp" in search_term:
        p = "assets/products/fried_potato_chips_crisps.jpg"
    elif "cashew" in search_term or "almond" in search_term or "nut" in search_term:
        p = "assets/products/roasted_cashews_almonds.jpg"
    elif "strawberr" in search_term or "berr" in search_term:
        p = "assets/products/strawberries_soft_berries.jpg"
    elif "cheese" in search_term:
        p = "assets/products/cheddar_cheese_block.jpg"
    elif "bread" in search_term or "toast" in search_term:
        p = "assets/products/white_sandwich_bread.jpg"
    elif "biscuit" in search_term or "cookie" in search_term:
        p = "assets/products/digestive_biscuits_cookies.jpg"
    elif "rice" in search_term or "basmati" in search_term:
        p = "assets/products/polished_basmati_rice.jpg"
    elif "flour" in search_term or "atta" in search_term or "wheat" in search_term:
        p = "assets/products/whole_wheat_flour_atta.jpg"
    elif "butter" in search_term:
        p = "assets/products/salted_butter.jpg"
    elif "milk" in search_term or "powder" in search_term:
        p = "assets/products/whole_milk_powder_full_cream.jpg"
    elif "apple" in search_term:
        p = "assets/products/fresh_cut_apple_slices.jpg"
    elif "lentil" in search_term or "dal" in search_term or "chickpea" in search_term or "pulse" in search_term:
        p = "assets/products/green_lentils_chickpeas.jpg"
    elif category_or_name in FOOD_CATEGORY_IMAGE_MAP:
        p = FOOD_CATEGORY_IMAGE_MAP[category_or_name]
    else:
        p = "assets/products/placeholder_food.jpg"

    if os.path.exists(p):
        return p
    if os.path.exists("assets/products/placeholder_food.jpg"):
        return "assets/products/placeholder_food.jpg"
    return "assets/foods/snacks.svg" if os.path.exists("assets/foods/snacks.svg") else ""


def get_material_image_path(mat_name: str, mat_obj: Optional[Dict[str, Any]] = None) -> str:
    """Return local image path for packaging material, resolving exact 16 assets with clean SVG/BOPP fallback."""
    if mat_obj and mat_obj.get("image_path") and os.path.exists(mat_obj["image_path"]):
        return mat_obj["image_path"]

    name_lower = str(mat_name).lower()

    if "glass" in name_lower:
        slug = "glass_container"
    elif "aluminium foil" in name_lower or "al foil" in name_lower:
        slug = "alu_foil_laminate"
    elif "tri-laminate" in name_lower or "bopp / met-bopp" in name_lower:
        slug = "bopp_met_bopp_pe"
    elif "met-bopp" in name_lower or "metallized bopp" in name_lower:
        slug = "met_bopp"
    elif "met-pet" in name_lower or "metallized pet" in name_lower:
        slug = "met_pet"
    elif "evoh" in name_lower:
        slug = "pet_evoh_pe"
    elif "bopa" in name_lower or "nylon" in name_lower:
        slug = "bopa_pe_laminate"
    elif "paperboard" in name_lower or "carton" in name_lower:
        slug = "paperboard_pe"
    elif "rpet" in name_lower or "recycled pet" in name_lower:
        slug = "rpet_film"
    elif "pet film" in name_lower:
        slug = "pet_film"
    elif "micro-perforated" in name_lower or "perforated" in name_lower:
        slug = "microperf_bopp"
    elif "natureflex" in name_lower or "cellulose" in name_lower:
        slug = "cellulose_natureflex"
    elif "pla" in name_lower or "compost" in name_lower:
        slug = "pla_bioplastic"
    elif "hdpe" in name_lower:
        slug = "hdpe_blown"
    elif "ldpe" in name_lower:
        slug = "ldpe_monolayer"
    elif "bopp" in name_lower:
        slug = "bopp_film"
    else:
        slug = "bopp_film"

    for cand in [f"assets/materials/{slug}.jpg", f"assets/materials/{slug}.svg", "assets/materials/bopp_film.jpg"]:
        if os.path.exists(cand):
            return cand
    return ""



def get_image_base64(path: str) -> str:
    """Return base64 data URI for embedding local images in HTML."""
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                mime = "image/svg+xml" if path.endswith(".svg") else "image/jpeg" if (path.endswith(".jpg") or path.endswith(".jpeg")) else "image/png"
                return f"data:{mime};base64,{encoded}"
        except Exception:
            return ""
    return ""


def render_html(html_str: str) -> None:
    """
    Render HTML cleanly in Streamlit without triggering Markdown indented code blocks or blank line breaks.
    Uses textwrap.dedent, strips whitespace per line, and eliminates empty lines so CommonMark
    never interprets HTML elements as indented code blocks (<pre><code>).
    """
    if not html_str:
        return
    dedented = textwrap.dedent(str(html_str))
    cleaned = "\n".join(line.strip() for line in dedented.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Premium FoodTech AI Design System (CSS Loaded from assets/styles.css)
# Palette: Warm Cream (#FFFDF5 / #FFF8E1), Vibrant Orange (#FF8A00 / #FF5722),
#          Deep Espresso Brown (#3E2723), Amber Gold (#FFB300), Eco Green (#2E7D32)
# -----------------------------------------------------------------------------
def load_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "analysis_profile" not in st.session_state:
    st.session_state["analysis_profile"] = None
if "analysis_weights" not in st.session_state:
    st.session_state["analysis_weights"] = None
if "advisor_category" not in st.session_state:
    st.session_state["advisor_category"] = "Bakery products"
if "advisor_product_name" not in st.session_state:
    st.session_state["advisor_product_name"] = None
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "🎯 Packaging Advisor"
if "material_shortlist" not in st.session_state:
    st.session_state["material_shortlist"] = []
if "compare_basket" not in st.session_state:
    st.session_state["compare_basket"] = []
if "show_compare_dialog" not in st.session_state:
    st.session_state["show_compare_dialog"] = False
if "scroll_to_workspace" not in st.session_state:
    st.session_state["scroll_to_workspace"] = False


# -----------------------------------------------------------------------------
# Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_materials():
    return database.get_all_materials()

@st.cache_data(show_spinner=False)
def load_all_presets():
    return database.get_all_commodity_presets()

all_materials = load_all_materials()
all_presets = load_all_presets()

# Preset lookup by name
preset_by_name = {p["name"]: p for p in all_presets}
categories = sorted(list(set(p["category"] for p in all_presets))) + ["Custom Profile"]
all_mat_names = [m["name"] for m in all_materials]
current_result = st.session_state.get("analysis_result")


# -----------------------------------------------------------------------------
# Material Card Mini Metric Bars & Visual Formatting Helpers
# -----------------------------------------------------------------------------
def compute_mini_bars(mat: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute 4 normalized mini metric bars (WVTR, OTR, Cost, CO2e) where higher/fuller is always better."""
    # 1. WVTR: 0.001 to 180 g/m²/d (lower is better)
    wvtr = float(mat.get("wvtr_g_m2_day", 10.0))
    wvtr_pct = max(10.0, min(100.0, 100.0 - ((np.log10(max(0.001, wvtr)) - (-3.0)) / (2.3 - (-3.0))) * 85.0))
    wvtr_color = "#2E7D32" if wvtr_pct >= 75 else "#E9B44C" if wvtr_pct >= 45 else "#94A3B8"

    # 2. OTR: 0.001 to 8500 cc/m²/d/atm (lower is better)
    otr = float(mat.get("otr_cc_m2_day_atm", 100.0))
    otr_pct = max(10.0, min(100.0, 100.0 - ((np.log10(max(0.001, otr)) - (-3.0)) / (4.0 - (-3.0))) * 85.0))
    otr_color = "#2E7D32" if otr_pct >= 75 else "#E9B44C" if otr_pct >= 45 else "#94A3B8"

    # 3. Cost: $1.75 to $5.80 / kg (lower is better)
    cost = float(mat.get("cost_index_usd_per_kg", 3.0) or 3.0)
    cost_pct = max(12.0, min(100.0, 100.0 - ((cost - 1.5) / (6.0 - 1.5)) * 82.0))
    cost_color = "#2E7D32" if cost_pct >= 75 else "#E9B44C" if cost_pct >= 45 else "#94A3B8"

    # 4. CO2e: 1.15 to 5.20 kg CO2e / kg (lower is better)
    co2 = float(mat.get("carbon_footprint_kg_co2_per_kg", 2.5) or 2.5)
    co2_pct = max(12.0, min(100.0, 100.0 - ((co2 - 1.0) / (5.5 - 1.0)) * 82.0))
    co2_color = "#2E7D32" if co2_pct >= 75 else "#E9B44C" if co2_pct >= 45 else "#94A3B8"

    return [
        {"name": "Moisture Barrier", "val_str": f"{wvtr} g/m²/d", "pct": wvtr_pct, "color": wvtr_color},
        {"name": "Oxygen Barrier", "val_str": f"{otr} cc/m²/d", "pct": otr_pct, "color": otr_color},
        {"name": "Cost Economy", "val_str": f"${cost:.2f} / kg", "pct": cost_pct, "color": cost_color},
        {"name": "Carbon Footprint", "val_str": f"{co2:.2f} kg CO₂e", "pct": co2_pct, "color": co2_color},
    ]


def get_recyclability_pill(status: str) -> str:
    """Return stylized HTML badge with color for recyclability stream."""
    st_low = str(status).lower()
    if "curbside" in st_low or "mono" in st_low or ("stream" in st_low and "specialized" not in st_low):
        return "<span class='meta-badge' style='background:#E8F5E9; color:#2E7D32; font-weight:700; border: 1px solid rgba(46, 125, 50, 0.3); font-size: 0.70rem;'>♻️ Curbside Recyclable</span>"
    elif "compost" in st_low or "bio" in st_low:
        return "<span class='meta-badge' style='background:#FFF3E0; color:#E65100; font-weight:700; border: 1px solid rgba(230, 81, 0, 0.3); font-size: 0.70rem;'>🍂 Industrially Compostable</span>"
    else:
        return "<span class='meta-badge' style='background:#FFEBEE; color:#C62828; font-weight:700; border: 1px solid rgba(198, 40, 40, 0.3); font-size: 0.70rem;'>⚠️ Specialized Stream</span>"


def render_full_width_comparison_matrix(materials: List[Dict[str, Any]]) -> str:
    """Build readable, full-width HTML comparison matrix with sticky first column and green winner highlights."""
    if not materials:
        return ""

    min_wvtr = min(float(m.get("wvtr_g_m2_day", 9999)) for m in materials)
    min_otr = min(float(m.get("otr_cc_m2_day_atm", 9999)) for m in materials)
    min_cost = min(float(m.get("cost_index_usd_per_kg", 9999) or 9999) for m in materials)
    min_co2 = min(float(m.get("carbon_footprint_kg_co2_per_kg", 999) or 999) for m in materials)
    max_tensile = max(float(m.get("tensile_strength_mpa") or 0) for m in materials)

    rows = [
        ("Structure Type", [m.get("structure_type", "—") for m in materials], None),
        ("Nominal Thickness", [f"{m.get('nominal_thickness_um', '—')} µm" for m in materials], None),
        ("Water Vapor Transmission (WVTR)", [
            f"<span class='cmp-matrix-winner-val'>⭐ {m.get('wvtr_g_m2_day')}</span> g/m²/d" if float(m.get("wvtr_g_m2_day", 9999)) == min_wvtr
            else f"{m.get('wvtr_g_m2_day')} g/m²/d" for m in materials
        ], "↓ lower is better (38°C, 90% RH)"),
        ("Oxygen Transmission (OTR)", [
            f"<span class='cmp-matrix-winner-val'>⭐ {m.get('otr_cc_m2_day_atm')}</span> cc/m²/d" if float(m.get("otr_cc_m2_day_atm", 9999)) == min_otr
            else f"{m.get('otr_cc_m2_day_atm')} cc/m²/d" for m in materials
        ], "↓ lower is better (23°C, 0% RH)"),
        ("Unit Cost Index", [
            f"<span class='cmp-matrix-winner-val'>⭐ ${float(m.get('cost_index_usd_per_kg') or 0):.2f}</span> / kg" if m.get("cost_index_usd_per_kg") and float(m.get("cost_index_usd_per_kg")) == min_cost
            else (f"${float(m.get('cost_index_usd_per_kg') or 0):.2f} / kg" if m.get("cost_index_usd_per_kg") is not None else "Unreported") for m in materials
        ], "↓ lower is better"),
        ("Embodied Carbon Footprint", [
            f"<span class='cmp-matrix-winner-val'>⭐ {float(m.get('carbon_footprint_kg_co2_per_kg') or 0):.2f}</span> kg CO₂e" if m.get("carbon_footprint_kg_co2_per_kg") and float(m.get("carbon_footprint_kg_co2_per_kg")) == min_co2
            else (f"{float(m.get('carbon_footprint_kg_co2_per_kg') or 0):.2f} kg CO₂e" if m.get("carbon_footprint_kg_co2_per_kg") is not None else "Unreported") for m in materials
        ], "↓ lower is better"),
        ("Tensile Strength", [
            f"<span class='cmp-matrix-winner-val'>⭐ {float(m.get('tensile_strength_mpa') or 0):.1f}</span> MPa" if m.get("tensile_strength_mpa") and float(m.get("tensile_strength_mpa")) == max_tensile
            else (f"{float(m.get('tensile_strength_mpa') or 0):.1f} MPa" if m.get("tensile_strength_mpa") is not None else "Unreported") for m in materials
        ], "↑ higher is better"),
        ("Circularity & Recyclability", [get_recyclability_pill(m.get("recyclability_status", "")) for m in materials], None),
        ("Food Contact Compliance", [m.get("food_contact_compliance", "Compliant") for m in materials], None),
        ("Standard Test Conditions", [m.get("test_conditions", "ASTM F1249 / D3985") for m in materials], None),
        ("Literature Data Source", [m.get("data_source", "Published Literature") for m in materials], None)
    ]

    html = '<div style="overflow-x: auto; width: 100%; border-radius: 8px; margin-top: 10px; margin-bottom: 16px;">'
    html += '<table class="cmp-matrix-table">'
    html += '<thead><tr>'
    html += '<th style="min-width: 190px; position: sticky; left: 0; z-index: 2; background: var(--forest-deep);">Property / Specification</th>'
    for m in materials:
        m_img = get_image_base64(get_material_image_path(m["name"]))
        thumb_tag = f'<img src="{m_img}" style="width: 24px; height: 24px; object-fit: cover; border-radius: 4px; vertical-align: middle; margin-right: 6px;" />' if m_img else ''
        html += f'<th style="min-width: 180px;">{thumb_tag}<b>{m["name"].split("(")[0].strip()}</b></th>'
    html += '</tr></thead><tbody>'

    for prop_name, vals, direction in rows:
        dir_badge = f' <span style="font-size: 0.68rem; color: #536058; font-weight: normal; display: block;">{direction}</span>' if direction else ''
        html += f'<tr><td style="font-weight: 700; position: sticky; left: 0; background: #F8FAFC; z-index: 1;">{prop_name}{dir_badge}</td>'
        for v in vals:
            html += f'<td>{v}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html


# -----------------------------------------------------------------------------
# Universal Side-by-Side Material Comparison Modal Dialog (st.dialog)
# -----------------------------------------------------------------------------
@st.dialog("⚖️ Side-by-Side Material Comparison", width="large")
def compare_materials_dialog(selected_material_names: List[str], active_food: Optional[Dict[str, Any]] = None):
    """Modal comparison dialog opening directly in-place without page scroll or tab change."""
    selected_mats = [m for m in all_materials if m["name"] in selected_material_names]
    if len(selected_mats) < 2:
        st.warning("Please add at least 2 materials to compare.")
        return

    st.markdown(f"#### Comparing {len(selected_mats)} Benchmark Packaging Materials")
    if active_food:
        st.caption(f"Evaluating physical preservation suitability for **{active_food.get('name', 'Selected Product')}**.")

    # 1. Specimen Cards side by side
    c_cols = st.columns(len(selected_mats))
    for col, m in zip(c_cols, selected_mats):
        with col:
            img_p = get_material_image_path(m["name"])
            img_b64 = get_image_base64(img_p)
            bars = compute_mini_bars(m)
            rec_pill = get_recyclability_pill(m.get("recyclability_status", ""))

            bars_html = "".join(f"""
            <div class="mini-metric-row">
                <div class="mini-metric-label-val">
                    <span class="mini-metric-label">{b['name']}</span>
                    <span class="mini-metric-val">{b['val_str']}</span>
                </div>
                <div class="mini-bar-track">
                    <div class="mini-bar-fill" style="width: {b['pct']:.1f}%; background-color: {b['color']};"></div>
                </div>
            </div>
            """ for b in bars)

            render_html(f"""
            <div class="mat-specimen-card">
                <img src="{img_b64}" class="mat-specimen-img" />
                <div class="mat-specimen-body">
                    <div class="mat-specimen-name">{m['name'].split('(')[0].strip()}</div>
                    <div style="display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 6px;">
                        <span class="meta-badge" style="font-size: 0.68rem;">{m.get('structure_type', 'Flexible')}</span>
                        <span class="meta-badge" style="font-size: 0.68rem;">{m.get('nominal_thickness_um', '-')} µm</span>
                    </div>
                    <div class="mini-metrics-container">{bars_html}</div>
                    <div style="margin-top: 4px;">{rec_pill}</div>
                </div>
            </div>
            """)
            if st.button("✕ Remove", key=f"btn_dlg_rm_{m['name'][:10]}", use_container_width=True):
                st.session_state["compare_basket"].remove(m["name"])
                st.rerun()

    # 2. Category Leaders
    best_b = min(selected_mats, key=lambda x: float(x.get("wvtr_g_m2_day", 9999)))
    best_c = min(selected_mats, key=lambda x: float(x.get("cost_index_usd_per_kg", 9999) or 9999))
    best_e = min(selected_mats, key=lambda x: float(x.get("carbon_footprint_kg_co2_per_kg", 999) or 999))

    render_html(f"""
    <div class="winner-summary-grid" style="margin: 16px 0;">
        <div class="winner-highlight-card">
            <span class="winner-badge-pill" style="background: rgba(33, 150, 243, 0.12); color: #0288D1;">🛡️ Best Barrier</span>
            <div style="font-weight: 800; font-size: 0.90rem; color: var(--text-main); margin-bottom: 2px;">{best_b['name'].split('(')[0].strip()}</div>
            <div style="font-size: 0.74rem; color: var(--text-muted);">WVTR: <b>{best_b.get('wvtr_g_m2_day')}</b> g/m²/d</div>
        </div>
        <div class="winner-highlight-card">
            <span class="winner-badge-pill" style="background: rgba(76, 175, 80, 0.12); color: #2E7D32;">💰 Most Economical</span>
            <div style="font-weight: 800; font-size: 0.90rem; color: var(--text-main); margin-bottom: 2px;">{best_c['name'].split('(')[0].strip()}</div>
            <div style="font-size: 0.74rem; color: var(--text-muted);">Cost: <b>${float(best_c.get('cost_index_usd_per_kg', 0) or 0):.2f}</b> / kg</div>
        </div>
        <div class="winner-highlight-card">
            <span class="winner-badge-pill" style="background: rgba(46, 125, 50, 0.12); color: #1B5E20;">🌱 Most Eco-Friendly</span>
            <div style="font-weight: 800; font-size: 0.90rem; color: var(--text-main); margin-bottom: 2px;">{best_e['name'].split('(')[0].strip()}</div>
            <div style="font-size: 0.74rem; color: var(--text-muted);">Carbon: <b>{best_e.get('carbon_footprint_kg_co2_per_kg', '-')}</b> kg CO₂e</div>
        </div>
    </div>
    """)

    # 3. Interactive Charts
    st.markdown("##### Visual Analytics: Multi-Axis Radar & Property Breakdown")
    c_tab1, c_tab2 = st.tabs(["🕸️ Multidimensional Radar", "📊 Metric Subplots (WVTR, OTR, Cost, CO₂e)"])
    with c_tab1:
        fig_r, _ = analytics.create_radar_comparison_chart(selected_mats)
        st.plotly_chart(fig_r, use_container_width=True)
    with c_tab2:
        fig_s = analytics.create_comparator_subplots(selected_mats)
        st.plotly_chart(fig_s, use_container_width=True)

    # 4. Full-width comparison matrix
    st.markdown("##### Full Specification Comparison Matrix")
    matrix_html = render_full_width_comparison_matrix(selected_mats)
    render_html(matrix_html)

    # 5. Export Actions
    exp_c1, exp_c2, exp_c3 = st.columns([1, 1, 0.8], gap="small")
    with exp_c1:
        csv_rows = []
        for m in selected_mats:
            csv_rows.append({
                "Material": m["name"],
                "Structure": m.get("structure_type", ""),
                "Thickness (um)": m.get("nominal_thickness_um", ""),
                "WVTR (g/m2/d)": m.get("wvtr_g_m2_day", ""),
                "OTR (cc/m2/d)": m.get("otr_cc_m2_day_atm", ""),
                "Cost ($/kg)": m.get("cost_index_usd_per_kg", ""),
                "Carbon (kg CO2e)": m.get("carbon_footprint_kg_co2_per_kg", ""),
                "Tensile (MPa)": m.get("tensile_strength_mpa", ""),
                "Recyclability": m.get("recyclability_status", ""),
                "Food Contact": m.get("food_contact_compliance", "")
            })
        csv_b = pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export CSV", data=csv_b, file_name="FoodPackAI_Comparison.csv", mime="text/csv", type="primary", use_container_width=True)

    with exp_c2:
        try:
            food_lbl = active_food.get("name", "Comparative Audit") if active_food else "Comparative Audit"
            pdf_b = pdf_generator.generate_comparator_pdf(selected_mats, food_lbl)
            st.download_button("📄 Export PDF Report", data=pdf_b, file_name="FoodPackAI_Comparison.pdf", mime="application/pdf", type="primary", use_container_width=True)
        except Exception as e:
            st.error(f"PDF export error: {e}")

    with exp_c3:
        if st.button("Close Dialog", use_container_width=True):
            st.session_state["show_compare_dialog"] = False
            st.rerun()


# Deep-linking & testing query parameters support
query_p = st.query_params
if "cat" in query_p and query_p["cat"] in categories:
    st.session_state["advisor_category"] = query_p["cat"]
if "prod" in query_p and (query_p["prod"] in preset_by_name or query_p.get("cat") == "Custom Profile"):
    st.session_state["advisor_product_name"] = query_p["prod"]
if "tab" in query_p:
    tab_val = query_p["tab"]
    for t in ["🎯 Packaging Advisor", "⚖️ Material Comparator", "⏱️ Shelf-Life Simulator", "🔬 Multi-Criteria Explorer"]:
        if tab_val.lower() in t.lower():
            st.session_state["active_tab"] = t
            st.session_state["main_nav_pills"] = t
if "basket" in query_p:
    b_items = [item.strip() for item in query_p["basket"].split(";") if item.strip()]
    st.session_state["compare_basket"] = b_items[:4]
if "show_dialog" in query_p and query_p["show_dialog"] == "1":
    st.session_state["show_compare_dialog"] = True
if "autorun" in query_p and query_p["autorun"] == "1" and st.session_state.get("analysis_result") is None:
    p_name = st.session_state.get("advisor_product_name") or "Dry Digestive Biscuits / Cookies"
    active_p = preset_by_name.get(p_name, all_presets[0] if all_presets else {})
    if active_p:
        c_input = {
            "name": active_p.get("name", "Selected Food"),
            "category": active_p.get("category", "General"),
            "moisture_sensitivity": active_p.get("moisture_sensitivity", "Moderate"),
            "oxygen_sensitivity": active_p.get("oxygen_sensitivity", "Moderate"),
            "fat_content_level": active_p.get("fat_content_level", "Low"),
            "light_sensitivity": active_p.get("light_sensitivity", "Low"),
            "storage_temp_c": 23.0,
            "storage_rh_percent": 65.0,
            "target_shelf_life_days": int(active_p.get("target_shelf_life_days", 180) or 180),
            "puncture_risk": active_p.get("puncture_risk", "Low")
        }
        w_dict = {"barrier": 0.40, "cost": 0.25, "sustainability": 0.25, "mechanical": 0.10}
        res = recommender.rank_recommendations(all_materials, c_input, w_dict)
        st.session_state["analysis_result"] = res
        st.session_state["analysis_profile"] = c_input
        st.session_state["analysis_weights"] = w_dict


# -----------------------------------------------------------------------------
# Collapsible Sidebar (Platform Info, Compare Basket, Shortlist, History)
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("assets/logo.svg"):
        st.image("assets/logo.svg", use_container_width=True)
    elif os.path.exists("assets/logo_mark.svg"):
        st.image("assets/logo_mark.svg", width=80)
        st.markdown("### Smart FoodPack AI")
    else:
        st.markdown("### 📦 Smart FoodPack AI")
    
    render_html(
        "<p style='color: var(--text-muted); font-size: 0.85rem; margin-top: -6px; margin-bottom: 12px; line-height: 1.45;'>"
        "Intelligent packaging decision support for shelf-life protection and circularity."
        "</p>"
    )
    render_html("<hr style='border: 0; border-top: 1px solid var(--border-color); margin: 8px 0 12px 0;' />")

    # 1. ALWAYS-VISIBLE COMPARE BASKET (Max 4 items with thumbnails, remove X, Compare Now)
    compare_basket = st.session_state.get("compare_basket", [])
    basket_count = len(compare_basket)

    render_html(f"""
    <div class="compare-basket-box">
        <div class="compare-basket-header">
            <span class="compare-basket-title">⚖️ Compare Basket</span>
            <span class="compare-basket-badge">{basket_count} / 4</span>
        </div>
    </div>
    """)

    if not compare_basket:
        render_html("<p style='font-size: 0.76rem; color: var(--text-muted); margin: 2px 0 8px 0;'>No materials selected. Click <b>➕ Add to Compare</b> on any material card (up to 4).</p>")
    else:
        for b_idx, b_mat_name in enumerate(list(compare_basket)):
            b_col1, b_col2 = st.columns([4.2, 1])
            with b_col1:
                b_img_path = get_material_image_path(b_mat_name)
                b_img_b64 = get_image_base64(b_img_path)
                b_thumb = f"<img src='{b_img_b64}' class='compare-basket-thumb' />" if b_img_b64 else "<div style='width:32px;height:32px;background:#E2E8F0;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:0.8rem;'>📦</div>"
                render_html(f"""
                <div class="compare-basket-item" style="margin-bottom: 4px;">
                    {b_thumb}
                    <div class="compare-basket-title" title="{b_mat_name}">{b_mat_name.split('(')[0].strip()}</div>
                </div>
                """)
            with b_col2:
                if st.button("✕", key=f"btn_rm_bsk_{b_idx}_{b_mat_name[:8]}", help=f"Remove {b_mat_name}"):
                    st.session_state["compare_basket"].remove(b_mat_name)
                    st.rerun()

        # Action Buttons
        render_html("<div style='margin-top: 8px;'></div>")
        cmp_now_btn = st.button(
            f"🚀 Compare Now ({basket_count})",
            type="primary",
            use_container_width=True,
            key="btn_sidebar_compare_now",
            disabled=(basket_count < 2),
            help="Compare selected materials side-by-side in dialog" if basket_count >= 2 else "Select at least 2 materials to compare"
        )
        if cmp_now_btn:
            st.session_state["show_compare_dialog"] = True
            st.rerun()

        if st.button("Clear Basket", use_container_width=True, key="btn_clear_basket"):
            st.session_state["compare_basket"] = []
            st.rerun()

    render_html("<hr style='border: 0; border-top: 1px solid var(--border-color); margin: 12px 0 12px 0;' />")

    # 2. Saved Material Shortlist Section
    shortlist = st.session_state.get("material_shortlist", [])
    with st.expander(f"⭐ Saved Shortlist ({len(shortlist)})", expanded=bool(shortlist)):
        if not shortlist:
            render_html("<p style='font-size: 0.78rem; color: var(--text-muted); margin: 0;'>No materials saved yet. Click <b>⭐ Save to Shortlist</b> in recommendations or the comparator.</p>")
        else:
            for s_idx, s_mat in enumerate(list(shortlist)):
                sc1, sc2 = st.columns([3.8, 1])
                with sc1:
                    render_html(f"<div style='font-size: 0.78rem; font-weight: 700; color: var(--text-main); line-height: 1.3;'>{s_mat.split('(')[0].strip()}</div>")
                with sc2:
                    if st.button("✕", key=f"slist_del_{s_idx}", help=f"Remove {s_mat}"):
                        st.session_state["material_shortlist"].remove(s_mat)
                        st.rerun()
            render_html("<div style='margin-top: 6px;'></div>")
            if len(shortlist) >= 2:
                if st.button("⚖️ Compare Shortlist", use_container_width=True, key="btn_shortlist_compare"):
                    st.session_state["comparator_selected_mats"] = list(shortlist[:4])
                    st.session_state["active_tab"] = "⚖️ Material Comparator"
                    st.session_state["main_nav_pills"] = "⚖️ Material Comparator"
                    st.session_state["scroll_to_workspace"] = True
                    st.rerun()

    # 3. Recent Analysis History Section
    with st.expander("🕒 Recent Analysis History", expanded=False):
        recent_runs = database.get_recommendation_history(limit=5)
        if not recent_runs:
            render_html("<p style='font-size: 0.78rem; color: var(--text-muted); margin: 0;'>No past analysis runs recorded in audit log yet.</p>")
        else:
            for r in recent_runs:
                c_name = r.get("commodity_name", "Commodity")
                c_cat = r.get("category", "")
                c_ts = str(r.get("created_at", ""))
                ts_disp = c_ts[:16].replace("T", " ") if c_ts else ""
                render_html(f"""
                <div style='background: #FFFFFF; border: 1px solid var(--border-color); border-radius: 6px; padding: 6px 10px; margin-bottom: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);'>
                    <div style='font-weight: 700; font-size: 0.80rem; color: var(--text-main);'>{c_name}</div>
                    <div style='font-size: 0.70rem; color: var(--text-muted); display: flex; justify-content: space-between; margin-top: 2px;'>
                        <span>{c_cat}</span>
                        <span>{ts_disp}</span>
                    </div>
                </div>
                """)

    with st.expander("🔬 Decision Methodology & Standards", expanded=False):
        render_html(
            "<ul style='font-size: 0.80rem; color: var(--text-muted); padding-left: 16px; margin-bottom: 4px; line-height: 1.5;'>"
            "<li><b>Stage-1 Physics Knockout:</b> ASTM F1249 / D3985 barrier screening and thermal limits.</li>"
            "<li><b>Stage-2 MCDA Engine:</b> Transparent Multi-Criteria Decision Analysis (Barrier, Cost, Eco, Mechanics).</li>"
            "<li><b>Kinetics Engine:</b> Steady-state Fickian diffusion mass transfer (Robertson &amp; Labuza).</li>"
            "</ul>"
        )

    render_html(
        f"<div style='font-size: 0.80rem; color: var(--text-muted); background: var(--card-bg); padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border-color); line-height: 1.5; margin-bottom: 14px;'>"
        f"<div style='font-weight: 700; color: var(--text-main); font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px;'>📊 Knowledge Repository</div>"
        f"&bull; <b>{len(all_materials)} Literature Benchmarks</b> (ASTM / FDA)<br/>"
        f"&bull; <b>{len(all_presets)} Commodity Profiles</b> (Verified sensitivities)"
        f"</div>"
    )

    if st.button("🔄 Reset Analysis Session", use_container_width=True, key="btn_reset_sidebar"):
        st.session_state["analysis_result"] = None
        st.session_state["analysis_profile"] = None
        st.session_state["analysis_weights"] = None
        st.rerun()


# -----------------------------------------------------------------------------
# Top Header Banner with Logo & Standards Badges
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([2.5, 1.5])
with header_col1:
    if os.path.exists("assets/logo.svg"):
        st.image("assets/logo.svg", width=420)
    else:
        render_html(
            "<h1>Smart FoodPack <span class='gradient-heading'>AI</span></h1>"
            "<p style='color: var(--text-muted); font-size: 1.05rem; margin-top: -12px; margin-bottom: 8px;'>"
            "AI-Based Intelligent Food Packaging Material Recommendation & Shelf-Life Simulation System"
            "</p>"
        )
with header_col2:
    render_html(
        "<div style='text-align: right; padding-top: 10px;'>"
        "<span class='meta-badge' style='background: var(--primary-light); color: var(--primary-deep); font-weight: 700;'>ASTM F1249 / D3985</span>"
        "<span class='meta-badge'>FDA 21 CFR §177</span>"
        "<span class='meta-badge'>EU 10/2011</span>"
        "<span class='meta-badge'>FSSAI 2018</span>"
        "<p style='color: var(--text-muted); font-size: 0.74rem; margin-top: 6px; margin-bottom: 0;'>"
        "MCDA Decision-Support &bull; ASTM/FDA/EU Criteria &bull; Comparative Guidance"
        "</p>"
        "</div>"
    )

# -----------------------------------------------------------------------------
# Premium 3D Hero Section Banner with Real Interactive 3D Buttons
# -----------------------------------------------------------------------------
with st.container(key="hero_container"):
    hero_col1, hero_col2 = st.columns([1.65, 1.0], gap="large")
    with hero_col1:
        render_html(
            """
            <div class="hero-badge">🌱 ORGANIC INTELLIGENCE × ADVANCED MATERIALS SCIENCE</div>
            <div class="hero-title">
                Engineer Better Packaging.<br/><span class="gradient-heading">Preserve More Food.</span>
            </div>
            <div class="hero-subtitle">
                An intelligent research platform for exploring packaging materials, evaluating protection requirements, comparing alternatives, and making evidence-aware food preservation decisions.
            </div>
            """
        )
        hbtn_c1, hbtn_c2 = st.columns([1.1, 1.25])
        with hbtn_c1:
            hero_start_clicked = st.button("🚀 Start Analysis", key="hero_btn_start", use_container_width=True)
        with hbtn_c2:
            hero_compare_clicked = st.button("⚖️ Compare Materials", key="hero_btn_compare", use_container_width=True)

        if hero_start_clicked:
            st.session_state["active_tab"] = "🎯 Packaging Advisor"
            st.session_state["main_nav_pills"] = "🎯 Packaging Advisor"
            st.session_state["scroll_to_workspace"] = True
            st.rerun()

        if hero_compare_clicked:
            st.session_state["active_tab"] = "⚖️ Material Comparator"
            st.session_state["main_nav_pills"] = "⚖️ Material Comparator"
            st.session_state["scroll_to_workspace"] = True
            st.rerun()

        total_commodities_count = len(all_presets)
        render_html(
            f"""
            <div class="hero-stats-row">
                <div class="hero-stat-box">
                    <span class="hero-stat-number">{len(all_materials)}+</span>
                    <span class="hero-stat-label">ASTM Benchmarks</span>
                </div>
                <div class="hero-stat-box">
                    <span class="hero-stat-number">{total_commodities_count}</span>
                    <span class="hero-stat-label">Food Commodities</span>
                </div>
                <div class="hero-stat-box">
                    <span class="hero-stat-number">4</span>
                    <span class="hero-stat-label">Global Standards</span>
                </div>
                <div class="hero-stat-box">
                    <span class="hero-stat-number" style="color: #2E7D32;">100%</span>
                    <span class="hero-stat-label">Fickian Kinetics</span>
                </div>
            </div>
            """
        )
    with hero_col2:
        render_html(
            """
            <div class="hero-3d-visual" style="margin-top: 24px;">
                <div class="cube-scene">
                    <div class="cube-face cube-face-front">
                        <div style="font-size: 32px; margin-bottom: 4px;">📦</div>
                        <div>BARRIER</div>
                        <div style="font-size: 10px; opacity: 0.9; font-weight: normal;">ASTM F1249</div>
                    </div>
                    <div class="cube-face cube-face-back">
                        <div style="font-size: 32px; margin-bottom: 4px;">🌿</div>
                        <div>CIRCULAR</div>
                        <div style="font-size: 10px; opacity: 0.9; font-weight: normal;">Compostable</div>
                    </div>
                    <div class="cube-face cube-face-right">
                        <div style="font-size: 32px; margin-bottom: 4px;">🔬</div>
                        <div>MCDA AI</div>
                        <div style="font-size: 10px; opacity: 0.9; font-weight: normal;">Stage-1 &amp; 2</div>
                    </div>
                    <div class="cube-face cube-face-left">
                        <div style="font-size: 32px; margin-bottom: 4px;">🛡️</div>
                        <div>HERMETIC</div>
                        <div style="font-size: 10px; opacity: 0.9; font-weight: normal;">Oxygen Exclusion</div>
                    </div>
                    <div class="cube-face cube-face-top">
                        <div style="font-size: 26px; margin-bottom: 2px;">⚡</div>
                        <div style="font-size: 11px;">FOOD TECH</div>
                    </div>
                    <div class="cube-face cube-face-bottom">
                        <div style="font-size: 26px; margin-bottom: 2px;">⏱️</div>
                        <div style="font-size: 11px;">SHELF LIFE</div>
                    </div>
                </div>
            </div>
            """
        )

# Smooth scroll target anchor and auto-scroll execution
render_html('<div id="workspace-anchor" style="position: relative; top: -16px;"></div>')
if st.session_state.pop("scroll_to_workspace", False):
    st.components.v1.html(
        """
        <script>
            setTimeout(() => {
                const target = window.parent.document.getElementById('workspace-anchor');
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 100);
        </script>
        """,
        height=0,
        width=0
    )

# -----------------------------------------------------------------------------
# Primary Navigation Pills (3D Custom Segmented Control bound to Session State)
# -----------------------------------------------------------------------------
NAV_TABS = [
    "🎯 Packaging Advisor",
    "⚖️ Material Comparator",
    "⏱️ Shelf-Life Simulator",
    "📦 Materials & Standards"
]

cur_active = st.session_state.get("active_tab", "🎯 Packaging Advisor")
if cur_active not in NAV_TABS:
    cur_active = "🎯 Packaging Advisor"
st.session_state["main_nav_pills"] = cur_active

nav_choice = st.segmented_control(
    "Select Workspace",
    options=NAV_TABS,
    key="main_nav_pills",
    label_visibility="collapsed"
)
if nav_choice and nav_choice != st.session_state["active_tab"]:
    st.session_state["active_tab"] = nav_choice

active_tab = st.session_state["active_tab"]

# Check if Universal Compare Dialog is active
if st.session_state.get("show_compare_dialog"):
    current_food_name = st.session_state.get("advisor_product_name", "Butter Cookie (Shortbread)")
    active_food_preset_obj = preset_by_name.get(current_food_name, all_presets[0])
    compare_materials_dialog(st.session_state.get("compare_basket", []), active_food_preset_obj)


# =============================================================================
# TAB 1: PACKAGING ADVISOR (Primary 5-Step Workflow & Clean Dashboard)
# =============================================================================
if active_tab == "🎯 Packaging Advisor":
    # -------------------------------------------------------------------------
    # A. 6-Step Visual Workflow Roadmap & KPI Metrics
    # -------------------------------------------------------------------------
    current_result = st.session_state.get("analysis_result")
    total_mats_count = len(all_materials)
    s4_act = "active" if current_result else ""
    render_html(f"""
    <div class="workflow-row">
        <div class="workflow-step active">1. Food Commodity</div>
        <div class="workflow-step active">2. Storage Bounds</div>
        <div class="workflow-step active">3. Decision Priorities</div>
        <div class="workflow-step {s4_act}">4. Physics Screening</div>
        <div class="workflow-step {s4_act}">5. Candidate Comparison</div>
        <div class="workflow-step {s4_act}">6. Technical PDF Export</div>
    </div>
    """)

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.metric("Materials in Database", f"{total_mats_count} Benchmarks")

    with kpi_col2:
        if current_result:
            qual = current_result.get("total_qualified", 0)
            st.metric("Eligible Candidates", f"{qual} of {total_mats_count}")
        else:
            st.metric("Eligible Candidates", "Awaiting Analysis")

    with kpi_col3:
        if current_result and current_result.get("top_candidates"):
            top_name = current_result["top_candidates"][0].get("name", "N/A")
            short_name = top_name if len(top_name) <= 24 else top_name[:22] + "…"
            st.metric("Top Recommendation", short_name)
        elif current_result and not current_result.get("top_candidates"):
            st.metric("Top Recommendation", "No Feasible Match")
        else:
            st.metric("Top Recommendation", "Not Selected")

    with kpi_col4:
        if current_result and current_result.get("top_candidates"):
            top_score = current_result["top_candidates"][0].get("total_mcda_score", 0.0)
            st.metric("Top Match Score", f"{top_score:.1f} / 100")
        else:
            st.metric("Top Match Score", "—")

    render_html("<hr style='border: 0; border-top: 1px solid var(--border-color); margin: 18px 0 24px 0;' />")

    # -------------------------------------------------------------------------
    # B. Steps 1 & 2: Food Profile & Preferences (Two-Column Input Grid)
    # -------------------------------------------------------------------------
    step_col1, step_col2 = st.columns([1.05, 0.95], gap="large")

    with step_col1:
        st.markdown("### 1. Select Food Product")
        render_html(
            "<p style='color: var(--text-muted); font-size: 0.88rem; margin-top: -8px; margin-bottom: 12px;'>"
            "Filter verified food commodity profiles by category and click a card to select:"
            "</p>"
        )

        # Category Selectbox with session state sync
        cat_index = categories.index(st.session_state["advisor_category"]) if st.session_state["advisor_category"] in categories else 0
        selected_cat = st.selectbox(
            "Food Category",
            options=categories,
            index=cat_index,
            key="advisor_food_category",
            help="Select food sector to filter verified food commodity profiles."
        )
        if selected_cat != st.session_state.get("advisor_category"):
            st.session_state["advisor_category"] = selected_cat
            # If switching category, reset product selection
            if selected_cat != "Custom Profile":
                st.session_state["advisor_product_name"] = None
            st.rerun()

        # Product selection flow
        if selected_cat != "Custom Profile":
            filtered_presets = [p["name"] for p in all_presets if p["category"] == selected_cat]
            if not filtered_presets:
                filtered_presets = [p["name"] for p in all_presets]

            current_prod = st.session_state.get("advisor_product_name")
            if current_prod and current_prod not in filtered_presets:
                current_prod = None
                st.session_state["advisor_product_name"] = None

            # Interactive Food Product Visual Cards Grid (3 columns)
            if filtered_presets:
                card_cols = st.columns(3)
                for idx, p_name in enumerate(filtered_presets):
                    p_obj = preset_by_name.get(p_name, {})
                    is_active = (p_name == current_prod)
                    with card_cols[idx % 3]:
                        card_class = "food-card-wrapper selected" if is_active else "food-card-wrapper"
                        status_tag = "<span class='food-card-badge'>✓ SELECTED</span>" if is_active else f"<span class='food-card-badge-unselected'>{FOOD_EMOJI_MAP.get(p_name, '📦')} Profile</span>"
                        render_html(f"""
                        <div class="{card_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                {status_tag}
                                <span style="font-size: 0.68rem; color: var(--text-muted); font-weight: 600;">{p_obj.get('target_shelf_life_days', 180)}d target</span>
                            </div>
                            <div class="food-card-title">{p_name}</div>
                        </div>
                        """)
                        card_img = get_food_image_path(p_obj.get("category", selected_cat), p_name, p_obj.get("image_path", ""))
                        if card_img and os.path.exists(card_img):
                            st.image(card_img, use_container_width=True)
                        else:
                            st.image("assets/products/placeholder_food.jpg", use_container_width=True)

                        btn_label = "✓ Selected" if is_active else "Select"
                        btn_type = "primary" if is_active else "secondary"
                        if st.button(btn_label, key=f"fcard_btn_{idx}_{p_name}", type=btn_type, use_container_width=True):
                            st.session_state["advisor_product_name"] = p_name
                            st.session_state["advisor_storage_temp"] = float(p_obj.get("recommended_temp_c", 23.0))
                            st.session_state["advisor_storage_rh"] = int(p_obj.get("recommended_rh_percent", 60))
                            st.session_state["advisor_target_days"] = int(p_obj.get("target_shelf_life_days", 180))
                            st.rerun()

            active_preset = preset_by_name.get(current_prod) if current_prod else None
        else:
            selected_product_name = st.text_input(
                "Custom Product Name",
                value=st.session_state.get("advisor_product_name") or "Custom Food Product",
                key="advisor_custom_name",
                help="Enter custom commodity name."
            )
            st.session_state["advisor_product_name"] = selected_product_name
            
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                c_moist = st.selectbox("Moisture Sensitivity", ["Extreme", "Critical", "High", "Moderate", "Low"], index=2, key="c_moist")
                c_fat = st.selectbox("Fat Content Level", ["Very High", "High", "Medium-High", "Moderate", "Low-Medium", "Low", "Zero"], index=3, key="c_fat")
            with c_col2:
                c_oxy = st.selectbox("Oxygen Sensitivity", ["Critical", "High", "Moderate", "Low"], index=2, key="c_oxy")
                c_aw = st.number_input("Critical Water Activity (aw)", min_value=0.10, max_value=0.99, value=0.60, step=0.05, key="c_aw")
            c_spoilage = st.text_input("Typical Spoilage Mode", value="Moisture absorption & lipid oxidation", key="c_spoilage")
            
            active_preset = {
                "name": selected_product_name,
                "category": "Custom Profile",
                "moisture_sensitivity": c_moist,
                "oxygen_sensitivity": c_oxy,
                "fat_content_level": c_fat,
                "light_sensitivity": "Moderate",
                "recommended_temp_c": 23.0,
                "recommended_rh_percent": 60.0,
                "target_shelf_life_days": 180,
                "puncture_risk": "Low",
                "typical_spoilage_mode": c_spoilage,
                "critical_aw": float(c_aw),
                "image_path": "assets/products/placeholder_food.jpg",
                "source": "Custom User Specification"
            }

        # Food Storage & Environmental Parameters in step_col1
        render_html("<div style='font-size: 0.92rem; font-weight: 700; color: var(--forest-deep); margin-top: 18px; margin-bottom: 8px;'>Storage & Environmental Constraints</div>")
        env_col1, env_col2 = st.columns(2)
        default_temp = float(active_preset.get("recommended_temp_c", 23.0)) if active_preset else 23.0
        default_rh = int(active_preset.get("recommended_rh_percent", 60)) if active_preset else 60
        default_days = int(active_preset.get("target_shelf_life_days", 180)) if active_preset else 180

        with env_col1:
            storage_temp = st.number_input(
                "Storage Temperature (°C)",
                min_value=-25.0,
                max_value=60.0,
                value=float(st.session_state.get("advisor_storage_temp", default_temp)),
                step=1.0,
                key="advisor_storage_temp",
                help="Ambient warehousing temperature for Arrhenius permeation scaling."
            )
        with env_col2:
            storage_rh = st.slider(
                "Ambient Humidity (% RH)",
                min_value=10,
                max_value=95,
                value=int(st.session_state.get("advisor_storage_rh", default_rh)),
                step=5,
                key="advisor_storage_rh",
                help="Ambient relative humidity governing driving vapor pressure."
            )

        spec_col1, spec_col2 = st.columns(2)
        with spec_col1:
            target_days = st.number_input(
                "Target Shelf Life (Days)",
                min_value=7,
                max_value=730,
                value=int(st.session_state.get("advisor_target_days", default_days)),
                step=15,
                key="advisor_target_days",
                help="Required commercial distribution shelf life."
            )
        with spec_col2:
            pack_area_cm2 = st.number_input(
                "Package Surface Area (cm²)",
                min_value=50.0,
                max_value=2500.0,
                value=400.0,
                step=50.0,
                key="advisor_pack_area",
                help="Total exterior pouch or tray surface area."
            )

    with step_col2:
        st.markdown("### Selected Product Profile")
        if active_preset:
            spoilage_mode = active_preset.get("typical_spoilage_mode", "Moisture / oxygen degradation")
            crit_aw_val = active_preset.get("critical_aw")
            crit_aw_str = f"{crit_aw_val:.2f}" if crit_aw_val is not None else "N/A"
            food_img = get_food_image_path(active_preset.get("category", selected_cat), active_preset.get("name", ""), active_preset.get("image_path", ""))
            source_note = active_preset.get("source", "Standard Food Science Literature")

            f_img_col, f_info_col = st.columns([1, 1.4], gap="medium")
            with f_img_col:
                if food_img and os.path.exists(food_img):
                    st.image(food_img, caption=f"Product: {active_preset.get('name')}", use_container_width=True)
                else:
                    st.image("assets/products/placeholder_food.jpg", caption=f"Product: {active_preset.get('name')}", use_container_width=True)
            with f_info_col:
                render_html(f"""
                <div class="summary-panel-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.78rem; font-weight: 800; color: var(--forest-deep); text-transform: uppercase; letter-spacing: 0.05em;">
                            🌿 Verified Profile
                        </span>
                        <span class="meta-badge" style="background-color: var(--primary-light); color: var(--primary-deep); margin: 0; font-weight: 700; border: 1px solid rgba(233, 120, 61, 0.3);">
                            {active_preset.get('category', 'Custom')}
                        </span>
                    </div>
                    <div style="font-size: 1.10rem; font-weight: 800; color: var(--text-main); margin-bottom: 6px; font-family: 'Poppins', sans-serif;">
                        {active_preset.get('name', 'Food Product')}
                    </div>
                    <div style="font-size: 0.82rem; color: var(--text-main); line-height: 1.4; margin-bottom: 10px; background: rgba(23, 61, 50, 0.05); padding: 6px 10px; border-radius: 6px; border-left: 3px solid var(--forest-deep);">
                        <b>Spoilage Mode:</b> {spoilage_mode}
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; font-size: 0.78rem;">
                        <div class="summary-metric-tile">
                            <span class="summary-metric-label">Moisture Sens.</span>
                            <span class="summary-metric-value">{active_preset.get('moisture_sensitivity', 'Moderate')}</span>
                        </div>
                        <div class="summary-metric-tile">
                            <span class="summary-metric-label">Oxygen Sens.</span>
                            <span class="summary-metric-value">{active_preset.get('oxygen_sensitivity', 'Moderate')}</span>
                        </div>
                        <div class="summary-metric-tile">
                            <span class="summary-metric-label">Fat Level</span>
                            <span class="summary-metric-value">{active_preset.get('fat_content_level', 'Low')}</span>
                        </div>
                        <div class="summary-metric-tile">
                            <span class="summary-metric-label">Critical aw</span>
                            <span class="summary-metric-value" style="color: var(--primary-deep); font-weight: 800;">{crit_aw_str}</span>
                        </div>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.72rem; color: var(--text-light); line-height: 1.35;">
                        <b>Source:</b> {source_note}
                    </div>
                </div>
                """)
        else:
            render_html("""
            <div class="summary-empty-state">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">🍽️</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: var(--forest-deep); margin-bottom: 6px;">No Food Product Selected</div>
                <p style="color: var(--text-muted); font-size: 0.86rem; line-height: 1.45; margin: 0 auto; max-width: 360px;">
                    Click <b>Select</b> on any product card on the left to inspect its physical sensitivities and unlock packaging analysis.
                </p>
            </div>
            """)

        st.markdown("### Set Decision Preferences")
        render_html(
            "<p style='color: var(--text-muted); font-size: 0.88rem; margin-top: -8px;'>"
            "Configure Multi-Criteria Decision Analysis (MCDA) weighting strategy across performance dimensions."
            "</p>"
        )

        strategy_options = [
            "⚖️ Balanced Performance (40% Barrier, 25% Cost, 25% Eco, 10% Mech)",
            "🛡️ High Barrier Priority (60% Barrier, 15% Cost, 15% Eco, 10% Mech)",
            "💰 Cost Economy Priority (25% Barrier, 50% Cost, 15% Eco, 10% Mech)",
            "🌱 Circularity & Eco Priority (25% Barrier, 15% Cost, 50% Eco, 10% Mech)",
            "⚙️ Custom Weighting Distribution"
        ]

        selected_strategy = st.radio(
            "Priority Strategy",
            options=strategy_options,
            index=0,
            key="advisor_strategy_radio"
        )

        if selected_strategy.startswith("⚖️"):
            w_barrier, w_cost, w_sust, w_mech = 0.40, 0.25, 0.25, 0.10
        elif selected_strategy.startswith("🛡️"):
            w_barrier, w_cost, w_sust, w_mech = 0.60, 0.15, 0.15, 0.10
        elif selected_strategy.startswith("💰"):
            w_barrier, w_cost, w_sust, w_mech = 0.25, 0.50, 0.15, 0.10
        elif selected_strategy.startswith("🌱"):
            w_barrier, w_cost, w_sust, w_mech = 0.25, 0.15, 0.50, 0.10
        else:
            render_html("<div style='margin-top: 8px;'></div>")
            w_c1, w_c2 = st.columns(2)
            with w_c1:
                wb_slider = st.slider("Barrier Weight (%)", 10, 80, 40, 5, key="w_barrier_slider")
                wc_slider = st.slider("Cost Weight (%)", 5, 80, 25, 5, key="w_cost_slider")
            with w_c2:
                ws_slider = st.slider("Circularity Weight (%)", 5, 80, 25, 5, key="w_sust_slider")
                wm_slider = st.slider("Mechanical Weight (%)", 0, 50, 10, 5, key="w_mech_slider")

            w_total = wb_slider + wc_slider + ws_slider + wm_slider
            w_barrier = wb_slider / w_total
            w_cost = wc_slider / w_total
            w_sust = ws_slider / w_total
            w_mech = wm_slider / w_total
            st.caption(f"Normalized Weights: Barrier {w_barrier*100:.0f}%, Cost {w_cost*100:.0f}%, Circularity {w_sust*100:.0f}%, Mech {w_mech*100:.0f}%")

        # Mandatory Regulatory Screening & Safety Gates moved to st.expander
        with st.expander("🛡️ Mandatory Regulatory Screening & Safety Gates", expanded=False):
            render_html(
                """
                <div style="font-size: 0.86rem; color: var(--text-main); line-height: 1.55; padding: 4px 0;">
                    <b>Mandatory Regulatory Screening:</b> All materials undergo Stage-1 physics knockout filters 
                    (moisture/oxygen threshold, thermal tolerance, puncture hazard) and verified 
                    <b>US FDA 21 CFR §177</b> / <b>EU Regulation 10/2011</b> food-contact compliance screening 
                    before multi-criteria scoring.
                </div>
                """
            )

        render_html("<div style='margin-top: 18px;'></div>")

        # ---------------------------------------------------------------------
        # STEP 3 — Prominent Action Button: "Analyze Packaging"
        # ---------------------------------------------------------------------
        is_analyzable = (active_preset is not None)
        analyze_clicked = st.button(
            "🔍 Analyze Packaging",
            type="primary",
            use_container_width=True,
            key="btn_run_analysis",
            disabled=(not is_analyzable),
            help="Run physical barrier filters and MCDA analysis" if is_analyzable else "Select a food product card on the left to enable analysis"
        )
        if not is_analyzable:
            render_html("<p style='text-align: center; color: var(--text-muted); font-size: 0.80rem; margin-top: 6px;'>👈 Select a food product above to unlock packaging analysis.</p>")

        if analyze_clicked:
            # 1. Validation
            if target_days <= 0:
                st.error("Target shelf life must be greater than 0 days.")
            elif pack_area_cm2 <= 0:
                st.error("Package surface area must be greater than 0 cm².")
            else:
                # 2. Build Commodity Profile Dict
                commodity_input = {
                    "name": active_preset.get("name", "Selected Food"),
                    "category": active_preset.get("category", "General"),
                    "moisture_sensitivity": active_preset.get("moisture_sensitivity", "Moderate"),
                    "oxygen_sensitivity": active_preset.get("oxygen_sensitivity", "Moderate"),
                    "fat_content_level": active_preset.get("fat_content_level", "Low"),
                    "light_sensitivity": active_preset.get("light_sensitivity", "Low"),
                    "storage_temp_c": float(storage_temp),
                    "storage_rh_percent": float(storage_rh),
                    "target_shelf_life_days": int(target_days),
                    "puncture_risk": active_preset.get("puncture_risk", "Low")
                }

                weights_dict = {
                    "barrier": float(w_barrier),
                    "cost": float(w_cost),
                    "sustainability": float(w_sust),
                    "mechanical": float(w_mech)
                }

                # 3. Execute Recommendation Engine with Multi-Step Engineering Feedback
                with st.status("🔬 Running Intelligent Packaging Evaluation...", expanded=True) as status_box:
                    st.write("⚙️ **Stage 1:** Evaluating Physical Barrier & Thermal Knockout Filters (WVTR, OTR, Max Temp, Puncture)...")
                    result = recommender.rank_recommendations(
                        materials=all_materials,
                        commodity_profile=commodity_input,
                        weights=weights_dict
                    )
                    st.write("📊 **Stage 2:** Computing Multi-Criteria Decision Analysis (MCDA) Scoring & Weight Synthesis...")
                    st.write("🏆 **Stage 3:** Ranking Qualified Benchmark Materials & Calculating Trade-Off Audits...")
                    
                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_profile"] = commodity_input
                    st.session_state["analysis_weights"] = weights_dict
                    st.session_state["just_completed_analysis"] = True

                    # Safely log recommendation run in SQLite audit log
                    try:
                        database.log_recommendation_run(
                            commodity_name=commodity_input["name"],
                            category=commodity_input["category"],
                            inputs_dict=commodity_input,
                            weights_dict=weights_dict,
                            recommendations=result.get("top_candidates", [])
                        )
                    except Exception:
                        pass  # Audit logging failure should not break UI

                    status_box.update(label=f"✅ Analysis Completed for {commodity_input['name']}!", state="complete", expanded=False)

                st.success(f"Analysis completed successfully for {commodity_input['name']}!")
                st.rerun()

    # -------------------------------------------------------------------------
    # C. STEP 4 — Recommendation Results & Supporting Insights
    # -------------------------------------------------------------------------
    analysis_res = st.session_state.get("analysis_result")
    analysis_prof = st.session_state.get("analysis_profile")
    analysis_w = st.session_state.get("analysis_weights")

    if analysis_res is not None and analysis_prof is not None:
        render_html("<hr style='border: 0; border-top: 1px solid var(--border-color); margin: 32px 0 24px 0;' />")
        st.markdown("## Recommendation Results")

        top_candidates = analysis_res.get("top_candidates", [])
        knocked_out = analysis_res.get("knocked_out", [])

        if not top_candidates:
            st.warning(
                f"⚠️ **No eligible materials satisfied all physical and safety constraints for {analysis_prof['name']}.**\n\n"
                f"All {analysis_res.get('total_evaluated', len(all_materials))} benchmark materials were disqualified "
                f"during Stage-1 physics knockout filters. This commonly happens if the target shelf-life is exceptionally long "
                f"at high temperature/humidity, or if storage temperature exceeds standard polymer glass transitions.\n\n"
                f"**Suggested adjustment:** Try reducing target shelf life or lowering ambient storage temperature."
            )
            if knocked_out:
                with st.expander("Review Disqualified Materials and Knockout Reasons"):
                    for ko in knocked_out:
                        st.markdown(f"**{ko.get('name')}**: {', '.join(ko.get('knockout_reasons', []))}")
        else:
            # Confetti Celebration on analysis completion
            if st.session_state.get("just_completed_analysis"):
                st.components.v1.html(
                    """
                    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.3/dist/confetti.browser.min.js"></script>
                    <script>
                        setTimeout(() => {
                            if (window.confetti) {
                                confetti({
                                    particleCount: 75,
                                    spread: 80,
                                    origin: { y: 0.6 },
                                    colors: ['#E9783D', '#D8672C', '#173D32', '#E9B44C', '#A7BCA7']
                                });
                            }
                        }, 250);
                    </script>
                    """,
                    height=0,
                    width=0
                )
                st.session_state["just_completed_analysis"] = False

            top_cand = top_candidates[0]
            expl = top_cand.get("explanation", {})
            top_score_val = top_cand.get('total_mcda_score', 0.0)
            dashoffset = 226.2 * (1.0 - min(max(top_score_val / 100.0, 0.0), 1.0))
            top_mat_img = get_material_image_path(top_cand.get("name", ""))
            top_img_b64 = get_image_base64(top_mat_img)

            # -----------------------------------------------------------------
            # 0. TOP VERDICT CARD (Clear recommendation, physical reasoning, shelf life, export actions)
            # -----------------------------------------------------------------
            wvtr_top = float(top_cand.get("wvtr_g_m2_day", 1.0))
            otr_top = float(top_cand.get("otr_cc_m2_day_atm", 1.0))
            try:
                s_calc = shelf_life.estimate_combined_shelf_life(
                    wvtr_standard=wvtr_top,
                    otr_standard=otr_top,
                    package_surface_area_m2=analysis_prof.get("package_surface_area_m2", 0.04),
                    product_dry_weight_g=150.0,
                    initial_moisture_percent=2.0,
                    critical_moisture_percent=5.0,
                    headspace_volume_cc=100.0,
                    initial_headspace_o2_percent=0.5,
                    critical_headspace_o2_percent=2.0,
                    ambient_temp_c=float(analysis_prof.get("storage_temp_c", 23.0)),
                    ambient_rh_percent=float(analysis_prof.get("storage_rh_percent", 60.0))
                )
                est_shelf_days = int(s_calc.get("limiting_days", analysis_prof.get("target_shelf_life_days", 180)))
            except Exception:
                est_shelf_days = int(analysis_prof.get("target_shelf_life_days", 180))

            m_sens = analysis_prof.get('moisture_sensitivity', 'Moderate')
            o_sens = analysis_prof.get('oxygen_sensitivity', 'Moderate')
            t_deg = analysis_prof.get('storage_temp_c', 23.0)
            rh_p = analysis_prof.get('storage_rh_percent', 60.0)

            render_html(f"""
            <div class="verdict-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                    <div style="flex: 1; min-width: 280px;">
                        <div class="verdict-badge-lead">🎯 Top Engineering Recommendation</div>
                        <div class="verdict-title" style="margin-top: 6px;">
                            Recommended: <span style="color: var(--primary-deep);">{top_cand['name']}</span> for <span style="color: var(--forest-deep);">{analysis_prof['name']}</span>
                        </div>
                    </div>
                    <div style="display: flex; gap: 14px; align-items: center; background: #FFFFFF; border: 1.5px solid var(--border-color); border-radius: var(--radius-md); padding: 10px 18px; box-shadow: var(--shadow-sm);">
                        <div style="text-align: center;">
                            <div style="font-size: 0.68rem; font-weight: 700; color: var(--text-light); text-transform: uppercase;">MCDA Score</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: var(--primary-deep); font-family: 'Poppins', sans-serif;">{top_score_val:.1f} <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: normal;">/ 100</span></div>
                        </div>
                        <div style="border-left: 1px solid var(--border-color); height: 32px;"></div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.68rem; font-weight: 700; color: var(--text-light); text-transform: uppercase;">Est. Shelf Life</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: var(--forest-deep); font-family: 'Poppins', sans-serif;">{est_shelf_days} <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: normal;">days</span></div>
                        </div>
                    </div>
                </div>
                <div class="verdict-reasoning">
                    <b>Physical Barrier Match:</b> {analysis_prof['name']} exhibits <b>{m_sens}</b> moisture sensitivity and <b>{o_sens}</b> oxygen sensitivity. 
                    <b>{top_cand['name']}</b> provides a high-barrier profile with WVTR <b>{top_cand.get('wvtr_g_m2_day')} g/m²/day</b> and OTR <b>{top_cand.get('otr_cc_m2_day_atm')} cc/m²/day/atm</b>, 
                    delivering an estimated shelf life of <b>{est_shelf_days} days</b> under {t_deg}°C / {rh_p}% RH while complying with US FDA 21 CFR §177 and EU Regulation 10/2011 criteria.
                </div>
            </div>
            """)

            # Quick Verdict Action Buttons (Export PDF, Export CSV, Add Winner to Basket)
            v_act1, v_act2, v_act3 = st.columns([1.2, 1.2, 1.2], gap="small")
            clean_name = "".join(c for c in analysis_prof["name"] if c.isalnum() or c in (" ", "_")).replace(" ", "_")
            with v_act1:
                try:
                    pdf_bytes = pdf_generator.generate_recommendation_pdf(analysis_prof, analysis_res, analysis_w)
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"FoodPackAI_Report_{clean_name}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                        key="btn_verdict_export_pdf"
                    )
                except Exception as e:
                    st.error(f"PDF export error: {e}")

            with v_act2:
                res_csv_rows = []
                for c_i in top_candidates:
                    res_csv_rows.append({
                        "Rank": c_i.get("rank", ""),
                        "Material": c_i.get("name", ""),
                        "MCDA Score": round(float(c_i.get("total_mcda_score", 0)), 1),
                        "WVTR (g/m2/d)": c_i.get("wvtr_g_m2_day"),
                        "OTR (cc/m2/d)": c_i.get("otr_cc_m2_day_atm"),
                        "Cost ($/kg)": c_i.get("cost_index_usd_per_kg"),
                        "Carbon Footprint": c_i.get("carbon_footprint_kg_co2_per_kg"),
                        "Tensile Strength": c_i.get("tensile_strength_mpa"),
                        "Structure": c_i.get("structure_type"),
                        "Recyclability": c_i.get("recyclability_status"),
                        "Compliance": c_i.get("food_contact_compliance")
                    })
                csv_bytes_res = pd.DataFrame(res_csv_rows).to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Results CSV",
                    data=csv_bytes_res,
                    file_name=f"FoodPackAI_Recommendations_{clean_name}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_verdict_export_csv"
                )

            with v_act3:
                is_top_in_bsk = top_cand["name"] in st.session_state.get("compare_basket", [])
                bsk_btn_lbl = "✓ Winner In Basket" if is_top_in_bsk else "⚖️ Add Winner to Basket"
                bsk_btn_typ = "primary" if is_top_in_bsk else "secondary"
                if st.button(bsk_btn_lbl, key="btn_add_winner_to_basket", type=bsk_btn_typ, use_container_width=True):
                    if is_top_in_bsk:
                        st.session_state["compare_basket"].remove(top_cand["name"])
                    else:
                        if len(st.session_state["compare_basket"]) < 4:
                            st.session_state["compare_basket"].append(top_cand["name"])
                        else:
                            st.toast("Compare basket holds up to 4 materials.", icon="⚠️")
                    st.rerun()

            render_html("<div style='margin-bottom: 20px;'></div>")

            # -----------------------------------------------------------------
            # 1. 3D Recommendation Podium (🥇 🥈 🥉) with 4 Mini Metric Bars
            # -----------------------------------------------------------------
            st.markdown("### Top Recommendation Podium")

            # Rank 1 bars & badges
            r1_bars = compute_mini_bars(top_cand)
            r1_bars_html = "".join(f"""
            <div class="mini-metric-row">
                <div class="mini-metric-label-val">
                    <span class="mini-metric-label">{b['name']}</span>
                    <span class="mini-metric-val">{b['val_str']}</span>
                </div>
                <div class="mini-bar-track">
                    <div class="mini-bar-fill" style="width: {b['pct']:.1f}%; background-color: {b['color']};"></div>
                </div>
            </div>
            """ for b in r1_bars)
            r1_rec = get_recyclability_pill(top_cand.get("recyclability_status", ""))
            rank1_img_tag = f'<img src="{top_img_b64}" style="width: 100%; height: 130px; object-fit: cover; border-radius: 8px; margin-bottom: 10px; border: 1px solid rgba(0,0,0,0.06);" />' if top_img_b64 else ''

            # Rank 2 (Silver)
            rank2_cand = top_candidates[1] if len(top_candidates) >= 2 else None
            rank2_html = ""
            if rank2_cand:
                r2_score = rank2_cand.get('total_mcda_score', 0.0)
                r2_img = get_image_base64(get_material_image_path(rank2_cand.get("name", "")))
                r2_img_tag = f'<img src="{r2_img}" style="width: 100%; height: 110px; object-fit: cover; border-radius: 6px; margin-bottom: 8px; border: 1px solid rgba(0,0,0,0.06);" />' if r2_img else ''
                r2_bars = compute_mini_bars(rank2_cand)
                r2_bars_html = "".join(f"""
                <div class="mini-metric-row">
                    <div class="mini-metric-label-val">
                        <span class="mini-metric-label">{b['name']}</span>
                        <span class="mini-metric-val">{b['val_str']}</span>
                    </div>
                    <div class="mini-bar-track">
                        <div class="mini-bar-fill" style="width: {b['pct']:.1f}%; background-color: {b['color']};"></div>
                    </div>
                </div>
                """ for b in r2_bars)
                r2_rec = get_recyclability_pill(rank2_cand.get("recyclability_status", ""))

                rank2_html = f"""
                <div class="podium-card" style="border: 1.5px solid #94A3B8; background: #FFFFFF;">
                    <div class="podium-badge podium-badge-silver">🥈 Rank #2 Silver</div>
                    {r2_img_tag}
                    <div style="font-size: 0.95rem; font-weight: 700; color: var(--text-main); line-height: 1.3; margin-bottom: 6px;">
                        {rank2_cand.get('name')}
                    </div>
                    <div style="display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 6px;">
                        <span class="meta-badge" style="font-size: 0.70rem;">{rank2_cand.get('structure_type', 'Flexible')}</span>
                        <span class="meta-badge" style="font-size: 0.70rem;">{rank2_cand.get('nominal_thickness_um', '-')} µm</span>
                    </div>
                    <div style="background: var(--card-sub); border: 1px solid var(--border-color); border-radius: 8px; padding: 6px 8px; text-align: center; margin-bottom: 8px;">
                        <span style="font-size: 0.68rem; color: var(--text-light); text-transform: uppercase; font-weight: 700;">Score: </span>
                        <b style="font-size: 1.15rem; color: #455A64; font-family: 'Poppins', sans-serif;">{r2_score:.1f}</b> <span style="font-size: 0.72rem; color: var(--text-muted);">/ 100</span>
                    </div>
                    <div class="mini-metrics-container">{r2_bars_html}</div>
                    <div style="margin-top: 6px;">{r2_rec}</div>
                </div>
                """

            # Rank 3 (Bronze)
            rank3_cand = top_candidates[2] if len(top_candidates) >= 3 else None
            rank3_html = ""
            if rank3_cand:
                r3_score = rank3_cand.get('total_mcda_score', 0.0)
                r3_img = get_image_base64(get_material_image_path(rank3_cand.get("name", "")))
                r3_img_tag = f'<img src="{r3_img}" style="width: 100%; height: 110px; object-fit: cover; border-radius: 6px; margin-bottom: 8px; border: 1px solid rgba(0,0,0,0.06);" />' if r3_img else ''
                r3_bars = compute_mini_bars(rank3_cand)
                r3_bars_html = "".join(f"""
                <div class="mini-metric-row">
                    <div class="mini-metric-label-val">
                        <span class="mini-metric-label">{b['name']}</span>
                        <span class="mini-metric-val">{b['val_str']}</span>
                    </div>
                    <div class="mini-bar-track">
                        <div class="mini-bar-fill" style="width: {b['pct']:.1f}%; background-color: {b['color']};"></div>
                    </div>
                </div>
                """ for b in r3_bars)
                r3_rec = get_recyclability_pill(rank3_cand.get("recyclability_status", ""))

                rank3_html = f"""
                <div class="podium-card" style="border: 1.5px solid #CBD5E1; background: #FFFFFF;">
                    <div class="podium-badge podium-badge-bronze">🥉 Rank #3 Bronze</div>
                    {r3_img_tag}
                    <div style="font-size: 0.95rem; font-weight: 700; color: var(--text-main); line-height: 1.3; margin-bottom: 6px;">
                        {rank3_cand.get('name')}
                    </div>
                    <div style="display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 6px;">
                        <span class="meta-badge" style="font-size: 0.70rem;">{rank3_cand.get('structure_type', 'Flexible')}</span>
                        <span class="meta-badge" style="font-size: 0.70rem;">{rank3_cand.get('nominal_thickness_um', '-')} µm</span>
                    </div>
                    <div style="background: var(--card-sub); border: 1px solid var(--border-color); border-radius: 8px; padding: 6px 8px; text-align: center; margin-bottom: 8px;">
                        <span style="font-size: 0.68rem; color: var(--text-light); text-transform: uppercase; font-weight: 700;">Score: </span>
                        <b style="font-size: 1.15rem; color: #5D4037; font-family: 'Poppins', sans-serif;">{r3_score:.1f}</b> <span style="font-size: 0.72rem; color: var(--text-muted);">/ 100</span>
                    </div>
                    <div class="mini-metrics-container">{r3_bars_html}</div>
                    <div style="margin-top: 6px;">{r3_rec}</div>
                </div>
                """

            # Rank 1 (Gold) Center Card
            rank1_html = f"""
            <div class="podium-card podium-rank-1">
                <div class="podium-badge podium-badge-gold">👑 🥇 Rank #1 Optimal Match</div>
                <div style="display: flex; gap: 14px; align-items: center; margin-bottom: 10px; margin-top: 4px;">
                    <div class="gauge-wrapper" style="flex-shrink: 0; margin: 0;">
                        <svg class="gauge-svg" width="90" height="90" viewBox="0 0 90 90">
                            <defs>
                                <linearGradient id="scoreGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stop-color="#E9783D" />
                                    <stop offset="100%" stop-color="#D8672C" />
                                </linearGradient>
                            </defs>
                            <circle cx="45" cy="45" r="36" class="gauge-bg" style="stroke: #FDEEE6; stroke-width: 7; fill: none;" />
                            <circle cx="45" cy="45" r="36" class="gauge-fill" style="stroke: url(#scoreGrad); stroke-width: 7; fill: none; stroke-linecap: round; stroke-dasharray: 226.2; stroke-dashoffset: {dashoffset:.1f};" />
                        </svg>
                        <div class="gauge-text" style="position: absolute; font-family: 'Poppins', sans-serif; font-size: 1.25rem; font-weight: 800; color: var(--primary-deep);">{top_score_val:.0f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: var(--primary-deep); text-transform: uppercase; letter-spacing: 0.05em;">
                            Top Decision Score
                        </div>
                        <div style="font-size: 0.82rem; color: var(--text-muted); margin-top: 2px;">
                            Stage-2 Multi-Criteria Synthesis
                        </div>
                        <div style="display: flex; gap: 4px; flex-wrap: wrap; margin-top: 4px;">
                            <span class="meta-badge" style="background: var(--primary-light); color: var(--primary-deep); font-weight: 700; border-color: rgba(233,120,61,0.3); font-size: 0.70rem;">
                                {top_cand.get('structure_type', 'Flexible')}
                            </span>
                            <span class="meta-badge" style="font-size: 0.70rem;">{top_cand.get('nominal_thickness_um', '-')} µm</span>
                        </div>
                    </div>
                </div>
                {rank1_img_tag}
                <h3 style="font-size: 1.20rem; font-weight: 800; color: var(--text-main); line-height: 1.25; margin: 4px 0 6px 0;">
                    {top_cand.get('name')}
                </h3>
                <p style="color: var(--text-muted); font-size: 0.80rem; margin-bottom: 8px; line-height: 1.45; border-left: 3px solid var(--primary-orange); padding-left: 10px;">
                    {expl.get('rationale', 'Optimal packaging solution satisfying barrier, cost, and circularity criteria.')}
                </p>
                <div class="mini-metrics-container">{r1_bars_html}</div>
                <div style="margin-top: 6px;">{r1_rec}</div>
            </div>
            """

            # Render Podium Layout
            if rank2_html and rank3_html:
                podium_grid_html = f"""
                <div class="podium-container">
                    {rank2_html}
                    {rank1_html}
                    {rank3_html}
                </div>
                """
            elif rank2_html:
                podium_grid_html = f"""
                <div style="display: grid; grid-template-columns: 1.3fr 1fr; gap: 16px; align-items: end; margin: 20px 0 28px 0;">
                    {rank1_html}
                    {rank2_html}
                </div>
                """
            else:
                podium_grid_html = f"""
                <div style="max-width: 600px; margin: 20px auto 28px auto;">
                    {rank1_html}
                </div>
                """
            render_html(podium_grid_html)

            # Podium Card Compare Toggles Row
            if rank2_html and rank3_html:
                pod_btn_c1, pod_btn_c2, pod_btn_c3 = st.columns([1, 1.25, 1])
                with pod_btn_c1:
                    if rank2_cand:
                        in_b2 = rank2_cand["name"] in st.session_state.get("compare_basket", [])
                        l2 = "✓ In Basket" if in_b2 else "➕ Add to Compare"
                        t2 = "primary" if in_b2 else "secondary"
                        can_a2 = in_b2 or len(st.session_state.get("compare_basket", [])) < 4
                        if st.button(l2, key=f"btn_pod_cmp_r2_{rank2_cand['name'][:6]}", type=t2, disabled=(not can_a2 and not in_b2), use_container_width=True):
                            if in_b2:
                                st.session_state["compare_basket"].remove(rank2_cand["name"])
                            else:
                                st.session_state["compare_basket"].append(rank2_cand["name"])
                            st.rerun()

                with pod_btn_c2:
                    in_b1 = top_cand["name"] in st.session_state.get("compare_basket", [])
                    l1 = "✓ In Basket" if in_b1 else "➕ Add to Compare"
                    t1 = "primary" if in_b1 else "secondary"
                    can_a1 = in_b1 or len(st.session_state.get("compare_basket", [])) < 4
                    if st.button(l1, key=f"btn_pod_cmp_r1_{top_cand['name'][:6]}", type=t1, disabled=(not can_a1 and not in_b1), use_container_width=True):
                        if in_b1:
                            st.session_state["compare_basket"].remove(top_cand["name"])
                        else:
                            st.session_state["compare_basket"].append(top_cand["name"])
                        st.rerun()

                with pod_btn_c3:
                    if rank3_cand:
                        in_b3 = rank3_cand["name"] in st.session_state.get("compare_basket", [])
                        l3 = "✓ In Basket" if in_b3 else "➕ Add to Compare"
                        t3 = "primary" if in_b3 else "secondary"
                        can_a3 = in_b3 or len(st.session_state.get("compare_basket", [])) < 4
                        if st.button(l3, key=f"btn_pod_cmp_r3_{rank3_cand['name'][:6]}", type=t3, disabled=(not can_a3 and not in_b3), use_container_width=True):
                            if in_b3:
                                st.session_state["compare_basket"].remove(rank3_cand["name"])
                            else:
                                st.session_state["compare_basket"].append(rank3_cand["name"])
                            st.rerun()
            else:
                in_b1 = top_cand["name"] in st.session_state.get("compare_basket", [])
                l1 = "✓ In Basket" if in_b1 else "➕ Add to Compare"
                t1 = "primary" if in_b1 else "secondary"
                if st.button(l1, key=f"btn_pod_cmp_single_{top_cand['name'][:6]}", type=t1, use_container_width=True):
                    if in_b1:
                        st.session_state["compare_basket"].remove(top_cand["name"])
                    else:
                        st.session_state["compare_basket"].append(top_cand["name"])
                    st.rerun()

            render_html("<div style='margin-bottom: 20px;'></div>")

            # -----------------------------------------------------------------
            # 2. 4 3D Metric Pillar Cards for Rank #1
            # -----------------------------------------------------------------
            b_score = top_cand.get('barrier_score', 0)
            c_score = top_cand.get('cost_score', 0)
            s_score = top_cand.get('sustainability_score', 0)
            m_score = top_cand.get('mechanical_score', 0)

            render_html(
                f"""
                <div class="metric-pillar-grid" style="margin-bottom: 20px;">
                    <div class="metric-pillar-card">
                        <span class="metric-pillar-label">🛡️ Barrier Protection</span>
                        <div class="metric-pillar-val">
                            {b_score:.1f} <span class="metric-pillar-sub">/ 100</span>
                        </div>
                    </div>
                    <div class="metric-pillar-card">
                        <span class="metric-pillar-label">💰 Cost Affordability</span>
                        <div class="metric-pillar-val">
                            {c_score:.1f} <span class="metric-pillar-sub">/ 100</span>
                        </div>
                    </div>
                    <div class="metric-pillar-card">
                        <span class="metric-pillar-label">🌱 Circularity &amp; Eco</span>
                        <div class="metric-pillar-val">
                            {s_score:.1f} <span class="metric-pillar-sub">/ 100</span>
                        </div>
                    </div>
                    <div class="metric-pillar-card">
                        <span class="metric-pillar-label">⚙️ Mechanical Resilience</span>
                        <div class="metric-pillar-val">
                            {m_score:.1f} <span class="metric-pillar-sub">/ 100</span>
                        </div>
                    </div>
                </div>
                """
            )

            # -----------------------------------------------------------------
            # 3. "Why This Material Won?" Explanation Card
            # -----------------------------------------------------------------
            req_items = "".join(f"<div style='font-size: 0.84rem; color: var(--text-main); margin-bottom: 6px;'>&bull; <span style='color: var(--forest-deep); font-weight: 700;'>&#10003;</span> {r}</div>" for r in expl.get("requirements_met", []))
            lim_items = "".join(f"<div style='font-size: 0.84rem; color: var(--text-main); margin-bottom: 6px;'>&bull; <span style='color: var(--primary-deep); font-weight: 700;'>&#9888;</span> {l}</div>" for l in expl.get("limitations", []))
            mis_items = "".join(f"<div style='font-size: 0.80rem; color: var(--text-light); font-style: italic; margin-bottom: 4px;'>&bull; Literature gap: {m}</div>" for m in expl.get("missing_data", []))

            render_html(
                f"""
                <div class="why-won-box">
                    <div class="why-won-title">
                        💡 Why This Material Won? — Decision Rationale &amp; Engineering Evidence
                    </div>
                    <p style="font-size: 0.90rem; color: var(--text-muted); margin-bottom: 12px; line-height: 1.5;">
                        {expl.get('rationale', 'Optimal packaging solution satisfying barrier, cost, and circularity criteria.')}
                    </p>
                    <div class="why-won-grid">
                        <div style="background: #FFFFFF; padding: 14px 16px; border-radius: var(--radius-sm); border: 1px solid var(--border-color);">
                            <div style="font-size: 0.82rem; font-weight: 700; color: var(--forest-deep); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.04em;">
                                ✅ Key Requirements Satisfied
                            </div>
                            {req_items if req_items else "<div style='font-size: 0.84rem; color: var(--text-muted);'>All physical barrier criteria satisfied.</div>"}
                        </div>
                        <div style="background: #FFFFFF; padding: 14px 16px; border-radius: var(--radius-sm); border: 1px solid var(--border-color);">
                            <div style="font-size: 0.82rem; font-weight: 700; color: var(--primary-deep); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.04em;">
                                ⚖️ Operational Considerations &amp; Trade-offs
                            </div>
                            {lim_items if lim_items else "<div style='font-size: 0.84rem; color: var(--text-muted);'>No significant material concessions recorded.</div>"}
                            {mis_items}
                        </div>
                    </div>
                </div>
                """
            )

            # Technical Specifications Details (Consistent grid, no bright green backtick text, clear labels vs values)
            with st.expander("📋 View Detailed Technical Specifications & Literature Provenance", expanded=True):
                cost_val = top_cand.get('cost_index_usd_per_kg')
                cost_str = f"${cost_val:.2f} / kg" if cost_val is not None else "Unreported in literature"
                co2_val = top_cand.get('carbon_footprint_kg_co2_per_kg')
                co2_str = f"{co2_val:.2f} kg CO2e / kg" if co2_val is not None else "Unreported in literature"
                ts_val = top_cand.get('tensile_strength_mpa')
                ts_str = f"{ts_val:.1f} MPa" if ts_val is not None else "Unreported in literature"
                food_contact = top_cand.get('food_contact_compliance', 'Standard Food Contact Compliance')
                test_cond = top_cand.get('test_conditions', 'ASTM F1249 (38°C/90% RH); ASTM D3985 (23°C/0% RH)')
                data_src = top_cand.get('data_source', 'Published Packaging Literature')

                render_html(
                    f"""
                    <div class="spec-grid-container">
                        <div class="spec-card-item">
                            <div class="spec-label-text">Water Vapor Transmission (WVTR)</div>
                            <div class="spec-val-text">{top_cand.get('wvtr_g_m2_day', '-')} <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">g/m²/day</span></div>
                        </div>
                        <div class="spec-card-item">
                            <div class="spec-label-text">Oxygen Transmission (OTR)</div>
                            <div class="spec-val-text">{top_cand.get('otr_cc_m2_day_atm', '-')} <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">cc/m²/day/atm</span></div>
                        </div>
                        <div class="spec-card-item">
                            <div class="spec-label-text">Polymer Code &amp; Structure</div>
                            <div class="spec-val-text">{top_cand.get('polymer_code', 'N/A')}</div>
                        </div>
                        <div class="spec-card-item">
                            <div class="spec-label-text">Cost Index</div>
                            <div class="spec-val-text">{cost_str}</div>
                        </div>
                        <div class="spec-card-item">
                            <div class="spec-label-text">Cradle-to-Gate Carbon Footprint</div>
                            <div class="spec-val-text">{co2_str}</div>
                        </div>
                        <div class="spec-card-item">
                            <div class="spec-label-text">Tensile Strength</div>
                            <div class="spec-val-text">{ts_str}</div>
                        </div>
                        <div class="spec-card-item">
                            <div class="spec-label-text">Recycling Stream</div>
                            <div class="spec-val-text">{top_cand.get('recyclability_status', '-')}</div>
                        </div>
                        <div class="spec-card-item">
                            <div class="spec-label-text">Standard Test Protocol</div>
                            <div class="spec-val-text" style="font-size: 0.82rem;">{test_cond}</div>
                        </div>
                        <div class="spec-card-item">
                            <div class="spec-label-text">Literature Data Provenance</div>
                            <div class="spec-val-text" style="font-size: 0.82rem;">{data_src}</div>
                        </div>
                    </div>
                    <div style="background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 14px; margin-bottom: 14px;">
                        <div class="spec-label-text">Food Contact Regulatory Compliance</div>
                        <div class="spec-val-text" style="font-size: 0.86rem; color: var(--text-main);">{food_contact}</div>
                    </div>
                    """
                )

                # Requirements met vs Limitations
                render_html("<hr style='border: 0; border-top: 1px solid var(--border-color); margin: 12px 0;' />")
                rm_col1, rm_col2 = st.columns(2)
                with rm_col1:
                    render_html("<h5 style='font-size: 0.92rem; color: var(--forest-deep); margin-bottom: 8px;'>Key Requirements Satisfied</h5>")
                    for req in expl.get("requirements_met", []):
                        render_html(f"<div style='font-size: 0.86rem; color: var(--text-muted); margin-bottom: 4px;'>&bull; <span style='color: var(--forest-deep); font-weight: 700;'>&#10003;</span> {req}</div>")
                with rm_col2:
                    render_html("<h5 style='font-size: 0.92rem; color: var(--forest-deep); margin-bottom: 8px;'>Operational Considerations &amp; Disclosures</h5>")
                    for lim in expl.get("limitations", []):
                        render_html(f"<div style='font-size: 0.86rem; color: var(--text-muted); margin-bottom: 4px;'>&bull; <span style='color: var(--primary-deep); font-weight: 700;'>&#9888;</span> {lim}</div>")
                    for mis in expl.get("missing_data", []):
                        render_html(f"<div style='font-size: 0.84rem; color: #94A3B8; font-style: italic; margin-bottom: 4px;'>&bull; Missing literature datum: {mis}</div>")

            # Supporting Insights: Horizontal Bar Chart + "Why #1 Won" Tradeoff
            st.markdown("### Supporting Decision Insights")
            chart_col, tradeoff_col = st.columns([1, 1], gap="medium")

            with chart_col:
                st.markdown("##### Candidate Score Comparison")
                fig_ranking = analytics.create_ranking_horizontal_bar_chart(top_candidates, top_n=6)
                st.plotly_chart(fig_ranking, use_container_width=True)

            with tradeoff_col:
                st.markdown("##### Comparative Trade-Off Analysis")
                tradeoffs = analysis_res.get("leading_tradeoffs", {})
                if tradeoffs and tradeoffs.get("headline"):
                    render_html(
                        f"""
                        <div class="tradeoff-box">
                            <h4 style="font-size: 1.0rem; color: var(--forest-deep); margin-top: 0;">{tradeoffs.get('headline')}</h4>
                            <p style="font-size: 0.88rem; color: var(--text-main); margin-bottom: 8px;">{tradeoffs.get('verdict', '')}</p>
                            <ul style="font-size: 0.85rem; color: var(--text-muted); padding-left: 20px; margin-bottom: 0;">
                                {"".join(f"<li><b>Advantage:</b> {adv}</li>" for adv in tradeoffs.get('advantages', []))}
                                {"".join(f"<li><b>Trade-off:</b> {cnc}</li>" for cnc in tradeoffs.get('concessions', []))}
                            </ul>
                        </div>
                        """
                    )
                else:
                    st.info("Top candidate dominates all evaluation criteria for this food commodity.")

            # Visual Analytics: Multi-Axis Radar and 4 Independent Metric Subplots
            st.markdown("### Decision Analytics: Multi-Axis Radar & Property Performance")
            res_chart_tab1, res_chart_tab2 = st.tabs(["🕸️ Performance Radar (Top Candidates)", "📊 Property Subplots (WVTR, OTR, Cost, CO₂e)"])
            with res_chart_tab1:
                radar_candidates = top_candidates[:min(4, len(top_candidates))]
                fig_radar, radar_warnings = analytics.create_radar_comparison_chart(radar_candidates)
                st.plotly_chart(fig_radar, use_container_width=True)
                if radar_warnings:
                    for rw in radar_warnings:
                        st.caption(f"ℹ️ {rw}")
            with res_chart_tab2:
                fig_sub_res = analytics.create_comparator_subplots(top_candidates[:min(4, len(top_candidates))])
                st.plotly_chart(fig_sub_res, use_container_width=True)

            # Full-Width Comparison Matrix for Top Ranked Candidates
            st.markdown("### Top Candidates Specification Comparison Matrix")
            render_html(
                "<p style='color: var(--text-muted); font-size: 0.88rem; margin-top: -8px;'>"
                "Detailed physical, economic, and circularity comparison with sticky specification labels and best-in-class highlights (⭐)."
                "</p>"
            )
            matrix_res_html = render_full_width_comparison_matrix(top_candidates[:min(4, len(top_candidates))])
            render_html(matrix_res_html)

            # Qualified Alternatives List & Quick Add to Basket
            st.markdown("### Qualified Alternative Materials")
            alt_mats_pool = [c["name"] for c in top_candidates]
            add_to_bsk_picks = st.multiselect(
                "Quick Add Alternatives to Compare Basket (Max 4 Total):",
                options=alt_mats_pool,
                default=[c for c in st.session_state.get("compare_basket", []) if c in alt_mats_pool],
                key="res_alts_basket_multiselect",
                help="Select materials to immediately update your Compare Basket in the sidebar."
            )
            # Sync basket if user toggles picks here
            if set(add_to_bsk_picks) != set([c for c in st.session_state.get("compare_basket", []) if c in alt_mats_pool]):
                for p in add_to_bsk_picks:
                    if p not in st.session_state["compare_basket"] and len(st.session_state["compare_basket"]) < 4:
                        st.session_state["compare_basket"].append(p)
                for p in list(st.session_state["compare_basket"]):
                    if p in alt_mats_pool and p not in add_to_bsk_picks:
                        st.session_state["compare_basket"].remove(p)
                st.rerun()

            # Collapsible Disqualified Benchmark Materials Section
            if knocked_out:
                with st.expander(f"⚠️ Disqualified Benchmark Materials ({len(knocked_out)} Screened Out in Stage-1 Physics Filters)", expanded=False):
                    render_html(
                        "<p style='color: var(--text-muted); font-size: 0.85rem; margin-bottom: 12px;'>"
                        "The following candidate materials failed critical physical barrier, temperature stability, or safety constraints "
                        "and were eliminated prior to MCDA scoring to guarantee food protection integrity:"
                        "</p>"
                    )
                    ko_html = '<div class="ko-card-container">'
                    for ko in knocked_out:
                        reasons = ko.get("knockout_reasons", [])
                        reasons_badges = "".join(f'<span class="ko-badge">🚨 {r}</span>' for r in reasons)
                        ko_html += f"""
                        <div class="ko-card">
                            <div class="ko-card-title">{ko.get('name')}</div>
                            <div style="font-size: 0.76rem; color: var(--text-muted); margin-bottom: 6px;">
                                {ko.get('structure_type', '')} &bull; {ko.get('nominal_thickness_um', '-')} µm
                            </div>
                            <div>{reasons_badges}</div>
                        </div>
                        """
                    ko_html += '</div>'
                    render_html(ko_html)

        # ---------------------------------------------------------------------
        # D. STEP 5 — Export Technical PDF Report
        # ---------------------------------------------------------------------
        render_html("<hr style='border: 0; border-top: 1px solid var(--border-color); margin: 28px 0 20px 0;' />")
        st.markdown("## Technical Decision Report Export")
        render_html(
            "<p style='color: var(--text-muted); font-size: 0.9rem; margin-top: -8px;'>"
            "Download an executive, publication-quality 2-page PDF summarizing commodity specifications, ranking scores, tradeoff audits, and ASTM standard test conditions."
            "</p>"
        )

        pdf_col1, pdf_col2 = st.columns([1, 2])
        with pdf_col1:
            try:
                pdf_bytes = pdf_generator.generate_recommendation_pdf(
                    commodity_profile=analysis_prof,
                    recommendation_result=analysis_res,
                    user_weights=analysis_w
                )
                clean_name = "".join(c for c in analysis_prof["name"] if c.isalnum() or c in (" ", "_")).replace(" ", "_")
                st.download_button(
                    label="📄 Download 2-Page Technical PDF Report",
                    data=pdf_bytes,
                    file_name=f"FoodPackAI_Report_{clean_name}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                    key="btn_download_pdf"
                )
            except Exception as e:
                st.error(f"Failed to generate PDF report: {e}")

        with pdf_col2:
            render_html(
                "<div style='font-size: 0.82rem; color: var(--text-muted); padding-top: 4px;'>"
                "<b>Document Quality Notice:</b> Report includes transparent scoring breakdowns, "
                "Stage-1 knockout audit, literature citations, and scientific test condition disclosures "
                "(ASTM F1249 at 38°C/90% RH, ASTM D3985 at 23°C/0% RH)."
                "</div>"
            )


# =============================================================================
# TAB 2: MATERIAL COMPARATOR (Side-by-side comparison of 2-4 materials)
# =============================================================================
elif active_tab == "⚖️ Material Comparator":
    st.markdown("### Side-by-Side Material Comparator")
    render_html("<p style='color: var(--text-muted); font-size: 0.92rem; margin-top: -8px;'>Directly compare barrier properties, economics, sustainability metrics, and regulatory evidence across benchmark materials.</p>")

    all_mat_names = [m["name"] for m in all_materials]
    active_food_name = st.session_state.get("advisor_product_name", "Fried Potato Chips & Crisps")
    active_food_preset = preset_by_name.get(active_food_name, all_presets[0])

    # Structure Filter & Quick Recommend Button
    filter_col1, filter_col2 = st.columns([1.6, 1.2], gap="medium")
    with filter_col1:
        structure_filter = st.selectbox(
            "Filter Benchmark Materials by Structure:",
            options=["All Structures", "Monolayer Films", "Multi-layer Laminates", "Bio-based & Compostable", "Rigid Containers"],
            key="cmp_structure_filter"
        )
    with filter_col2:
        render_html("<div style='height: 28px;'></div>")
        if st.button("✨ Recommend Best 3 for Current Food", use_container_width=True, key="btn_quick_best3"):
            comm_inp = {
                "name": active_food_preset.get("name", "Selected Food"),
                "category": active_food_preset.get("category", "General"),
                "moisture_sensitivity": active_food_preset.get("moisture_sensitivity", "Moderate"),
                "oxygen_sensitivity": active_food_preset.get("oxygen_sensitivity", "Moderate"),
                "fat_content_level": active_food_preset.get("fat_content_level", "Low"),
                "light_sensitivity": active_food_preset.get("light_sensitivity", "Low"),
                "storage_temp_c": float(active_food_preset.get("recommended_temp_c", 23.0)),
                "storage_rh_percent": float(active_food_preset.get("recommended_rh_percent", 60.0)),
                "target_shelf_life_days": int(active_food_preset.get("target_shelf_life_days", 180)),
                "puncture_risk": active_food_preset.get("puncture_risk", "Low")
            }
            q_res = recommender.rank_recommendations(all_materials, comm_inp)
            best3 = [c["name"] for c in q_res.get("top_candidates", [])[:3]]
            if best3:
                st.session_state["comparator_selected_mats"] = best3
                st.rerun()

    # Filter material list by structure type if requested
    if structure_filter == "Monolayer Films":
        filtered_mat_names = [m["name"] for m in all_materials if "mono" in m.get("structure_type", "").lower() or ("flexible" in m.get("structure_type", "").lower() and "laminate" not in m.get("structure_type", "").lower() and "composite" not in m.get("structure_type", "").lower())]
    elif structure_filter == "Multi-layer Laminates":
        filtered_mat_names = [m["name"] for m in all_materials if "laminate" in m.get("structure_type", "").lower() or "composite" in m.get("structure_type", "").lower() or "co-extruded" in m.get("structure_type", "").lower()]
    elif structure_filter == "Bio-based & Compostable":
        filtered_mat_names = [m["name"] for m in all_materials if "bio" in m.get("structure_type", "").lower() or "compost" in m.get("notes", "").lower() or "pla" in m.get("name", "").lower() or "cellulose" in m.get("name", "").lower()]
    elif structure_filter == "Rigid Containers":
        filtered_mat_names = [m["name"] for m in all_materials if "rigid" in m.get("structure_type", "").lower() or "glass" in m.get("name", "").lower() or "can" in m.get("name", "").lower()]
    else:
        filtered_mat_names = all_mat_names

    if not filtered_mat_names:
        filtered_mat_names = all_mat_names

    # Pre-populate defaults: session state selections, or top 2 candidates, or standard films
    default_compare_picks = st.session_state.get("comparator_selected_mats", [])
    if not default_compare_picks:
        if current_result and current_result.get("top_candidates"):
            default_compare_picks = [c["name"] for c in current_result["top_candidates"][:2]]
        if len(default_compare_picks) < 2:
            default_compare_picks = [
                "BOPP / Met-BOPP / PE Tri-laminate (65 µm)",
                "PET / EVOH / PE Multi-layer Barrier Film (70 µm)",
                "PLA Bio-plastic Film (30 µm)"
            ]
    valid_defaults = [name for name in default_compare_picks if name in filtered_mat_names]
    if len(valid_defaults) < 2 and len(filtered_mat_names) >= 2:
        valid_defaults = filtered_mat_names[:min(3, len(filtered_mat_names))]
    elif not valid_defaults:
        valid_defaults = filtered_mat_names[:min(3, len(filtered_mat_names))]

    # Load from Compare Basket button if basket has items matching filter
    c_basket = st.session_state.get("compare_basket", [])
    if c_basket and any(m in filtered_mat_names for m in c_basket):
        if st.button(f"📥 Load {len([m for m in c_basket if m in filtered_mat_names])} Materials from Compare Basket", key="btn_load_basket_to_cmp"):
            valid_defaults = [m for m in c_basket if m in filtered_mat_names]
            st.session_state["comparator_selected_mats"] = valid_defaults
            st.rerun()

    selected_compare_names = st.multiselect(
        "Select Materials to Compare (Choose 2 to 4 materials):",
        options=filtered_mat_names,
        default=valid_defaults[:min(4, len(valid_defaults))],
        max_selections=4,
        key=f"cmp_filter_mats_{structure_filter.replace(' ', '_')}"
    )

    if len(selected_compare_names) < 2:
        st.info("ℹ️ Please select at least 2 materials to display the comparative engineering matrix and analytics.")
    else:
        selected_mats = [m for m in all_materials if m["name"] in selected_compare_names]

        # ---------------------------------------------------------------------
        # 1. Candidate Material Specimens Cards with Badges & Basket Toggle
        # ---------------------------------------------------------------------
        st.markdown("##### Candidate Material Specimens")
        mat_cards_cols = st.columns(len(selected_mats))
        for m_col, m_data in zip(mat_cards_cols, selected_mats):
            with m_col:
                m_img = get_material_image_path(m_data["name"])
                m_b64 = get_image_base64(m_img)
                m_bars = compute_mini_bars(m_data)
                rec_pill = get_recyclability_pill(m_data.get("recyclability_status", ""))

                bars_html = "".join(f"""
                <div class="mini-metric-row">
                    <div class="mini-metric-label-val">
                        <span class="mini-metric-label">{b['name']}</span>
                        <span class="mini-metric-val">{b['val_str']}</span>
                    </div>
                    <div class="mini-bar-track">
                        <div class="mini-bar-fill" style="width: {b['pct']:.1f}%; background-color: {b['color']};"></div>
                    </div>
                </div>
                """ for b in m_bars)

                render_html(f"""
                <div class="mat-specimen-card">
                    <img src="{m_b64}" class="mat-specimen-img" />
                    <div class="mat-specimen-body">
                        <div class="mat-specimen-name">{m_data['name'].split('(')[0].strip()}</div>
                        <div style="display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 6px;">
                            <span class="meta-badge" style="font-size: 0.68rem;">{m_data.get('structure_type', 'Flexible')}</span>
                            <span class="meta-badge" style="font-size: 0.68rem;">{m_data.get('nominal_thickness_um', '-')} µm</span>
                        </div>
                        <div class="mini-metrics-container">{bars_html}</div>
                        <div style="margin-top: 4px;">{rec_pill}</div>
                    </div>
                </div>
                """)

                # Add to Compare Basket / Remove Toggle
                is_in_bsk = m_data["name"] in st.session_state.get("compare_basket", [])
                bsk_lbl = "✓ In Compare Basket" if is_in_bsk else "➕ Add to Compare"
                bsk_typ = "primary" if is_in_bsk else "secondary"
                can_add_bsk = is_in_bsk or len(st.session_state.get("compare_basket", [])) < 4
                if st.button(bsk_lbl, key=f"btn_cmp_card_bsk_{m_data['name'][:8]}", type=bsk_typ, disabled=(not can_add_bsk and not is_in_bsk), use_container_width=True):
                    if is_in_bsk:
                        st.session_state["compare_basket"].remove(m_data["name"])
                    else:
                        st.session_state["compare_basket"].append(m_data["name"])
                    st.rerun()

        # ---------------------------------------------------------------------
        # 2. View Controls: Normalized Scale, Chart Type & Weighting Preset
        # ---------------------------------------------------------------------
        ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.1, 1.45, 1.45], gap="medium")
        with ctrl_c1:
            is_normalized = st.toggle(
                "Normalized Scale (0-100)",
                value=False,
                help="Normalized view scores all metrics on a common 0-100 index where higher is always better, preventing scale domination."
            )
        with ctrl_c2:
            chart_view = st.selectbox(
                "Visual Analytics View:",
                options=[
                    "📊 Independent Subplots (WVTR, OTR, Cost, Eco)",
                    "🕸️ Multidimensional Benchmark Radar",
                    "🗺️ Property Matrix Heatmap",
                    "📈 Cost vs Barrier (Pareto Frontier)"
                ],
                key="cmp_chart_view"
            )
        with ctrl_c3:
            weighting_preset = st.selectbox(
                "Decision Weighting Preset:",
                options=[
                    "Balanced (Standard)",
                    "Maximum Shelf-Life & Barrier",
                    "Cost-Optimized / Economy",
                    "Circular & Eco-First"
                ],
                key="cmp_weighting_preset"
            )

        # ---------------------------------------------------------------------
        # 3. "Winner Summary" Card (Best Barrier, Cheapest, Most Eco, Overall Best)
        # ---------------------------------------------------------------------
        best_barrier_mat = min(selected_mats, key=lambda m: float(m.get("wvtr_g_m2_day", 9999)))
        cheapest_mat = min(selected_mats, key=lambda m: float(m.get("cost_index_usd_per_kg", 9999) or 9999))
        most_eco_mat = min(selected_mats, key=lambda m: float(m.get("carbon_footprint_kg_co2_per_kg", 999) or 999))

        preset_weights = {
            "Balanced (Standard)": {"barrier": 0.35, "cost": 0.25, "sustainability": 0.25, "mechanical": 0.15},
            "Maximum Shelf-Life & Barrier": {"barrier": 0.60, "cost": 0.15, "sustainability": 0.15, "mechanical": 0.10},
            "Cost-Optimized / Economy": {"barrier": 0.20, "cost": 0.55, "sustainability": 0.15, "mechanical": 0.10},
            "Circular & Eco-First": {"barrier": 0.20, "cost": 0.15, "sustainability": 0.55, "mechanical": 0.10}
        }.get(weighting_preset, {"barrier": 0.35, "cost": 0.25, "sustainability": 0.25, "mechanical": 0.15})

        cmp_scores = {}
        for m in selected_mats:
            wvtr_v = float(m.get("wvtr_g_m2_day", 10.0))
            b_sc = max(10.0, 100.0 - np.log10(max(0.01, wvtr_v)) * 30.0)
            c_sc = max(10.0, 100.0 - ((float(m.get("cost_index_usd_per_kg", 3.0) or 3.0) - 1.0) * 16.0))
            e_sc = float(analytics.RECYCLABILITY_SCORE_MAP.get(m.get("recyclability_status", ""), 40.0))
            m_sc = min(100.0, float(m.get("tensile_strength_mpa") or 50.0) / 1.5)
            cmp_scores[m["name"]] = min(99.5, max(45.0, (
                b_sc * preset_weights["barrier"] +
                c_sc * preset_weights["cost"] +
                e_sc * preset_weights["sustainability"] +
                m_sc * preset_weights["mechanical"]
            )))

        overall_best_mat = max(selected_mats, key=lambda m: cmp_scores[m["name"]])

        render_html(f"""
        <div class="winner-summary-grid">
            <div class="winner-highlight-card">
                <span class="winner-badge-pill" style="background: rgba(33, 150, 243, 0.12); color: #0288D1;">🛡️ Best Barrier</span>
                <div style="font-weight: 800; font-size: 0.95rem; color: var(--text-main); margin-bottom: 2px;">{best_barrier_mat['name'].split('(')[0].strip()}</div>
                <div style="font-size: 0.76rem; color: var(--text-muted);">WVTR: <b>{best_barrier_mat.get('wvtr_g_m2_day')}</b> g/m²/d</div>
            </div>
            <div class="winner-highlight-card">
                <span class="winner-badge-pill" style="background: rgba(76, 175, 80, 0.12); color: #2E7D32;">💰 Most Economical</span>
                <div style="font-weight: 800; font-size: 0.95rem; color: var(--text-main); margin-bottom: 2px;">{cheapest_mat['name'].split('(')[0].strip()}</div>
                <div style="font-size: 0.76rem; color: var(--text-muted);">Cost: <b>${cheapest_mat.get('cost_index_usd_per_kg', 0):.2f}</b> / kg</div>
            </div>
            <div class="winner-highlight-card">
                <span class="winner-badge-pill" style="background: rgba(46, 125, 50, 0.12); color: #1B5E20;">🌱 Most Eco-Friendly</span>
                <div style="font-weight: 800; font-size: 0.95rem; color: var(--text-main); margin-bottom: 2px;">{most_eco_mat['name'].split('(')[0].strip()}</div>
                <div style="font-size: 0.76rem; color: var(--text-muted);">Carbon: <b>{most_eco_mat.get('carbon_footprint_kg_co2_per_kg', '-')}</b> kg CO2e/kg</div>
            </div>
            <div class="winner-highlight-card" style="border: 2px solid var(--primary-orange); background: linear-gradient(180deg, #FFFFFF 0%, #FFFDF5 100%);">
                <span class="winner-badge-pill" style="background: var(--primary-gradient); color: #FFFFFF;">🏆 Overall Leader</span>
                <div style="font-weight: 800; font-size: 0.95rem; color: var(--primary-deep); margin-bottom: 2px;">{overall_best_mat['name'].split('(')[0].strip()}</div>
                <div style="font-size: 0.76rem; color: var(--text-muted);">Score: <b>{cmp_scores[overall_best_mat['name']]:.1f} / 100</b> ({weighting_preset.split()[0]})</div>
            </div>
        </div>
        """)

        # ---------------------------------------------------------------------
        # 4. Automated Plain-Language Engineering Insight Box
        # ---------------------------------------------------------------------
        sorted_w = sorted(selected_mats, key=lambda x: float(x.get("wvtr_g_m2_day", 9999)))
        sorted_c = sorted(selected_mats, key=lambda x: float(x.get("cost_index_usd_per_kg", 9999) or 9999))
        
        b_top = sorted_w[0]
        b_bot = sorted_w[-1]
        c_top = sorted_c[0]
        c_bot = sorted_c[-1]

        ratio_barrier = float(b_bot.get("wvtr_g_m2_day", 1.0)) / max(0.001, float(b_top.get("wvtr_g_m2_day", 1.0)))
        cost_diff = ((float(c_bot.get("cost_index_usd_per_kg", 1.0) or 1.0) - float(c_top.get("cost_index_usd_per_kg", 1.0) or 1.0)) / max(0.01, float(c_top.get("cost_index_usd_per_kg", 1.0) or 1.0))) * 100.0

        insights_list = []
        if ratio_barrier > 1.2:
            insights_list.append(f"<b>{b_top['name'].split('(')[0].strip()}</b> is a <b>{ratio_barrier:.1f}x better moisture barrier</b> than {b_bot['name'].split('(')[0].strip()} ({b_top.get('wvtr_g_m2_day')} vs {b_bot.get('wvtr_g_m2_day')} g/m²/d).")
        if cost_diff > 10 and c_top["name"] != c_bot["name"]:
            insights_list.append(f"<b>{c_top['name'].split('(')[0].strip()}</b> costs <b>{cost_diff:.0f}% less</b> than {c_bot['name'].split('(')[0].strip()} (${c_top.get('cost_index_usd_per_kg', 0):.2f} vs ${c_bot.get('cost_index_usd_per_kg', 0):.2f}/kg).")

        bio_ones = [m for m in selected_mats if "bio" in m.get("structure_type", "").lower() or "pla" in m.get("name", "").lower()]
        if bio_ones:
            insights_list.append(f"<b>{bio_ones[0]['name'].split('(')[0].strip()}</b> provides bio-based circularity with lower fossil dependence.")

        insight_text = " &bull; ".join(insights_list) if insights_list else "All selected materials exhibit comparable engineering barrier and economic properties."

        render_html(f"""
        <div style="background: linear-gradient(135deg, #FFF8E1 0%, #FFFFFF 100%); border: 1.5px solid rgba(255, 138, 0, 0.4); border-left: 5px solid var(--primary-orange); border-radius: var(--radius-md); padding: 14px 18px; margin-bottom: 18px; box-shadow: var(--shadow-sm);">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="font-size: 1.1rem;">💡</span>
                <span style="font-weight: 800; font-size: 0.90rem; color: var(--primary-deep); font-family: 'Poppins', sans-serif; text-transform: uppercase; letter-spacing: 0.04em;">Automated Engineering Insight</span>
            </div>
            <div style="font-size: 0.88rem; color: var(--text-main); line-height: 1.5;">{insight_text}</div>
        </div>
        """)

        # ---------------------------------------------------------------------
        # 5. Selected Interactive Visual Analytics View
        # ---------------------------------------------------------------------
        if "Subplots" in chart_view:
            st.markdown("#### Independent Comparative Subplots (Log/Linear per Dimension)")
            fig_sub = analytics.create_comparator_subplots(selected_mats, normalized=is_normalized)
            st.plotly_chart(fig_sub, use_container_width=True)
        elif "Radar" in chart_view:
            st.markdown("#### Multidimensional Benchmark Radar (Thick Lines & Shaded Polygons)")
            fig_radar_cmp, r_warns = analytics.create_radar_comparison_chart(selected_mats)
            st.plotly_chart(fig_radar_cmp, use_container_width=True)
            if r_warns:
                for rw in r_warns:
                    st.caption(f"ℹ️ {rw}")
        elif "Heatmap" in chart_view:
            st.markdown("#### Benchmark Property Matrix Heatmap (Green = Better, Red = Concession)")
            fig_hm = analytics.create_comparator_heatmap(selected_mats)
            st.plotly_chart(fig_hm, use_container_width=True)
        elif "Pareto" in chart_view:
            st.markdown("#### Cost vs. Moisture Barrier Trade-off (Pareto Frontier ⭐)")
            fig_pareto = analytics.create_pareto_barrier_cost(all_materials, selected_mats)
            st.plotly_chart(fig_pareto, use_container_width=True)

        # ---------------------------------------------------------------------
        # 6. Comparative Shelf-Life Expectancy Chart for Current Food
        # ---------------------------------------------------------------------
        target_d = int(active_food_preset.get("target_shelf_life_days", 180))
        shelf_life_map = {}
        for m in selected_mats:
            try:
                wvtr_v = float(m.get("wvtr_g_m2_day", 1.0))
                otr_v = float(m.get("otr_cc_m2_day_atm", 1.0))
                s_out = shelf_life.estimate_combined_shelf_life(
                    wvtr_standard=wvtr_v,
                    otr_standard=otr_v,
                    package_surface_area_m2=0.04,
                    product_dry_weight_g=150.0,
                    initial_moisture_percent=2.0,
                    critical_moisture_percent=5.0,
                    headspace_volume_cc=100.0,
                    initial_headspace_o2_percent=0.5,
                    critical_headspace_o2_percent=2.0,
                    ambient_temp_c=float(active_food_preset.get("recommended_temp_c", 23.0)),
                    ambient_rh_percent=float(active_food_preset.get("recommended_rh_percent", 60.0))
                )
                shelf_life_map[m["name"]] = s_out.get("limiting_days", target_d)
            except Exception:
                shelf_life_map[m["name"]] = float(target_d)

        st.markdown("#### Comparative Shelf-Life Expectancy (Fickian Kinetics)")
        fig_shelf_cmp = analytics.create_shelf_life_comparison_bar(
            selected_materials=selected_mats,
            shelf_life_days_map=shelf_life_map,
            target_shelf_life_days=target_d,
            food_name=active_food_preset.get("name", "Active Food")
        )
        st.plotly_chart(fig_shelf_cmp, use_container_width=True)

        # ---------------------------------------------------------------------
        # 7. Comprehensive Technical Specifications Matrix with Direction Arrows
        # ---------------------------------------------------------------------
        st.markdown("#### Comprehensive Technical Specifications Matrix")
        matrix_rows = [
            {"Property / Specification": "Structure Type", "Direction / Unit": "Classification"},
            {"Property / Specification": "Nominal Thickness", "Direction / Unit": "µm"},
            {"Property / Specification": "Water Vapor Transmission (WVTR)", "Direction / Unit": "g/m²/day (38°C, 90% RH) ↓ lower is better"},
            {"Property / Specification": "Oxygen Transmission (OTR)", "Direction / Unit": "cc/m²/day/atm (23°C, 0% RH) ↓ lower is better"},
            {"Property / Specification": "Unit Cost Index", "Direction / Unit": "$ / kg ↓ lower is better"},
            {"Property / Specification": "Embodied Carbon Footprint", "Direction / Unit": "kg CO2e / kg ↓ lower is better"},
            {"Property / Specification": "Tensile Strength", "Direction / Unit": "MPa ↑ higher is better"},
            {"Property / Specification": "Circularity & Recyclability", "Direction / Unit": "Rating ↑ higher is better"},
            {"Property / Specification": "Food Contact Compliance", "Direction / Unit": "FDA / EU Standard"},
            {"Property / Specification": "Verification Level", "Direction / Unit": "Evidence Tier"},
            {"Property / Specification": "Data Source Citation", "Direction / Unit": "Literature"}
        ]

        for m in selected_mats:
            c_cost = m.get("cost_index_usd_per_kg")
            cost_str = f"${c_cost:.2f}" if c_cost is not None else "Unreported"
            co2 = m.get("carbon_footprint_kg_co2_per_kg")
            co2_str = f"{co2:.2f}" if co2 is not None else "Unreported"
            ts = m.get("tensile_strength_mpa")
            ts_str = f"{ts:.1f}" if ts is not None else "Unreported"
            verif = "Lab Verified" if m.get("is_verified_experimental") == 1 else "Published Literature"

            matrix_rows[0][m["name"]] = m.get("structure_type", "—")
            matrix_rows[1][m["name"]] = f"{m.get('nominal_thickness_um', '—')} µm"
            matrix_rows[2][m["name"]] = f"{m.get('wvtr_g_m2_day', '—')}"
            matrix_rows[3][m["name"]] = f"{m.get('otr_cc_m2_day_atm', '—')}"
            matrix_rows[4][m["name"]] = cost_str
            matrix_rows[5][m["name"]] = co2_str
            matrix_rows[6][m["name"]] = ts_str
            matrix_rows[7][m["name"]] = m.get("recyclability_status", "—")
            matrix_rows[8][m["name"]] = m.get("food_contact_compliance", "—")
            matrix_rows[9][m["name"]] = verif
            matrix_rows[10][m["name"]] = m.get("data_source", "—")

        df_matrix = pd.DataFrame(matrix_rows)
        render_html(render_full_width_comparison_matrix(selected_mats))

        # ---------------------------------------------------------------------
        # 8. Export Comparison: CSV & Executive PDF Report
        # ---------------------------------------------------------------------
        render_html("<div style='margin-top: 12px;'></div>")
        exp_c1, exp_c2 = st.columns([1, 1], gap="medium")
        with exp_c1:
            csv_bytes = df_matrix.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Comparison to CSV",
                data=csv_bytes,
                file_name="FoodPackAI_Material_Comparison.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
                key="btn_cmp_export_csv"
            )
        with exp_c2:
            try:
                pdf_bytes = pdf_generator.generate_comparator_pdf(
                    selected_materials=selected_mats,
                    commodity_name=active_food_preset.get("name", "Comparative Analysis"),
                    weighting_preset=weighting_preset
                )
                st.download_button(
                    "📄 Export Comparison to PDF Report",
                    data=pdf_bytes,
                    file_name="FoodPackAI_Material_Comparison.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                    key="btn_cmp_export_pdf"
                )
            except Exception as e:
                st.error(f"Failed to generate PDF report: {e}")


# =============================================================================
# TAB 3: SHELF-LIFE SIMULATOR (First-Principles Permeation Kinetics)
# =============================================================================
elif active_tab == "⏱️ Shelf-Life Simulator":
    st.markdown("### First-Principles Shelf-Life Permeation Simulator")
    render_html("<p style='color: var(--text-muted); font-size: 0.92rem; margin-top: -8px;'>Simulate dynamic moisture uptake and headspace oxygen accumulation over time using Robertson & Labuza Fickian mass-transfer equations.</p>")

    sim_col1, sim_col2 = st.columns([1, 1], gap="large")

    with sim_col1:
        st.markdown("##### 1. Select Material & Package Geometry")
        sim_mat_name = st.selectbox(
            "Packaging Material",
            options=all_mat_names,
            index=3,  # BOPP / Met-BOPP / PE
            key="sim_mat_select"
        )
        selected_sim_mat = next(m for m in all_materials if m["name"] == sim_mat_name)
        sim_thumb = get_material_image_path(selected_sim_mat["name"])
        if sim_thumb and os.path.exists(sim_thumb):
            st.image(sim_thumb, width=140, caption=f"Specimen: {selected_sim_mat.get('structure_type', '')}")

        sim_area_cm2 = st.number_input("Package Surface Area (cm²)", 50.0, 2000.0, 400.0, 50.0, key="sim_area_input")
        sim_area_m2 = sim_area_cm2 / 10000.0

        sim_dry_weight = st.number_input("Product Dry Weight (g)", 10.0, 5000.0, 150.0, 25.0, key="sim_weight_input")
        sim_headspace_cc = st.number_input("Headspace Volume (cc)", 10.0, 1000.0, 100.0, 10.0, key="sim_headspace_input")

    with sim_col2:
        st.markdown("##### 2. Food Sensitivities & Storage Environment")
        sim_t_col1, sim_t_col2 = st.columns(2)
        with sim_t_col1:
            sim_init_m = st.number_input("Initial Moisture (% dry basis)", 0.5, 30.0, 2.0, 0.5, key="sim_init_m")
        with sim_t_col2:
            sim_crit_m = st.number_input("Critical Moisture (% dry basis)", 1.0, 50.0, 5.0, 0.5, key="sim_crit_m")

        sim_o_col1, sim_o_col2 = st.columns(2)
        with sim_o_col1:
            sim_init_o2 = st.number_input("Initial Headspace O2 (%)", 0.0, 21.0, 0.5, 0.1, key="sim_init_o2")
        with sim_o_col2:
            sim_crit_o2 = st.number_input("Critical Headspace O2 (%)", 0.5, 21.0, 2.0, 0.1, key="sim_crit_o2")

        sim_env1, sim_env2 = st.columns(2)
        with sim_env1:
            sim_temp = st.number_input("Ambient Temperature (°C)", 0.0, 50.0, 23.0, 1.0, key="sim_temp_input")
        with sim_env2:
            sim_rh = st.slider("Ambient Humidity (% RH)", 10, 95, 60, 5, key="sim_rh_input")

    sim_btn = st.button("▶ Run Permeation Simulation", type="primary", use_container_width=True, key="btn_run_sim")

    if sim_btn or "sim_result" in st.session_state:
        if sim_crit_m <= sim_init_m:
            st.error("Critical moisture percentage must be greater than initial moisture percentage.")
        elif sim_crit_o2 <= sim_init_o2:
            st.error("Critical headspace O2 percentage must be greater than initial headspace O2 percentage.")
        else:
            wvtr_val = selected_sim_mat.get("wvtr_g_m2_day", 1.0)
            otr_val = selected_sim_mat.get("otr_cc_m2_day_atm", 1.0)

            sim_res = shelf_life.estimate_combined_shelf_life(
                wvtr_standard=wvtr_val,
                otr_standard=otr_val,
                package_surface_area_m2=sim_area_m2,
                product_dry_weight_g=sim_dry_weight,
                initial_moisture_percent=sim_init_m,
                critical_moisture_percent=sim_crit_m,
                headspace_volume_cc=sim_headspace_cc,
                initial_headspace_o2_percent=sim_init_o2,
                critical_headspace_o2_percent=sim_crit_o2,
                ambient_temp_c=sim_temp,
                ambient_rh_percent=sim_rh
            )
            st.session_state["sim_result"] = sim_res

            render_html("<hr style='border: 0; border-top: 1px solid var(--border-color); margin: 24px 0 16px 0;' />")
            st.markdown("#### Permeation Kinetics & Shelf-Life Estimate")

            lim_factor = sim_res.get("limiting_factor", "Unknown")
            pred_days = sim_res.get("limiting_days", 0.0)

            kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
            with kpi_c1:
                st.metric("Predicted Shelf Life", f"{pred_days:.0f} Days")
            with kpi_c2:
                st.metric("Primary Limiting Factor", lim_factor.replace("_", " ").title())
            with kpi_c3:
                st.metric("Degradation Mode", "Steady-State Fickian Diffusion")

            # Permeation Curve Graph
            m_res = sim_res.get("moisture_result", {})
            o_res = sim_res.get("oxygen_result", {})

            time_axis = m_res.get("days_array", [])
            moisture_curve = m_res.get("moisture_curve", [])

            fig_kinetics = go.Figure()
            if len(time_axis) > 0 and len(moisture_curve) > 0:
                fig_kinetics.add_trace(go.Scatter(
                    x=time_axis,
                    y=moisture_curve,
                    name="Product Moisture (% dry basis)",
                    line=dict(color="#FF8A00", width=2.5)
                ))
            fig_kinetics.add_hline(
                y=sim_crit_m,
                line_dash="dash",
                line_color="#E53935",
                annotation_text=f"Critical Moisture Threshold ({sim_crit_m}%)",
                annotation_position="bottom right"
            )

            fig_kinetics.update_layout(
                title=f"Moisture Permeation Kinetic Curve: {sim_mat_name}",
                xaxis_title="Storage Duration (Days)",
                yaxis_title="Moisture Content (% dry basis)",
                height=360,
                margin=dict(l=40, r=30, t=40, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#3E2723", family="Inter, sans-serif"),
                xaxis=dict(showgrid=True, gridcolor="rgba(62, 39, 35, 0.08)", linecolor="rgba(62, 39, 35, 0.2)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(62, 39, 35, 0.08)", linecolor="rgba(62, 39, 35, 0.2)")
            )
            st.plotly_chart(fig_kinetics, use_container_width=True)

            st.caption(
                "Scientific Disclaimer: Model assumes steady-state Fickian mass transfer across pristine packaging web under controlled storage. "
                "Actual commercial shelf life is influenced by pinholes, seal hermeticity, respiration kinetics (for fresh produce), "
                "and supply-chain temperature fluctuations. Accelerated Shelf-Life Testing (ASLT) is advised prior to production sign-off."
            )


# =============================================================================
# TAB 4: MATERIALS & STANDARDS (Database Inventory & Regulatory Reference)
# =============================================================================
elif active_tab == "📦 Materials & Standards":
    st.markdown("### Packaging Materials Inventory & Standards Reference")
    render_html("<p style='color: var(--text-muted); font-size: 0.92rem; margin-top: -8px;'>Explore the literature-referenced benchmark dataset, audit trail, and regulatory food-contact standards.</p>")

    mat_df = database.get_all_materials_df()

    # Search & Filter
    search_q = st.text_input("Filter Materials by Name or Polymer Code:", "", key="mat_search_input")
    if search_q:
        mat_df = mat_df[mat_df["name"].str.contains(search_q, case=False, na=False) | mat_df["polymer_code"].str.contains(search_q, case=False, na=False)]

    display_cols = [
        "name", "structure_type", "polymer_code", "nominal_thickness_um",
        "wvtr_g_m2_day", "otr_cc_m2_day_atm", "cost_index_usd_per_kg",
        "carbon_footprint_kg_co2_per_kg", "recyclability_status", "food_contact_compliance", "data_source"
    ]
    existing_cols = [c for c in display_cols if c in mat_df.columns]
    st.dataframe(mat_df[existing_cols], use_container_width=True, hide_index=True)

    csv_data = mat_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export Materials Database to CSV",
        data=csv_data,
        file_name="FoodPackagingAI_Materials_Database.csv",
        mime="text/csv",
        type="primary",
        key="btn_export_csv"
    )

    render_html("<hr style='border: 0; border-top: 1px solid var(--border-color); margin: 24px 0;' />")

    # Regulatory Reference
    st.markdown("#### Primary Regulatory & Test Method Standards")
    reg_c1, reg_c2 = st.columns(2)
    with reg_c1:
        st.markdown("""
        **Food-Contact Safety Standards:**
        - **US FDA 21 CFR §177:** Indirect Food Additives: Polymers (Basic components of single and repeated use food-contact surfaces).
        - **EU Regulation (EU) No 10/2011:** Plastic materials and articles intended to come into contact with food (Overall & specific migration limits).
        - **FSSAI Food Safety and Standards (Packaging) Regulations 2018:** Indian national migration and heavy metal compliance limits.
        """)
    with reg_c2:
        st.markdown("""
        **Standard Physical Test Conditions:**
        - **ASTM F1249:** Standard Test Method for Water Vapor Transmission Rate (WVTR) Through Plastic Film and Sheeting Using a Modulated Infrared Sensor (38.0°C, 90% RH).
        - **ASTM D3985 / ASTM F1927:** Standard Test Method for Oxygen Transmission Rate (OTR) Through Plastic Film and Sheeting Using a Coulometric Sensor (23.0°C, 0% RH).
        """)

# =============================================================================
# Premium Platform Footer & Standards Badges
# =============================================================================
render_html(
    """
    <div class="app-footer">
        <div style="font-family: 'Poppins', sans-serif; font-size: 1.15rem; font-weight: 800; color: var(--text-main); margin-bottom: 4px;">
            Smart FoodPack <span class="gradient-heading">AI</span>
        </div>
        <div style="font-size: 0.85rem; color: var(--text-muted); max-width: 600px; margin: 0 auto 12px auto; line-height: 1.45;">
            Intelligent Food Packaging Material Decision Support System &bull; Multi-Criteria Optimization &amp; Shelf-Life Permeation Kinetics.
        </div>
        <div class="footer-badges">
            <span class="footer-badge">🐍 Python 3.14</span>
            <span class="footer-badge">⚡ Streamlit</span>
            <span class="footer-badge">📊 Plotly</span>
            <span class="footer-badge">🗄️ SQLite 3</span>
            <span class="footer-badge">🤖 scikit-learn</span>
            <span class="footer-badge">🛡️ ASTM F1249 / D3985</span>
            <span class="footer-badge">📜 FDA 21 CFR §177</span>
            <span class="footer-badge">🇪🇺 EU 10/2011</span>
            <span class="footer-badge">🇮🇳 FSSAI 2018</span>
        </div>
    </div>
    """
)
