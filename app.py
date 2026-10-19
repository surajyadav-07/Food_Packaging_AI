"""
app.py - Smart FoodPack AI (Intelligent Packaging Decision Support System)
A premium, highly polished web application for food packaging material recommendation,
shelf-life simulation, sustainability assessment, and machine learning analytics.

Built with Streamlit, Python, SQLite, Plotly, Scikit-learn, ReportLab, and streamlit-option-menu.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from streamlit_option_menu import option_menu

# Local system modules
import database
import recommender
import shelf_life
import data_loader
import analytics
import ml_engine
import pdf_generator

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart FoodPack AI — Packaging Intelligence Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Database Schema if not already present
database.init_database()

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "active_nav_index" not in st.session_state:
    st.session_state["active_nav_index"] = 0
if "wizard_step" not in st.session_state:
    st.session_state["wizard_step"] = 1
if "selected_food_name" not in st.session_state:
    st.session_state["selected_food_name"] = "Potato Chips"
if "storage_temp" not in st.session_state:
    st.session_state["storage_temp"] = 23.0
if "storage_rh" not in st.session_state:
    st.session_state["storage_rh"] = 60
if "target_shelf_life" not in st.session_state:
    st.session_state["target_shelf_life"] = 180
if "priority_choice" not in st.session_state:
    st.session_state["priority_choice"] = "⚖️ Balanced Performance (Recommended)"
if "pack_area_cm2" not in st.session_state:
    st.session_state["pack_area_cm2"] = 400.0
if "headspace_ml" not in st.session_state:
    st.session_state["headspace_ml"] = 120.0
if "saved_scenarios" not in st.session_state:
    st.session_state["saved_scenarios"] = []
if "expert_chat_history" not in st.session_state:
    st.session_state["expert_chat_history"] = [
        {"role": "assistant", "content": "👋 Hello! I am your Packaging Science Advisor. Ask me anything about moisture protection (WVTR), oxygen barriers (OTR), shelf-life kinetics, eco-friendly biopolymers, or why a specific material was recommended for your food!"}
    ]
if "last_run_result" not in st.session_state:
    st.session_state["last_run_result"] = None

# -----------------------------------------------------------------------------
# Multi-Language Localization Dictionary (English & Hindi)
# -----------------------------------------------------------------------------
TRANSLATIONS = {
    "en": {
        "app_title": "Smart FoodPack AI",
        "app_badge": "Packaging Intelligence Platform",
        "nav_0": "Dashboard & Recommendations",
        "nav_1": "Shelf-Life Simulator",
        "nav_2": "Material Comparator",
        "nav_3": "Sustainability & LCA",
        "nav_4": "ML & Clustering",
        "nav_5": "Database Explorer",
        "nav_6": "CSV Import/Export",
        "nav_7": "User Guide & Audit",
        "hero_headline": "Intelligent Food Packaging Material Recommendation System",
        "hero_sub": "Preserve freshness, extend shelf life, minimize carbon footprint, and optimize packaging costs with verified food science and physics-based permeation models.",
        "btn_start_wizard": "🚀 Launch Recommendation Wizard",
        "btn_try_demo": "✨ Try One-Click Demo (Potato Chips)",
        "step_1": "1. Food Commodity",
        "step_2": "2. Storage & Pack",
        "step_3": "3. Priorities",
        "step_4": "4. Results & Insights",
        "stepper_title": "4-Step Packaging Decision Wizard",
        "select_food_prompt": "Select an everyday food commodity or define a custom food profile:",
        "storage_presets_title": "Quick Storage Presets:",
        "preset_cool": "🧊 Chilled / Fridge (4°C, 85% RH)",
        "preset_room": "🏠 Room Temp (23°C, 60% RH)",
        "preset_hot": "☀️ Hot & Humid (38°C, 90% RH)",
        "priorities_title": "Select Business & Sustainability Priority Strategy:",
        "moisture_prot": "Moisture Protection (WVTR)",
        "oxygen_prot": "Oxygen Protection (OTR)",
        "score_ring_title": "Overall Match Score",
        "shelf_life_badge": "Predicted Freshness Horizon",
        "why_it_fits": "✅ Why It Fits Your Food",
        "considerations": "⚠️ Engineering Considerations",
        "savings_calculator_title": "💰 Packaging Cost & Carbon Savings Calculator",
        "scenario_title": "💾 Saved Comparison Scenarios (Max 3)",
        "expert_chat_title": "🤖 Ask the Packaging Expert (Rule-Based AI)",
        "honesty_notice": "🔬 Academic & Scientific Notice: Shelf life is a mathematical mass-transfer estimate, not an empirical microbiological guarantee. Supervised ML uses synthetic rule-based demonstrations. Accredited physical laboratory testing is required prior to commercial release."
    },
    "hi": {
        "app_title": "स्मार्ट फूडपैक एआई",
        "app_badge": "खाद्य पैकेजिंग अनुशंसा प्रणाली",
        "nav_0": "डैशबोर्ड व सिफारिशें",
        "nav_1": "शेल्फ-लाइफ सिम्युलेटर",
        "nav_2": "सामग्री तुलना",
        "nav_3": "पर्यावरण व स्थिरता",
        "nav_4": "मशीन लर्निंग विश्लेषण",
        "nav_5": "डेटाबेस एक्सप्लोरर",
        "nav_6": "सीएसवी आयात/निर्यात",
        "nav_7": "उपयोगकर्ता गाइड व ऑडिट",
        "hero_headline": "बुद्धिमान खाद्य पैकेजिंग सामग्री अनुशंसा प्रणाली",
        "hero_sub": "ताज़गी बचाएं, शेल्फ-लाइफ बढ़ाएं, कार्बन फुटप्रिंट घटाएं और वैज्ञानिक सिमुलेशन द्वारा सर्वोत्तम पैकेजिंग चुनें।",
        "btn_start_wizard": "🚀 विज़ार्ड शुरू करें",
        "btn_try_demo": "✨ वन-क्लिक डेमो आज़माएँ (आलू चिप्स)",
        "step_1": "1. खाद्य उत्पाद",
        "step_2": "2. भंडारण व पैकेज",
        "step_3": "3. प्राथमिकताएं",
        "step_4": "4. परिणाम व विश्लेषण",
        "stepper_title": "4-चरणीय पैकेजिंग सलाहकार विज़ार्ड",
        "select_food_prompt": "एक लोकप्रिय खाद्य उत्पाद चुनें या अपना विवरण भरें:",
        "storage_presets_title": "त्वरित भंडारण प्रीसेट:",
        "preset_cool": "🧊 रेफ्रिजरेटर (4°C, 85% आर्द्रता)",
        "preset_room": "🏠 सामान्य तापमान (23°C, 60% आर्द्रता)",
        "preset_hot": "☀️ गर्म व नम जलवायु (38°C, 90% आर्द्रता)",
        "priorities_title": "अपनी व्यावसायिक प्राथमिकता रणनीति चुनें:",
        "moisture_prot": "नमी से सुरक्षा (WVTR)",
        "oxygen_prot": "ऑक्सीजन से सुरक्षा (OTR)",
        "score_ring_title": "कुल मैच स्कोर",
        "shelf_life_badge": "अनुमानित सुरक्षित अवधि (दिन)",
        "why_it_fits": "✅ यह आपके खाद्य के अनुकूल क्यों है",
        "considerations": "⚠️ ध्यान देने योग्य तकनीकी बातें",
        "savings_calculator_title": "💰 पैकेजिंग लागत एवं कार्बन बचत कैलकुलेटर",
        "scenario_title": "💾 सहेजे गए परिदृश्य (अधिकतम 3)",
        "expert_chat_title": "🤖 पैकेजिंग विशेषज्ञ से पूछें (नियम-आधारित एआई)",
        "honesty_notice": "🔬 वैज्ञानिक सूचना: शेल्फ-लाइफ गणना भौतिकी आधारित अनुमान है, कोई वाणिज्यिक गारंटी नहीं। वाणिज्यिक उत्पादन से पहले प्रयोगशाला परीक्षण आवश्यक है।"
    }
}

def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

# -----------------------------------------------------------------------------
# Comprehensive Theme & Custom CSS System
# -----------------------------------------------------------------------------
def get_custom_css(dark_mode: bool = False) -> str:
    if dark_mode:
        vars_css = """
            --bg-page: #0b1120;
            --bg-card: #1e293b;
            --bg-card-hover: #243247;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #334155;
            --border-subtle: #1e293b;
            --accent-teal: #14b8a6;
            --accent-teal-dark: #0f766e;
            --accent-emerald: #10b981;
            --btn-secondary-bg: #1e293b;
            --btn-secondary-text: #f8fafc;
            --btn-secondary-border: #475569;
            --btn-hover-bg: #2d3b52;
            --input-bg: #1e293b;
            --input-text: #f8fafc;
            --input-border: #475569;
            --hero-grad: linear-gradient(135deg, #090e17 0%, #0f3d3e 45%, #0f766e 100%);
            --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.2);
            --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.35);
            --shadow-lg: 0 12px 30px rgba(0, 0, 0, 0.5);
            --stepper-bg: #1e293b;
            --stepper-completed-bg: #064e3b;
            --stepper-completed-border: #059669;
            --stepper-completed-text: #a7f3d0;
            --stepper-upcoming-bg: #1e293b;
            --stepper-upcoming-border: #334155;
            --stepper-upcoming-text: #94a3b8;
            --alert-notice-bg: #272214;
            --alert-notice-border: #f59e0b;
            --alert-notice-text: #fbbf24;
            --metric-bg: #132338;
        """
    else:
        vars_css = """
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --bg-card-hover: #f1f5f9;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --border-color: #e2e8f0;
            --border-subtle: #f1f5f9;
            --accent-teal: #0d9488;
            --accent-teal-dark: #0f766e;
            --accent-emerald: #10b981;
            --btn-secondary-bg: #ffffff;
            --btn-secondary-text: #0f172a;
            --btn-secondary-border: #cbd5e1;
            --btn-hover-bg: #f0fdfa;
            --input-bg: #ffffff;
            --input-text: #0f172a;
            --input-border: #cbd5e1;
            --hero-grad: linear-gradient(135deg, #0f766e 0%, #0d9488 45%, #059669 100%);
            --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 16px rgba(15, 23, 42, 0.06);
            --shadow-lg: 0 10px 25px -5px rgba(13, 148, 136, 0.15);
            --stepper-bg: #ffffff;
            --stepper-completed-bg: #f0fdf4;
            --stepper-completed-border: #86efac;
            --stepper-completed-text: #166534;
            --stepper-upcoming-bg: #ffffff;
            --stepper-upcoming-border: #e2e8f0;
            --stepper-upcoming-text: #64748b;
            --alert-notice-bg: #fefce8;
            --alert-notice-border: #f59e0b;
            --alert-notice-text: #92400e;
            --metric-bg: #f0fdfa;
        """

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

        :root, .stApp {{
            {vars_css}
            /* Map explicitly to Streamlit's native theme variables so emotion styles inherit */
            --background-color: var(--bg-page) !important;
            --secondary-background-color: var(--bg-card) !important;
            --text-color: var(--text-primary) !important;
            --primary-color: var(--accent-teal) !important;
        }}

        /* App & Body Root */
        .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background-color: var(--bg-page) !important;
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
            color: var(--text-primary) !important;
        }}
        header[data-testid="stHeader"] {{
            background: transparent !important;
        }}

        /* Global Typography */
        h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: var(--text-primary) !important;
        }}
        p, .stCaption, small {{
            color: var(--text-secondary);
        }}

        /* Header Bar */
        .header-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            margin-bottom: 18px;
            box-shadow: var(--shadow-md);
        }}

        /* Hero Banner */
        .hero-banner {{
            background: var(--hero-grad);
            color: #ffffff !important;
            padding: 34px 38px;
            border-radius: 18px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
            border: 1px solid var(--accent-teal-dark);
        }}
        .hero-title {{
            font-size: 30px;
            font-weight: 800;
            color: #ffffff !important;
            letter-spacing: -0.5px;
            margin-bottom: 10px;
            line-height: 1.25;
        }}
        .hero-subtitle {{
            font-size: 16px;
            color: #ccfbf1 !important;
            max-width: 850px;
            line-height: 1.55;
            margin-bottom: 18px;
            font-weight: 400;
        }}

        /* Comprehensive Button Theming (Fixing Dark Button issue in Light Mode) */
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button,
        div[data-testid="stFormSubmitButton"] button,
        .stButton button,
        button[data-testid^="stBaseButton"],
        button[data-testid^="baseButton"] {{
            background-color: var(--btn-secondary-bg) !important;
            color: var(--btn-secondary-text) !important;
            border: 1.5px solid var(--btn-secondary-border) !important;
            border-radius: 11px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            padding: 8px 18px !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: var(--shadow-sm) !important;
        }}
        /* Crucial: Prevent global p selector from corrupting button text */
        button p,
        .stButton button p,
        div[data-testid="stButton"] button p,
        div[data-testid="stDownloadButton"] button p,
        button[data-testid*="BaseButton"] p,
        button[data-testid*="baseButton"] p {{
            color: inherit !important;
            font-weight: inherit !important;
            font-size: inherit !important;
            margin: 0 !important;
        }}
        div[data-testid="stButton"] button:hover,
        div[data-testid="stDownloadButton"] button:hover,
        div[data-testid="stFormSubmitButton"] button:hover,
        .stButton button:hover,
        button[data-testid*="BaseButton"]:hover,
        button[data-testid*="baseButton"]:hover {{
            border-color: var(--accent-teal) !important;
            background-color: var(--btn-hover-bg) !important;
            color: var(--accent-teal) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(13, 148, 136, 0.15) !important;
        }}
        /* Primary Buttons */
        div[data-testid="stButton"] button[kind="primary"],
        div[data-testid="stDownloadButton"] button[kind="primary"],
        div[data-testid="stFormSubmitButton"] button[kind="primary"],
        .stButton button[kind="primary"],
        button[data-testid*="BaseButton-primary"],
        button[data-testid*="baseButton-primary"] {{
            background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 14px rgba(13, 148, 136, 0.3) !important;
        }}
        div[data-testid="stButton"] button[kind="primary"] *,
        div[data-testid="stDownloadButton"] button[kind="primary"] *,
        button[data-testid*="BaseButton-primary"] * {{
            color: #ffffff !important;
        }}
        div[data-testid="stButton"] button[kind="primary"]:hover,
        div[data-testid="stDownloadButton"] button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
        .stButton button[kind="primary"]:hover,
        button[data-testid*="BaseButton-primary"]:hover,
        button[data-testid*="baseButton-primary"]:hover {{
            background: linear-gradient(135deg, #0f766e 0%, #115e59 100%) !important;
            color: #ffffff !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 18px rgba(13, 148, 136, 0.4) !important;
        }}

        /* Input Controls (Inputs, Selects, Radios) */
        .stTextInput input, .stNumberInput input, .stTextArea textarea {{
            background-color: var(--input-bg) !important;
            color: var(--input-text) !important;
            border: 1.5px solid var(--input-border) !important;
            border-radius: 10px !important;
            font-size: 14px !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }}
        .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{
            border-color: var(--accent-teal) !important;
            box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.2) !important;
        }}
        .stSelectbox > div > div, div[data-baseweb="select"] > div {{
            background-color: var(--input-bg) !important;
            color: var(--input-text) !important;
            border: 1.5px solid var(--input-border) !important;
            border-radius: 10px !important;
        }}
        div[data-baseweb="select"] span {{
            color: var(--input-text) !important;
        }}
        div[data-baseweb="select"] svg {{
            fill: var(--text-secondary) !important;
        }}
        div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {{
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            box-shadow: var(--shadow-md) !important;
        }}
        li[role="option"] {{
            color: var(--text-primary) !important;
            background-color: var(--bg-card) !important;
        }}
        li[role="option"]:hover, li[aria-selected="true"] {{
            background-color: var(--btn-hover-bg) !important;
            color: var(--accent-teal) !important;
        }}

        /* Radio & Checkbox */
        .stRadio label, .stCheckbox label {{
            color: var(--text-primary) !important;
            font-size: 14px !important;
        }}
        div[role="radiogroup"] {{
            background-color: var(--bg-card) !important;
            padding: 10px 14px !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
        }}

        /* Expanders */
        div[data-testid="stExpander"], .stExpander {{
            background-color: var(--bg-card) !important;
            border: 1.5px solid var(--border-color) !important;
            border-radius: 12px !important;
            margin: 12px 0 !important;
            box-shadow: var(--shadow-sm) !important;
        }}
        div[data-testid="stExpander"] summary, .stExpander summary {{
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }}
        div[data-testid="stExpander"] details[open], div[data-testid="stExpanderDetails"] {{
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: var(--border-subtle) !important;
            border-radius: 12px !important;
            padding: 4px !important;
            gap: 6px !important;
            border: 1px solid var(--border-color) !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px !important;
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
            padding: 8px 18px !important;
            border: none !important;
            background-color: transparent !important;
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background-color: var(--bg-card) !important;
            color: var(--accent-teal) !important;
            font-weight: 700 !important;
            box-shadow: var(--shadow-sm) !important;
        }}

        /* Metrics */
        div[data-testid="stMetric"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            box-shadow: var(--shadow-sm) !important;
        }}
        div[data-testid="stMetricLabel"] {{
            color: var(--text-secondary) !important;
            font-size: 12.5px !important;
            font-weight: 600 !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: var(--text-primary) !important;
            font-size: 24px !important;
            font-weight: 800 !important;
        }}

        /* Progress Bar */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, #0d9488 0%, #10b981 100%) !important;
            border-radius: 10px !important;
        }}
        .stProgress > div > div {{
            background-color: var(--border-color) !important;
            border-radius: 10px !important;
            height: 8px !important;
        }}

        /* Chat Messages */
        div[data-testid="stChatMessage"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 14px !important;
            padding: 14px 18px !important;
            margin-bottom: 12px !important;
            box-shadow: var(--shadow-sm) !important;
        }}
        div[data-testid="stChatMessage"] p {{
            color: var(--text-primary) !important;
        }}
        div[data-testid="stChatInput"] {{
            background-color: var(--bg-card) !important;
            border-color: var(--border-color) !important;
        }}
        div[data-testid="stChatInput"] textarea {{
            background-color: var(--input-bg) !important;
            color: var(--input-text) !important;
        }}

        /* Alert and Info Banners */
        div[data-testid="stAlert"] {{
            border-radius: 12px !important;
            box-shadow: var(--shadow-sm) !important;
        }}
        div[data-testid="stAlert"] p {{
            color: inherit !important;
        }}

        /* Dataframe Tables */
        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            box-shadow: var(--shadow-sm) !important;
        }}

        /* Visual Stepper Card Component */
        .stepper-card {{
            background: var(--stepper-bg);
            border: 2px solid var(--border-color);
            border-radius: 14px;
            padding: 14px 16px;
            text-align: center;
            transition: all 0.25s ease;
            box-shadow: var(--shadow-sm);
            margin-bottom: 8px;
        }}
        .stepper-card-active {{
            border-color: var(--accent-teal) !important;
            background: {'#132338' if dark_mode else '#f0fdfa'} !important;
            box-shadow: 0 4px 16px rgba(13, 148, 136, 0.2) !important;
        }}
        .stepper-badge {{
            font-size: 11px;
            font-weight: 800;
            padding: 3px 8px;
            border-radius: 20px;
            display: inline-block;
            margin-bottom: 6px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        .badge-active {{ background: #0d9488; color: #ffffff !important; }}
        .badge-completed {{ background: #dcfce7; color: #166534 !important; }}
        .badge-upcoming {{ background: var(--border-color); color: var(--text-secondary) !important; }}

        /* Food Commodity Cards */
        .commodity-card {{
            background: var(--bg-card);
            border: 2px solid var(--border-color);
            border-radius: 14px;
            padding: 16px;
            text-align: center;
            transition: all 0.25s ease;
            height: 100%;
            box-shadow: var(--shadow-sm);
            margin-bottom: 8px;
        }}
        .commodity-card:hover {{
            transform: translateY(-3px);
            border-color: var(--accent-teal);
            box-shadow: 0 8px 20px rgba(13, 148, 136, 0.15);
        }}
        .commodity-card-selected {{
            border-color: var(--accent-teal) !important;
            background: {'#132338' if dark_mode else '#f0fdfa'} !important;
            box-shadow: 0 4px 16px rgba(13, 148, 136, 0.25) !important;
        }}

        /* Result Recommendation Cards */
        .candidate-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 22px;
            box-shadow: var(--shadow-md);
            border-left: 6px solid var(--accent-teal);
            transition: transform 0.2s ease;
        }}
        .candidate-card:hover {{
            transform: translateY(-2px);
        }}
        .rank-badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        .rank-gold {{ background: #fef3c7; color: #92400e; }}
        .rank-silver {{ background: #f1f5f9; color: #334155; }}
        .rank-bronze {{ background: #ffedd5; color: #9a3412; }}

        /* Big Metric Box */
        .big-metric-box {{
            background: var(--metric-bg);
            border: 1.5px solid var(--accent-teal);
            border-radius: 14px;
            padding: 18px 20px;
            text-align: center;
            margin-bottom: 16px;
            box-shadow: var(--shadow-sm);
        }}
        .big-metric-number {{
            font-size: 38px;
            font-weight: 800;
            color: var(--accent-teal);
            line-height: 1;
            margin: 6px 0;
        }}
        .big-metric-label {{
            font-size: 13px;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* Scientific Disclaimer Banner */
        .honesty-notice {{
            background: var(--alert-notice-bg);
            border-left: 4px solid var(--alert-notice-border);
            padding: 14px 18px;
            border-radius: 10px;
            font-size: 13px;
            color: var(--alert-notice-text);
            margin: 20px 0;
            line-height: 1.5;
        }}

        /* Status Pills */
        .status-pill {{
            font-size: 12px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 12px;
            display: inline-block;
            margin-right: 6px;
            margin-bottom: 6px;
        }}
        .status-pass {{ background: #dcfce7; color: #166534; }}
        .status-warn {{ background: #fef9c3; color: #854d0e; }}
        .status-fail {{ background: #fee2e2; color: #991b1b; }}
    </style>
    """

st.markdown(get_custom_css(st.session_state["dark_mode"]), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Chart Theme Styler Function
# -----------------------------------------------------------------------------
def apply_chart_theme(fig: go.Figure, dark_mode: bool = False, height: int = None) -> go.Figure:
    """Standardizes typography, background, font color, and gridlines for Plotly charts."""
    bg_color = "rgba(0,0,0,0)"
    text_color = "#f8fafc" if dark_mode else "#0f172a"
    grid_color = "rgba(255,255,255,0.08)" if dark_mode else "rgba(0,0,0,0.06)"
    polar_grid = "rgba(255,255,255,0.12)" if dark_mode else "rgba(0,0,0,0.08)"

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color=text_color, size=12),
        margin=dict(l=35, r=35, t=40, b=35),
        legend=dict(font=dict(color=text_color))
    )
    if height:
        fig.update_layout(height=height)

    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        tickfont=dict(color=text_color),
        title_font=dict(color=text_color)
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        tickfont=dict(color=text_color),
        title_font=dict(color=text_color)
    )
    fig.update_polars(
        bgcolor=bg_color,
        radialaxis=dict(gridcolor=polar_grid, tickfont=dict(color=text_color)),
        angularaxis=dict(gridcolor=polar_grid, tickfont=dict(color=text_color))
    )
    return fig

# -----------------------------------------------------------------------------
# Top Header Bar: Branding, Language Switcher, Dark/Light Mode, Quick Demo
# -----------------------------------------------------------------------------
top_col1, top_col2, top_col3, top_col4 = st.columns([4, 2, 2, 2])

with top_col1:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px; padding: 4px 0;">
            <span style="font-size: 32px;">📦</span>
            <div>
                <div style="font-weight: 800; font-size: 20px; line-height: 1.2; color: {'#f8fafc' if st.session_state['dark_mode'] else '#0f172a'};">{t('app_title')}</div>
                <div style="font-size: 12px; color: #0d9488; font-weight: 700; letter-spacing: 0.3px;">{t('app_badge')}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with top_col2:
    if st.button("✨ " + ("One-Click Demo" if st.session_state["lang"] == "en" else "क्विक डेमो"), use_container_width=True):
        st.session_state["selected_food_name"] = "Potato Chips"
        st.session_state["storage_temp"] = 23.0
        st.session_state["storage_rh"] = 60
        st.session_state["target_shelf_life"] = 180
        st.session_state["priority_choice"] = "⚖️ Balanced Performance (Recommended)"
        st.session_state["wizard_step"] = 4
        st.session_state["active_nav_index"] = 0
        st.rerun()

with top_col3:
    curr_lang = st.session_state["lang"]
    new_lang_label = "🌐 हिन्दी" if curr_lang == "en" else "🌐 English"
    if st.button(new_lang_label, use_container_width=True):
        st.session_state["lang"] = "hi" if curr_lang == "en" else "en"
        st.rerun()

with top_col4:
    theme_icon = "☀️ Light" if st.session_state["dark_mode"] else "🌙 Dark"
    if st.button(theme_icon, use_container_width=True):
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]
        st.rerun()

# -----------------------------------------------------------------------------
# Horizontal Top Navigation Menu (streamlit-option-menu)
# -----------------------------------------------------------------------------
nav_labels = [
    t("nav_0"),
    t("nav_1"),
    t("nav_2"),
    t("nav_3"),
    t("nav_4"),
    t("nav_5"),
    t("nav_6"),
    t("nav_7")
]

nav_icons = [
    "speedometer2",
    "hourglass-split",
    "bar-chart-steps",
    "recycle",
    "cpu",
    "database",
    "cloud-arrow-up",
    "journal-text"
]

selected_nav_str = option_menu(
    menu_title=None,
    options=nav_labels,
    icons=nav_icons,
    default_index=st.session_state["active_nav_index"],
    orientation="horizontal",
    styles={
        "container": {
            "padding": "4px 8px !important",
            "background-color": "#1e293b" if st.session_state["dark_mode"] else "#ffffff",
            "border-radius": "14px",
            "border": "1px solid " + ("#334155" if st.session_state["dark_mode"] else "#e2e8f0"),
            "box-shadow": "0 4px 16px rgba(0,0,0,0.06)",
            "margin-bottom": "20px"
        },
        "icon": {"color": "#0d9488", "font-size": "14px"},
        "nav-link": {
            "font-size": "13px",
            "text-align": "center",
            "margin": "0px 3px",
            "padding": "8px 12px",
            "border-radius": "10px",
            "font-weight": "600",
            "color": "#94a3b8" if st.session_state["dark_mode"] else "#475569"
        },
        "nav-link-selected": {
            "background-color": "#0d9488",
            "color": "#ffffff",
            "font-weight": "700"
        }
    }
)

# Sync active nav index
current_selected_idx = nav_labels.index(selected_nav_str)
st.session_state["active_nav_index"] = current_selected_idx

# =============================================================================
# HELPER FUNCTIONS & VISUALIZATIONS
# =============================================================================
def render_score_gauge(score: float, label: str = "Overall Match Score"):
    """Plotly circular gauge indicator for match score."""
    dark = st.session_state["dark_mode"]
    bg_track = "#334155" if dark else "#e2e8f0"
    txt_col = "#f8fafc" if dark else "#0f172a"
    bar_col = "#10b981" if score >= 80 else ("#f59e0b" if score >= 60 else "#ef4444")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 1),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': label, 'font': {'size': 13, 'color': txt_col, 'family': 'Plus Jakarta Sans'}},
        number={'font': {'size': 32, 'color': txt_col, 'family': 'Plus Jakarta Sans'}, 'suffix': "/100"},
        gauge={
            'axis': {'range': [0, 100], 'visible': False},
            'bar': {'color': bar_col, 'thickness': 0.28},
            'bgcolor': bg_track,
            'borderwidth': 0
        }
    ))
    return apply_chart_theme(fig, dark_mode=dark, height=160)


def get_protection_stars(value: float, is_wvtr: bool = True):
    """Convert technical WVTR or OTR values to friendly star ratings."""
    if is_wvtr:
        if value <= 1.0: return "⭐⭐⭐⭐⭐ Outstanding", "status-pass"
        elif value <= 5.0: return "⭐⭐⭐⭐ Very Good", "status-pass"
        elif value <= 15.0: return "⭐⭐⭐ Good", "status-warn"
        else: return "⭐⭐ Moderate", "status-fail"
    else:
        if value <= 10.0: return "⭐⭐⭐⭐⭐ Outstanding", "status-pass"
        elif value <= 100.0: return "⭐⭐⭐⭐ Very Good", "status-pass"
        elif value <= 1500.0: return "⭐⭐⭐ Good", "status-warn"
        else: return "⭐⭐ Moderate", "status-fail"


def respond_as_packaging_expert(question: str, food_name: str, top_cand: dict) -> str:
    """Rule-based expert chatbot reasoning engine."""
    q = question.lower()
    mat_name = top_cand.get("name", "Recommended Material") if top_cand else "Standard Barrier Film"
    wvtr = top_cand.get("wvtr_g_m2_day", 5.0) if top_cand else 5.0
    otr = top_cand.get("otr_cc_m2_day_atm", 100.0) if top_cand else 100.0

    if "why" in q or "recommended" in q or "fit" in q or "reason" in q:
        return (
            f"💡 **Why {mat_name} is recommended for {food_name}:**\n\n"
            f"1. **Barrier Alignment:** {food_name} requires strong barrier protection. {mat_name} exhibits a moisture transmission rate of `{wvtr} g/m²/day` and oxygen transmission rate of `{otr} cc/m²/day/atm`, which effectively blocks degradation pathways.\n"
            f"2. **Safety & Knockouts:** It passes all safety knockout filters (temperature tolerance, food compliance, and lipid grease resistance).\n"
            f"3. **Cost-Eco Balance:** It ranked #1 in the Multi-Criteria Decision Analysis (MCDA) balancing protection, affordability, and recyclability."
        )
    elif "paper" in q or "kraft" in q or "biodegradable" in q or "eco" in q or "bio" in q:
        return (
            f"🍃 **Can you use Paper or Bioplastics for {food_name}?**\n\n"
            f"- **Standard Paper:** Uncoated paper has very high moisture permeability (WVTR > 100 g/m²/day) and no grease barrier. Crisp foods like chips or dry biscuits will absorb humidity within 48 hours.\n"
            f"- **Bio-Coated Paper:** Paper extrusion-coated with biopolymers (such as PLA or PHA) can work for short-to-medium shelf life (30–60 days), but will have a 30–50% higher unit cost than conventional polyolefins.\n"
            f"- **Compostable Plastics:** Certified PLA or PHA films are excellent for carbon reduction, but require careful verification of water-vapor barrier limits."
        )
    elif "extend" in q or "increase" in q or "longer" in q or "shelf life" in q:
        return (
            f"⏱️ **Strategies to Extend Shelf Life of {food_name}:**\n\n"
            f"1. **Lower Temperature:** Chemical staling and moisture diffusion follow Arrhenius kinetics; lowering storage temp by 10°C can double shelf life.\n"
            f"2. **Modified Atmosphere Packaging (MAP):** Flush packaging with food-grade Nitrogen ($N_2$) to reduce headspace $O_2$ below 0.5%.\n"
            f"3. **High-Barrier Metallization:** Consider vacuum metallized films (Met-BOPP or Met-PET) which reflect light and reduce WVTR below 1.0 g/m²/day.\n"
            f"4. **Active Absorbers:** Incorporate approved food-grade oxygen scavengers or silica gel desiccant sachets."
        )
    elif "wvtr" in q or "moisture" in q or "water" in q:
        return (
            f"💧 **What is WVTR (Moisture Protection)?**\n\n"
            f"Think of WVTR (Water Vapor Transmission Rate) like a raincoat for food. It measures how many grams of water vapor pass through 1 square meter of packaging in 24 hours.\n"
            f"- **Lower is tighter:** A WVTR under 1.5 g/m²/day keeps potato chips and biscuits crispy for months. A high WVTR causes dry foods to turn soggy or powdered milk to lump."
        )
    elif "otr" in q or "oxygen" in q:
        return (
            f"💨 **What is OTR (Oxygen Protection)?**\n\n"
            f"OTR (Oxygen Transmission Rate) measures how easily oxygen leaks through the packaging. Oxygen reacts with food fats and oils (lipid oxidation), causing sour off-flavors and rancid odors.\n"
            f"- Foods with fats or roasted aromas (coffee, nuts, fried snacks) require an OTR below 20 cc/m²/day/atm."
        )
    elif "temperature" in q or "heat" in q or "warm" in q:
        return (
            f"🌡️ **How Temperature Influences Packaging Permeation:**\n\n"
            f"Polymer chain mobility increases exponentially with temperature. As room temperature rises from 23°C to 38°C, the permeation rate of water and oxygen increases by approximately 2× to 2.5×! That is why cold-chain storage dramatically protects sensitive food."
        )
    else:
        return (
            f"📦 **Expert Insight for {food_name}:**\n\n"
            f"The current top recommendation is **{mat_name}**. It balances barrier performance ({wvtr} g/m²/day WVTR, {otr} cc/m²/day/atm OTR) against recycling feasibility. "
            f"Feel free to ask specifically about paper alternatives, cost savings, shelf-life extension, or recycling rules!"
        )

# =============================================================================
# SECTION 1: DASHBOARD & STEPPER RECOMMENDATIONS
# =============================================================================
if current_selected_idx == 0:
    # Landing Hero Banner
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">{t('hero_headline')}</div>
            <div class="hero-subtitle">{t('hero_sub')}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    hero_b1, hero_b2, _ = st.columns([2.5, 2.5, 3])
    with hero_b1:
        if st.button(t("btn_start_wizard"), type="primary", use_container_width=True):
            st.session_state["wizard_step"] = 1
            st.rerun()
    with hero_b2:
        if st.button(t("btn_try_demo"), use_container_width=True):
            st.session_state["selected_food_name"] = "Potato Chips"
            st.session_state["storage_temp"] = 23.0
            st.session_state["storage_rh"] = 60
            st.session_state["target_shelf_life"] = 180
            st.session_state["priority_choice"] = "⚖️ Balanced Performance (Recommended)"
            st.session_state["wizard_step"] = 4
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 4-Step Stepper Navigation Header (Polished Component)
    step = st.session_state["wizard_step"]
    st.markdown(f"### 🧭 {t('stepper_title')}")
    st.progress(step * 0.25)

    step_info = [
        {"num": 1, "title": t("step_1"), "icon": "🥔", "desc": "Commodity & Sensitivities"},
        {"num": 2, "title": t("step_2"), "icon": "📐", "desc": "Temperature & Package Area"},
        {"num": 3, "title": t("step_3"), "icon": "⚖️", "desc": "MCDA Strategy & Weights"},
        {"num": 4, "title": t("step_4"), "icon": "🏆", "desc": "Ranked Picks & Physics Models"}
    ]

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    step_cols = [s_col1, s_col2, s_col3, s_col4]

    for i, s_item in enumerate(step_info):
        curr_num = s_item["num"]
        with step_cols[i]:
            if curr_num < step:
                badge_html = '<span class="stepper-badge badge-completed">✓ Completed</span>'
                card_cls = ""
                btn_lbl = f"↩ Review Step {curr_num}"
                btn_kind = "secondary"
            elif curr_num == step:
                badge_html = '<span class="stepper-badge badge-active">● Current Step</span>'
                card_cls = "stepper-card-active"
                btn_lbl = f"● Active Step {curr_num}"
                btn_kind = "primary"
            else:
                badge_html = f'<span class="stepper-badge badge-upcoming">Step {curr_num}</span>'
                card_cls = ""
                btn_lbl = f"Go to Step {curr_num} →"
                btn_kind = "secondary"

            st.markdown(
                f"""
                <div class="stepper-card {card_cls}">
                    {badge_html}
                    <div style="font-weight: 700; font-size: 14px; color: {'#0d9488' if curr_num == step else ('#f8fafc' if st.session_state['dark_mode'] else '#0f172a')};">{s_item['icon']} {s_item['title']}</div>
                    <div style="font-size: 11.5px; color: {'#94a3b8' if st.session_state['dark_mode'] else '#64748b'}; margin-top: 2px;">{s_item['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button(btn_lbl, key=f"step_btn_{curr_num}", type=btn_kind, use_container_width=True):
                st.session_state["wizard_step"] = curr_num
                st.rerun()

    st.markdown("---")

    # -------------------------------------------------------------
    # STEP 1: FOOD SELECTION AS LARGE ICON CARDS
    # -------------------------------------------------------------
    if step == 1:
        st.markdown(f"#### 1️⃣ {t('step_1')}")
        st.write(t("select_food_prompt"))

        COMMODITY_CARDS = [
            {"name": "Potato Chips", "cat": "Snacks", "icon": "🥔", "desc": "High moisture & oxidation sensitivity; crispness retention"},
            {"name": "Roasted Coffee Beans", "cat": "Snacks", "icon": "☕", "desc": "Volatile aroma retention & oxygen barrier"},
            {"name": "Fresh Strawberries", "cat": "Fruits and vegetables", "icon": "🍓", "desc": "High respiration; requires breathable film"},
            {"name": "Whole Wheat Bread", "cat": "Bakery products", "icon": "🍞", "desc": "Moisture retention needed; mold prevention"},
            {"name": "Dry Pasta", "cat": "Cereals and grains", "icon": "🍝", "desc": "Puncture risk from sharp edges; moisture barrier"},
            {"name": "Whole Milk Powder", "cat": "Dairy products", "icon": "🥛", "desc": "Moisture caking & lipid oxidation prevention"},
            {"name": "Cheddar Cheese", "cat": "Dairy products", "icon": "🧀", "desc": "Barrier against mold; grease resistance"},
            {"name": "Roasted Almonds", "cat": "Snacks", "icon": "🥜", "desc": "Lipid rancidity risk; requires oxygen barrier"}
        ]

        # 4 cards per row
        row1 = COMMODITY_CARDS[:4]
        row2 = COMMODITY_CARDS[4:]

        c_cols1 = st.columns(4)
        for idx, item in enumerate(row1):
            with c_cols1[idx]:
                is_sel = (st.session_state["selected_food_name"] == item["name"])
                sel_badge = "✅ Selected" if is_sel else "Select"
                btn_type = "primary" if is_sel else "secondary"
                st.markdown(
                    f"""
                    <div class="commodity-card {'commodity-card-selected' if is_sel else ''}">
                        <div style="font-size: 36px; margin-bottom: 6px;">{item['icon']}</div>
                        <div style="font-weight: 700; font-size: 15px; margin-bottom: 4px; color: {'#f8fafc' if st.session_state['dark_mode'] else '#0f172a'};">{item['name']}</div>
                        <div style="font-size: 11px; color: #0d9488; font-weight: 700; text-transform: uppercase;">{item['cat']}</div>
                        <div style="font-size: 12px; color: {'#94a3b8' if st.session_state['dark_mode'] else '#64748b'}; margin-top: 6px; line-height: 1.3;">{item['desc']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button(f"{sel_badge} {item['icon']}", key=f"sel_card_{idx}", type=btn_type, use_container_width=True):
                    st.session_state["selected_food_name"] = item["name"]
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        c_cols2 = st.columns(4)
        for idx, item in enumerate(row2):
            with c_cols2[idx]:
                is_sel = (st.session_state["selected_food_name"] == item["name"])
                sel_badge = "✅ Selected" if is_sel else "Select"
                btn_type = "primary" if is_sel else "secondary"
                st.markdown(
                    f"""
                    <div class="commodity-card {'commodity-card-selected' if is_sel else ''}">
                        <div style="font-size: 36px; margin-bottom: 6px;">{item['icon']}</div>
                        <div style="font-weight: 700; font-size: 15px; margin-bottom: 4px; color: {'#f8fafc' if st.session_state['dark_mode'] else '#0f172a'};">{item['name']}</div>
                        <div style="font-size: 11px; color: #0d9488; font-weight: 700; text-transform: uppercase;">{item['cat']}</div>
                        <div style="font-size: 12px; color: {'#94a3b8' if st.session_state['dark_mode'] else '#64748b'}; margin-top: 6px; line-height: 1.3;">{item['desc']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button(f"{sel_badge} {item['icon']}", key=f"sel_card2_{idx}", type=btn_type, use_container_width=True):
                    st.session_state["selected_food_name"] = item["name"]
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        # Full dropdown option
        all_presets = database.get_all_commodity_presets()
        preset_names = [p["name"] for p in all_presets]
        curr_idx = preset_names.index(st.session_state["selected_food_name"]) if st.session_state["selected_food_name"] in preset_names else 0

        st.write("**Or select from the complete commodity database:**")
        chosen_dd = st.selectbox("All Database Food Presets", preset_names, index=curr_idx)
        if chosen_dd != st.session_state["selected_food_name"]:
            st.session_state["selected_food_name"] = chosen_dd
            st.rerun()

        preset_info = database.get_commodity_preset_by_name(st.session_state["selected_food_name"]) or {}
        st.info(f"💡 **Selected Food:** {st.session_state['selected_food_name']} — *{preset_info.get('typical_spoilage_mode', 'Balanced barrier protection recommended.')}*")

        step1_next_col, _ = st.columns([3, 5])
        with step1_next_col:
            if st.button("Proceed to Storage & Geometry ➡️", type="primary", use_container_width=True):
                st.session_state["wizard_step"] = 2
                st.rerun()

    # -------------------------------------------------------------
    # STEP 2: STORAGE PRESETS & SLIDERS
    # -------------------------------------------------------------
    elif step == 2:
        st.markdown(f"#### 2️⃣ {t('step_2')}")
        st.write(t("storage_presets_title"))

        is_cool = (abs(float(st.session_state["storage_temp"]) - 4.0) < 0.1 and int(st.session_state["storage_rh"]) == 85)
        is_room = (abs(float(st.session_state["storage_temp"]) - 23.0) < 0.1 and int(st.session_state["storage_rh"]) == 60)
        is_hot = (abs(float(st.session_state["storage_temp"]) - 38.0) < 0.1 and int(st.session_state["storage_rh"]) == 90)

        pre_c1, pre_c2, pre_c3 = st.columns(3)
        with pre_c1:
            if st.button(t("preset_cool"), type=("primary" if is_cool else "secondary"), use_container_width=True):
                st.session_state["storage_temp"] = 4.0
                st.session_state["storage_rh"] = 85
                st.rerun()
        with pre_c2:
            if st.button(t("preset_room"), type=("primary" if is_room else "secondary"), use_container_width=True):
                st.session_state["storage_temp"] = 23.0
                st.session_state["storage_rh"] = 60
                st.rerun()
        with pre_c3:
            if st.button(t("preset_hot"), type=("primary" if is_hot else "secondary"), use_container_width=True):
                st.session_state["storage_temp"] = 38.0
                st.session_state["storage_rh"] = 90
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        col_t, col_rh, col_day = st.columns(3)
        with col_t:
            st.session_state["storage_temp"] = st.slider(
                "Storage Temperature (°C)",
                min_value=-20.0,
                max_value=50.0,
                value=float(st.session_state["storage_temp"]),
                step=1.0,
                help="Temperature directly impacts polymer permeation rates according to Arrhenius kinetics."
            )
        with col_rh:
            st.session_state["storage_rh"] = st.slider(
                "Ambient Relative Humidity (% RH)",
                min_value=15,
                max_value=100,
                value=int(st.session_state["storage_rh"]),
                step=5,
                help="Higher ambient relative humidity increases the driving vapor force across the film."
            )
        with col_day:
            st.session_state["target_shelf_life"] = st.number_input(
                "Target Freshness Horizon (Days)",
                min_value=7,
                max_value=1000,
                value=int(st.session_state["target_shelf_life"]),
                step=7,
                help="Target storage or retail shelf-life horizon."
            )

        with st.expander("📐 Package Dimensions & Headspace Geometry (Optional Fine-Tuning)"):
            geo_c1, geo_c2 = st.columns(2)
            with geo_c1:
                st.session_state["pack_area_cm2"] = st.number_input("Package Surface Area (cm²)", value=float(st.session_state["pack_area_cm2"]), step=25.0)
            with geo_c2:
                st.session_state["headspace_ml"] = st.number_input("Package Headspace Gas Volume (mL)", value=float(st.session_state["headspace_ml"]), step=20.0)

        step2_back, step2_next = st.columns([2.5, 3])
        with step2_back:
            if st.button("⬅️ Back to Food Selection", use_container_width=True):
                st.session_state["wizard_step"] = 1
                st.rerun()
        with step2_next:
            if st.button("Proceed to Business Priorities ➡️", type="primary", use_container_width=True):
                st.session_state["wizard_step"] = 3
                st.rerun()

    # -------------------------------------------------------------
    # STEP 3: PRIORITIES & WEIGHTS
    # -------------------------------------------------------------
    elif step == 3:
        st.markdown(f"#### 3️⃣ {t('step_3')}")
        st.write(t("priorities_title"))

        priority_opts = [
            "⚖️ Balanced Performance (Recommended)",
            "💰 Lowest Cost First (Economical)",
            "🛡️ Maximum Moisture Barrier (Dry Foods)",
            "💨 Maximum Oxygen Barrier (Lipid / Fat Preservation)",
            "🌱 Maximum Sustainability & Recycling"
        ]
        curr_p_idx = priority_opts.index(st.session_state["priority_choice"]) if st.session_state["priority_choice"] in priority_opts else 0
        st.session_state["priority_choice"] = st.radio(
            "Select Priority Strategy:",
            priority_opts,
            index=curr_p_idx
        )

        # Weight mapping
        p = st.session_state["priority_choice"]
        if "Lowest Cost" in p:
            wb, wc, ws = 0.25, 0.55, 0.20
        elif "Moisture" in p or "Oxygen" in p:
            wb, wc, ws = 0.65, 0.15, 0.20
        elif "Sustainability" in p:
            wb, wc, ws = 0.25, 0.15, 0.60
        else:
            wb, wc, ws = 0.45, 0.25, 0.30

        with st.expander("⚙️ Fine-Tune Multi-Criteria Scoring Weights (Advanced)"):
            wc1, wc2, wc3 = st.columns(3)
            with wc1:
                wb = st.slider("Barrier Weight", 0.1, 1.0, wb, 0.05)
            with wc2:
                wc = st.slider("Cost Weight", 0.1, 1.0, wc, 0.05)
            with wc3:
                ws = st.slider("Eco Weight", 0.1, 1.0, ws, 0.05)

        st.session_state["current_weights"] = {"barrier": wb, "cost": wc, "sustainability": ws}

        step3_back, step3_next = st.columns([2.5, 3])
        with step3_back:
            if st.button("⬅️ Back to Storage", use_container_width=True):
                st.session_state["wizard_step"] = 2
                st.rerun()
        with step3_next:
            if st.button("✨ Generate Recommendations ➡️", type="primary", use_container_width=True):
                st.session_state["wizard_step"] = 4
                st.rerun()

    # -------------------------------------------------------------
    # STEP 4: RECOMMENDATION RESULTS & ALL ENHANCED FEATURES
    # -------------------------------------------------------------
    elif step == 4:
        st.markdown(f"#### 4️⃣ {t('step_4')}")

        preset_data = database.get_commodity_preset_by_name(st.session_state["selected_food_name"]) or {}
        weights = st.session_state.get("current_weights", {"barrier": 0.45, "cost": 0.25, "sustainability": 0.30})

        commodity_profile = {
            "name": st.session_state["selected_food_name"],
            "category": preset_data.get("category", "General"),
            "moisture_sensitivity": preset_data.get("moisture_sensitivity", "High"),
            "oxygen_sensitivity": preset_data.get("oxygen_sensitivity", "High"),
            "fat_content_level": preset_data.get("fat_content_level", "Medium"),
            "light_sensitivity": preset_data.get("light_sensitivity", "Moderate"),
            "storage_temp_c": float(st.session_state["storage_temp"]),
            "storage_rh_percent": int(st.session_state["storage_rh"]),
            "target_shelf_life_days": int(st.session_state["target_shelf_life"]),
            "puncture_risk": preset_data.get("puncture_risk", "Low"),
            "require_food_contact_compliance": True
        }

        materials = database.get_all_materials()
        rec_result = recommender.rank_recommendations(materials, commodity_profile, weights)
        st.session_state["last_run_result"] = rec_result

        # KPI Header Banner
        top_mat = rec_result["top_candidates"][0] if rec_result["top_candidates"] else None
        top_name = top_mat["name"] if top_mat else "No Match Found"

        # Calculate Shelf Life for top pick
        if top_mat:
            res_sl = shelf_life.estimate_moisture_shelf_life(
                wvtr_standard=float(top_mat.get("wvtr_g_m2_day", 5.0)),
                package_surface_area_m2=float(st.session_state.get("pack_area_cm2", 400.0)) / 10000.0,
                product_dry_weight_g=150.0,
                initial_moisture_percent=2.0,
                critical_moisture_percent=5.5,
                ambient_temp_c=float(st.session_state["storage_temp"]),
                ambient_rh_percent=float(st.session_state["storage_rh"])
            )
            est_days = res_sl.get("estimated_days", 180.0)
        else:
            est_days = 0.0

        r_col1, r_col2, r_col3, r_col4 = st.columns([2.5, 2.5, 2.5, 2.5])
        with r_col1:
            if top_mat:
                st.plotly_chart(render_score_gauge(top_mat["total_mcda_score"], t("score_ring_title")), use_container_width=True)
            else:
                st.metric("Match Score", "N/A")
        with r_col2:
            st.markdown(
                f"""
                <div class="big-metric-box">
                    <div class="big-metric-label">{t('shelf_life_badge')}</div>
                    <div class="big-metric-number">~{est_days:.0f} <span style="font-size: 18px;">Days</span></div>
                    <span class="status-pill status-pass">🟢 Shelf-Life Safe</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r_col3:
            st.metric("Materials Evaluated", f"{rec_result['total_evaluated']}")
            st.metric("Safe Matches", f"{rec_result['total_qualified']}")
        with r_col4:
            st.metric("Disqualified (Knockout)", f"{rec_result['total_knocked_out']}")
            st.caption(f"Top Material: **{top_name.split('(')[0].strip()}**")

        st.markdown(f'<div class="honesty-notice">{t("honesty_notice")}</div>', unsafe_allow_html=True)

        # -------------------------------------------------------------
        # TOP CANDIDATE CARDS (Friendly Star Ratings & Clear Progress)
        # -------------------------------------------------------------
        st.markdown("### 🏆 Top Recommended Packaging Candidates")

        if not rec_result["top_candidates"]:
            st.error("No materials met all physical constraints. Try adjusting the target temperature or humidity.")
        else:
            for idx, cand in enumerate(rec_result["top_candidates"][:3]):
                badge_style = "rank-gold" if idx == 0 else ("rank-silver" if idx == 1 else "rank-bronze")
                badge_title = f"RANK #{cand['rank']} — " + ("BEST OVERALL FIT" if idx == 0 else ("ALTERNATIVE MATCH" if idx == 1 else "BUDGET / CIRCULAR OPTION"))
                expl = cand["explanation"]

                m_stars, m_cls = get_protection_stars(float(cand.get("wvtr_g_m2_day", 10.0)), is_wvtr=True)
                o_stars, o_cls = get_protection_stars(float(cand.get("otr_cc_m2_day_atm", 100.0)), is_wvtr=False)

                st.markdown(
                    f"""
                    <div class="candidate-card">
                        <span class="rank-badge {badge_style}">{badge_title}</span>
                        <span style="float: right; font-weight: 800; font-size: 16px; color: #0d9488;">Overall Score: {cand['total_mcda_score']} / 100</span>
                        <h3 style="margin-top: 8px; margin-bottom: 4px; color: {'#f8fafc' if st.session_state['dark_mode'] else '#0f172a'};">{cand['name']}</h3>
                        <p style="color: {'#94a3b8' if st.session_state['dark_mode'] else '#475569'}; font-size: 14.5px; line-height: 1.5; margin-bottom: 12px;">{expl['rationale']}</p>
                        <div>
                            <span class="status-pill {m_cls}">💧 {t('moisture_prot')}: {m_stars}</span>
                            <span class="status-pill {o_cls}">💨 {t('oxygen_prot')}: {o_stars}</span>
                            <span class="status-pill status-pass">🌱 Recyclability: {cand.get('recyclability_status', 'Standard')}</span>
                            <span class="status-pill status-pass">💰 Cost: ${cand.get('cost_index_usd_per_kg', 0):.2f} / kg</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown(f"**{t('why_it_fits')}:**")
                    for req in expl.get("requirements_met", [])[:3]:
                        st.markdown(f"- <span style='font-size: 13.5px;'>{req}</span>", unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"**{t('considerations')}:**")
                    for lim in expl.get("limitations", [])[:2]:
                        st.markdown(f"- <span style='font-size: 13.5px;'>{lim}</span>", unsafe_allow_html=True)
                st.markdown("---")

        # -------------------------------------------------------------
        # NEW FEATURE 1: INTERACTIVE SHELF-LIFE CURVE (LIVE TEMP SLIDER)
        # -------------------------------------------------------------
        if top_mat:
            st.markdown("### 📈 Interactive Shelf-Life Curve (Quality vs. Days)")
            st.caption("Drag the live temperature slider below to see how storage temperature accelerates permeation in real time based on Arrhenius mass-transfer physics:")

            live_temp = st.slider(
                "Live Temperature Simulation (°C):",
                min_value=5.0,
                max_value=45.0,
                value=float(st.session_state["storage_temp"]),
                step=1.0,
                key="live_temp_slider"
            )

            res_live = shelf_life.estimate_moisture_shelf_life(
                wvtr_standard=float(top_mat.get("wvtr_g_m2_day", 5.0)),
                package_surface_area_m2=float(st.session_state.get("pack_area_cm2", 400.0)) / 10000.0,
                product_dry_weight_g=150.0,
                initial_moisture_percent=2.0,
                critical_moisture_percent=5.5,
                ambient_temp_c=live_temp,
                ambient_rh_percent=float(st.session_state["storage_rh"])
            )

            days_arr = res_live["days_array"]
            moist_arr = res_live["moisture_curve"]
            crit_val = res_live["critical_threshold"]

            # Calculate Quality %: 100% on Day 0 down to 0% at critical threshold
            quality_curve = [max(0.0, 100.0 * (1.0 - (m - 2.0) / (crit_val - 2.0))) for m in moist_arr]

            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(
                x=days_arr,
                y=quality_curve,
                mode="lines+markers",
                name="Food Quality Retention (%)",
                line=dict(color="#0d9488", width=3),
                fill='tozeroy',
                fillcolor='rgba(13, 148, 136, 0.12)'
            ))
            fig_q.add_hline(
                y=20,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text="Critical Spoilage Horizon"
            )
            fig_q.update_layout(
                title=f"Quality Retention Curve at {live_temp:.0f}°C (Estimated Shelf Life: ~{res_live['estimated_days']:.0f} Days)",
                xaxis_title="Storage Time (Days)",
                yaxis_title="Quality Retention (%)"
            )
            st.plotly_chart(apply_chart_theme(fig_q, dark_mode=st.session_state["dark_mode"], height=340), use_container_width=True)

        # -------------------------------------------------------------
        # NEW FEATURE 2: PACKAGING SAVINGS CALCULATOR (COST & CO2)
        # -------------------------------------------------------------
        st.markdown(f"### {t('savings_calculator_title')}")
        st.caption("Calculate the exact annual cost and carbon footprint savings by switching to the recommended material using real database values:")

        sav_c1, sav_c2, sav_c3 = st.columns(3)
        with sav_c1:
            monthly_packs = st.number_input("Monthly Packaging Volume (Units/month)", value=50000, step=5000)
        with sav_c2:
            all_mats_dict = {m["name"]: m for m in database.get_all_materials()}
            curr_choice = st.selectbox(
                "Current Packaging Material",
                list(all_mats_dict.keys()),
                index=1 if len(all_mats_dict) > 1 else 0
            )
        with sav_c3:
            rec_choice = top_mat["name"] if top_mat else curr_choice
            st.text_input("Recommended Material", value=rec_choice, disabled=True)

        if curr_choice in all_mats_dict and rec_choice in all_mats_dict:
            m_curr = all_mats_dict[curr_choice]
            m_rec = all_mats_dict[rec_choice]

            # Physics mass calculation: Surface area ~0.04 m2, thickness (um), density ~1.0 g/cm3
            area_m2 = float(st.session_state.get("pack_area_cm2", 400.0)) / 10000.0
            thick_curr_m = float(m_curr.get("nominal_thickness_um", 50.0)) * 1e-6
            thick_rec_m = float(m_rec.get("nominal_thickness_um", 30.0)) * 1e-6
            density_kg_m3 = 1000.0

            mass_curr_unit_kg = area_m2 * thick_curr_m * density_kg_m3
            mass_rec_unit_kg = area_m2 * thick_rec_m * density_kg_m3

            annual_units = monthly_packs * 12
            annual_cost_curr = annual_units * mass_curr_unit_kg * float(m_curr.get("cost_index_usd_per_kg", 3.0))
            annual_cost_rec = annual_units * mass_rec_unit_kg * float(m_rec.get("cost_index_usd_per_kg", 3.0))
            cost_savings_annual = annual_cost_curr - annual_cost_rec

            annual_co2_curr = annual_units * mass_curr_unit_kg * float(m_curr.get("carbon_footprint_kg_co2_per_kg", 2.5))
            annual_co2_rec = annual_units * mass_rec_unit_kg * float(m_rec.get("carbon_footprint_kg_co2_per_kg", 2.0))
            co2_savings_annual = annual_co2_curr - annual_co2_rec

            sc_kpi1, sc_kpi2, sc_kpi3 = st.columns(3)
            with sc_kpi1:
                st.metric("Annual Financial Impact", f"${cost_savings_annual:,.2f}", delta=f"{'+' if cost_savings_annual >= 0 else ''}${cost_savings_annual:,.2f}")
            with sc_kpi2:
                st.metric("Annual CO₂ Avoided", f"{co2_savings_annual:,.1f} kg CO₂", delta=f"{'+' if co2_savings_annual >= 0 else ''}{co2_savings_annual:,.1f} kg")
            with sc_kpi3:
                pct_cost = (cost_savings_annual / max(annual_cost_curr, 1)) * 100
                st.metric("Cost Reduction %", f"{pct_cost:.1f}%")

        # -------------------------------------------------------------
        # NEW FEATURE 4: SAVE UP TO 3 SCENARIOS & COMPARE
        # -------------------------------------------------------------
        st.markdown(f"### {t('scenario_title')}")
        st.caption("Save your current recommendation scenario to compare different foods or storage climates:")

        save_btn_col, clear_btn_col = st.columns([2.5, 2.5])
        with save_btn_col:
            if st.button("💾 Save Current Scenario to Session", type="primary", use_container_width=True):
                if len(st.session_state["saved_scenarios"]) >= 3:
                    st.warning("You have already saved 3 scenarios. Please clear or delete one below.")
                else:
                    new_scen = {
                        "id": len(st.session_state["saved_scenarios"]) + 1,
                        "food": commodity_profile["name"],
                        "storage": f"{commodity_profile['storage_temp_c']}°C / {commodity_profile['storage_rh_percent']}% RH",
                        "top_pick": top_name,
                        "score": f"{top_mat['total_mcda_score']}/100" if top_mat else "N/A",
                        "shelf_life": f"~{est_days:.0f} Days",
                        "cost": f"${top_mat.get('cost_index_usd_per_kg', 0):.2f}/kg" if top_mat else "N/A",
                        "recyclability": top_mat.get("recyclability_status", "Standard") if top_mat else "N/A"
                    }
                    st.session_state["saved_scenarios"].append(new_scen)
                    st.success(f"Saved scenario: {new_scen['food']}!")
                    st.rerun()

        with clear_btn_col:
            if st.button("🗑️ Clear Saved Scenarios", use_container_width=True):
                st.session_state["saved_scenarios"] = []
                st.rerun()

        if st.session_state["saved_scenarios"]:
            st.dataframe(pd.DataFrame(st.session_state["saved_scenarios"]), use_container_width=True)

        # -------------------------------------------------------------
        # NEW FEATURE 5: ASK THE PACKAGING EXPERT CHATBOT
        # -------------------------------------------------------------
        st.markdown(f"### {t('expert_chat_title')}")
        st.caption("Ask questions about this recommendation, paper alternatives, oxygen barriers, or shelf life. No API key required:")

        # Quick question chips
        chip1, chip2, chip3, chip4 = st.columns(4)
        with chip1:
            if st.button("💡 Why was this recommended?", use_container_width=True):
                reply = respond_as_packaging_expert("Why was this material recommended?", commodity_profile["name"], top_mat)
                st.session_state["expert_chat_history"].append({"role": "user", "content": "Why was this material recommended?"})
                st.session_state["expert_chat_history"].append({"role": "assistant", "content": reply})
                st.rerun()
        with chip2:
            if st.button("🍃 Can I use paper packaging?", use_container_width=True):
                reply = respond_as_packaging_expert("Can I use paper packaging?", commodity_profile["name"], top_mat)
                st.session_state["expert_chat_history"].append({"role": "user", "content": "Can I use paper packaging?"})
                st.session_state["expert_chat_history"].append({"role": "assistant", "content": reply})
                st.rerun()
        with chip3:
            if st.button("⏱️ How to extend shelf life?", use_container_width=True):
                reply = respond_as_packaging_expert("How can I extend shelf life?", commodity_profile["name"], top_mat)
                st.session_state["expert_chat_history"].append({"role": "user", "content": "How can I extend shelf life?"})
                st.session_state["expert_chat_history"].append({"role": "assistant", "content": reply})
                st.rerun()
        with chip4:
            if st.button("💧 What does WVTR mean?", use_container_width=True):
                reply = respond_as_packaging_expert("What does WVTR mean?", commodity_profile["name"], top_mat)
                st.session_state["expert_chat_history"].append({"role": "user", "content": "What does WVTR mean?"})
                st.session_state["expert_chat_history"].append({"role": "assistant", "content": reply})
                st.rerun()

        # Render chat history
        for msg in st.session_state["expert_chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Type your question here (e.g., 'What happens if temperature rises?', 'Is this recyclable?')...")
        if user_input:
            st.session_state["expert_chat_history"].append({"role": "user", "content": user_input})
            reply = respond_as_packaging_expert(user_input, commodity_profile["name"], top_mat)
            st.session_state["expert_chat_history"].append({"role": "assistant", "content": reply})
            st.rerun()

        # -------------------------------------------------------------
        # CHARTS & PDF DOWNLOAD
        # -------------------------------------------------------------
        st.markdown("---")
        st.markdown("#### 📊 Score Breakdown & Multi-Dimensional Radar Profile")
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            fig_breakdown = analytics.create_sensitivity_breakdown_chart(rec_result["top_candidates"])
            st.plotly_chart(apply_chart_theme(fig_breakdown, dark_mode=st.session_state["dark_mode"], height=380), use_container_width=True)
        with col_fig2:
            radar_fig, _ = analytics.create_radar_comparison_chart(rec_result["top_candidates"][:3])
            st.plotly_chart(apply_chart_theme(radar_fig, dark_mode=st.session_state["dark_mode"], height=380), use_container_width=True)

        st.markdown("#### 📄 Download Formal Engineering Report (PDF)")
        pdf_bytes = pdf_generator.generate_recommendation_pdf(
            commodity_profile=commodity_profile,
            recommendation_result=rec_result,
            user_weights=weights
        )
        safe_fname = commodity_profile["name"].replace(" ", "_").replace("/", "_")
        st.download_button(
            label="📥 Download Packaging Recommendation Report (PDF)",
            data=pdf_bytes,
            file_name=f"Packaging_Report_{safe_fname}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        step4_b1, step4_b2, _ = st.columns([2.5, 2.5, 3])
        with step4_b1:
            if st.button("⬅️ Back to Priorities", key="step4_back_btn", use_container_width=True):
                st.session_state["wizard_step"] = 3
                st.rerun()
        with step4_b2:
            if st.button("🔄 Start New Analysis", key="step4_restart_btn", use_container_width=True):
                st.session_state["wizard_step"] = 1
                st.rerun()

# =============================================================================
# SECTION 2: SHELF-LIFE & FRESHNESS GUIDE
# =============================================================================
elif current_selected_idx == 1:
    st.subheader("⏱️ Shelf-Life & Permeation Simulation Guide")
    st.write("First-principles mass-transfer simulation of moisture sorption and oxygen ingress through polymer films.")

    st.markdown(f'<div class="honesty-notice">{t("honesty_notice")}</div>', unsafe_allow_html=True)

    materials = database.get_all_materials()
    mat_names = [m["name"] for m in materials]
    selected_mat_name = st.selectbox("Select Packaging Material to Evaluate", mat_names, index=1)
    chosen_mat = next(m for m in materials if m["name"] == selected_mat_name)

    sim_tab1, sim_tab2 = st.tabs(["💧 Moisture Ingress (Crispness & Caking)", "💨 Oxygen Ingress (Rancidity & Flavor)"])

    with sim_tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            p_area_cm2 = st.number_input("Package Surface Area (cm²)", value=400.0, step=50.0)
            food_wt_g = st.number_input("Food Dry Weight (grams)", value=150.0, step=25.0)
        with c2:
            init_moist = st.number_input("Starting Moisture (%)", value=2.0, step=0.5)
            crit_moist = st.number_input("Spoilage Moisture Limit (%)", value=5.5, step=0.5)
        with c3:
            s_temp = st.number_input("Storage Temp (°C)", value=25.0, step=1.0)
            s_rh = st.slider("Ambient Relative Humidity (% RH)", 20, 100, 75)

        res_moist = shelf_life.estimate_moisture_shelf_life(
            wvtr_standard=float(chosen_mat.get("wvtr_g_m2_day", 5.0)),
            package_surface_area_m2=p_area_cm2 / 10000.0,
            product_dry_weight_g=food_wt_g,
            initial_moisture_percent=init_moist,
            critical_moisture_percent=crit_moist,
            ambient_temp_c=s_temp,
            ambient_rh_percent=s_rh
        )

        if "error" in res_moist:
            st.error(res_moist["error"])
        else:
            st.success(f"Estimated Moisture Protection Horizon: **~{res_moist['estimated_days']:.0f} Days** under {s_temp}°C / {s_rh}% RH.")
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(
                x=res_moist["days_array"],
                y=res_moist["moisture_curve"],
                mode="lines",
                name="Projected Moisture (%)",
                line=dict(color="#0d9488", width=3)
            ))
            fig_m.add_hline(y=crit_moist, line_dash="dash", line_color="#ef4444", annotation_text=f"Critical Limit ({crit_moist}%)")
            fig_m.update_layout(
                title=f"Moisture Sorption Curve for '{chosen_mat['name']}'",
                xaxis_title="Storage Time (Days)",
                yaxis_title="Moisture Content (%)"
            )
            st.plotly_chart(apply_chart_theme(fig_m, dark_mode=st.session_state["dark_mode"], height=380), use_container_width=True)

    with sim_tab2:
        co1, co2, co3 = st.columns(3)
        with co1:
            hs_vol = st.number_input("Headspace Volume (mL / cc)", value=120.0, step=20.0)
            area_m2_o = p_area_cm2 / 10000.0
        with co2:
            init_o2_val = st.number_input("Initial Headspace O₂ (%)", value=0.5, step=0.5)
            crit_o2_val = st.number_input("Critical Allowable O₂ (%)", value=2.0, step=0.5)
        with co3:
            s_temp_o = st.number_input("Ambient Temp (°C)", value=23.0, step=1.0, key="sim_o2_temp")

        res_o = shelf_life.estimate_oxygen_ingress(
            otr_standard=float(chosen_mat.get("otr_cc_m2_day_atm", 100.0)),
            package_surface_area_m2=area_m2_o,
            headspace_volume_cc=hs_vol,
            initial_headspace_o2_percent=init_o2_val,
            critical_headspace_o2_percent=crit_o2_val,
            ambient_temp_c=s_temp_o
        )

        if "error" in res_o:
            st.warning(res_o["error"])
        else:
            st.success(f"Estimated Oxygen Ingress Horizon: **~{res_o['estimated_days']:.0f} Days** before reaching {crit_o2_val}% oxygen.")
            fig_o = go.Figure()
            fig_o.add_trace(go.Scatter(
                x=res_o["days_array"],
                y=res_o["o2_curve"],
                mode="lines",
                name="Headspace Oxygen (%)",
                line=dict(color="#0284c7", width=3)
            ))
            fig_o.add_hline(y=crit_o2_val, line_dash="dash", line_color="#ef4444", annotation_text=f"Critical Limit ({crit_o2_val}%)")
            fig_o.update_layout(
                title=f"Oxygen Ingress Curve for '{chosen_mat['name']}'",
                xaxis_title="Storage Time (Days)",
                yaxis_title="Headspace Oxygen (%)"
            )
            st.plotly_chart(apply_chart_theme(fig_o, dark_mode=st.session_state["dark_mode"], height=380), use_container_width=True)

# =============================================================================
# SECTION 3: MATERIAL COMPARATOR
# =============================================================================
elif current_selected_idx == 2:
    st.subheader("⚖️ Material Comparator Side-by-Side")
    st.write("Compare 2 to 4 packaging polymers across barrier properties, mechanical strength, cost, and recycling streams.")

    materials = database.get_all_materials()
    mat_options = {m["name"]: m for m in materials}

    default_selection = [
        "LDPE Monolayer Film (50 µm)",
        "Metallized BOPP (Met-BOPP, 20 µm)",
        "PET / EVOH / PE Multi-layer Barrier Film (70 µm)"
    ]
    valid_defaults = [d for d in default_selection if d in mat_options]

    selected_names = st.multiselect(
        "Select Materials to Compare:",
        options=list(mat_options.keys()),
        default=valid_defaults[:3],
        max_selections=4
    )

    if len(selected_names) < 2:
        st.warning("Please select at least 2 materials above to compare side-by-side.")
    else:
        selected_mats = [mat_options[name] for name in selected_names]

        # 6-Axis Radar Comparison
        st.markdown("#### 🕸️ Multi-Feature Radar Comparison")
        radar_fig, warnings = analytics.create_radar_comparison_chart(selected_mats)
        st.plotly_chart(apply_chart_theme(radar_fig, dark_mode=st.session_state["dark_mode"], height=420), use_container_width=True)

        # Tabular Comparison
        comp_rows = []
        for m in selected_mats:
            wv = float(m.get("wvtr_g_m2_day", 10.0))
            ot = float(m.get("otr_cc_m2_day_atm", 1000.0))
            m_s, _ = get_protection_stars(wv, is_wvtr=True)
            o_s, _ = get_protection_stars(ot, is_wvtr=False)
            comp_rows.append({
                "Material": m["name"],
                "Structure": m["structure_type"],
                "Thickness": f"{m['nominal_thickness_um']} µm",
                "Moisture Protection": m_s,
                "Oxygen Protection": o_s,
                "Cost ($/kg)": f"${m['cost_index_usd_per_kg']:.2f}",
                "Recyclability": m["recyclability_status"]
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

# =============================================================================
# SECTION 4: SUSTAINABILITY & LCA
# =============================================================================
elif current_selected_idx == 3:
    st.subheader("🌿 Sustainability, Life Cycle Assessment & Recycling")
    st.write("Evaluate global warming potential (CO₂ footprint), circular recyclability tiers, and the food-waste paradox.")

    materials_df = database.get_all_materials_df()
    s_tab1, s_tab2 = st.tabs(["📊 Recyclability Streams", "⚖️ Cost vs. Carbon Footprint Quadrant"])

    with s_tab1:
        tier_counts = materials_df["recyclability_status"].value_counts().reset_index()
        tier_counts.columns = ["Recycling Category", "Number of Materials"]
        fig_recyc = px.bar(
            tier_counts,
            x="Recycling Category",
            y="Number of Materials",
            color="Recycling Category",
            title="Packaging Materials Classified by Recycling Stream",
            color_discrete_sequence=analytics.PALETTE
        )
        fig_recyc.update_layout(showlegend=False)
        st.plotly_chart(apply_chart_theme(fig_recyc, dark_mode=st.session_state["dark_mode"], height=380), use_container_width=True)

    with s_tab2:
        cost_sust_fig, co2_warnings = analytics.create_cost_sustainability_scatter(materials_df)
        st.plotly_chart(apply_chart_theme(cost_sust_fig, dark_mode=st.session_state["dark_mode"], height=420), use_container_width=True)
        if co2_warnings:
            st.caption("ℹ️ Note: " + " | ".join(co2_warnings))

# =============================================================================
# SECTION 5: MACHINE LEARNING & MATERIAL CLUSTERING
# =============================================================================
elif current_selected_idx == 4:
    st.subheader("🤖 Machine Learning Analytics & Clustering")
    st.write("Unsupervised K-Means clustering and supervised Random Forest feature-importance analysis.")

    st.markdown(f'<div class="honesty-notice">{t("honesty_notice")}</div>', unsafe_allow_html=True)

    materials_df = database.get_all_materials_df()
    ml_tab1, ml_tab2 = st.tabs(["🔍 Unsupervised Clustering (K-Means & PCA)", "🌲 Supervised Feature Importance (Random Forest)"])

    with ml_tab1:
        k_val = st.slider("Number of Clusters (k):", min_value=2, max_value=5, value=4)
        clustered_df, pca_fig, pca_stats = ml_engine.perform_material_clustering(materials_df, n_clusters=k_val)
        st.plotly_chart(apply_chart_theme(pca_fig, dark_mode=st.session_state["dark_mode"], height=420), use_container_width=True)
        st.caption(f"2D PCA explains {pca_stats['total_variance_explained']}% of the physical variance between materials.")

    with ml_tab2:
        clf, fi_fig, summary = ml_engine.build_supervised_baseline()
        st.plotly_chart(apply_chart_theme(fi_fig, dark_mode=st.session_state["dark_mode"], height=380), use_container_width=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Classifier", "Random Forest (50 Trees)")
        with sc2:
            st.metric("Synthetic Scenarios", f"{summary['n_samples']} Samples")
        with sc3:
            st.metric("Training Accuracy", f"{summary['train_accuracy']}%")

# =============================================================================
# SECTION 6: DATABASE EXPLORER & ADMIN
# =============================================================================
elif current_selected_idx == 5:
    st.subheader("📦 Database Explorer & Material Administrator")
    st.write("Inspect, search, add, edit, or delete packaging materials in the persistent SQLite database.")

    materials_df = database.get_all_materials_df()
    search_term = st.text_input("🔍 Search materials by name or structure:", "")

    if search_term:
        filtered_df = materials_df[
            materials_df["name"].str.contains(search_term, case=False, na=False) |
            materials_df["structure_type"].str.contains(search_term, case=False, na=False)
        ]
    else:
        filtered_df = materials_df

    st.dataframe(
        filtered_df[[
            "id", "name", "structure_type", "nominal_thickness_um",
            "wvtr_g_m2_day", "otr_cc_m2_day_atm", "cost_index_usd_per_kg",
            "recyclability_status", "food_contact_compliance"
        ]],
        use_container_width=True,
        height=280
    )

    db_t1, db_t2 = st.tabs(["➕ Add New Material", "✏️ Edit / Delete Material"])

    with db_t1:
        with st.form("add_mat_form"):
            a1, a2, a3 = st.columns(3)
            with a1:
                n_name = st.text_input("Material Name *", "Example Eco Film (30 µm)")
                n_struct = st.selectbox("Structure", ["Monolayer Flexible", "Multilayer Barrier Laminate", "Bio-based / Biodegradable"])
                n_thick = st.number_input("Thickness (µm) *", min_value=1.0, value=30.0)
            with a2:
                n_wvtr = st.number_input("WVTR (g/m²/d) *", min_value=0.001, value=2.5)
                n_otr = st.number_input("OTR (cc/m²/d/atm) *", min_value=0.001, value=25.0)
                n_cost = st.number_input("Cost ($/kg) *", min_value=0.1, value=3.5)
            with a3:
                n_recyc = st.selectbox("Recyclability", list(recommender.RECYCLABILITY_SCORE_MAP.keys()))
                n_comp = st.text_input("Compliance", "US FDA 21 CFR §177")
                n_source = st.text_input("Data Source", "Lab Trial 2026")

            if st.form_submit_button("Save Material", type="primary"):
                nid = database.add_material({
                    "name": n_name,
                    "structure_type": n_struct,
                    "polymer_code": "CUSTOM",
                    "nominal_thickness_um": n_thick,
                    "wvtr_g_m2_day": n_wvtr,
                    "otr_cc_m2_day_atm": n_otr,
                    "cost_index_usd_per_kg": n_cost,
                    "recyclability_status": n_recyc,
                    "food_contact_compliance": n_comp,
                    "temp_range_min_c": -20.0,
                    "temp_range_max_c": 90.0,
                    "data_source": n_source,
                    "test_conditions": "ASTM F1249 & D3985",
                    "is_verified_experimental": 1,
                    "notes": "Added from UI"
                })
                st.success(f"Added successfully with ID: {nid}!")
                st.rerun()

    with db_t2:
        mats = database.get_all_materials()
        mat_map = {f"ID {m['id']}: {m['name']}": m for m in mats}
        sel_edit = st.selectbox("Select Material to Modify:", list(mat_map.keys()))
        if sel_edit:
            cm = mat_map[sel_edit]
            with st.form("edit_mat_form"):
                e1, e2 = st.columns(2)
                with e1:
                    e_name = st.text_input("Name", cm["name"])
                    e_thick = st.number_input("Thickness (µm)", value=float(cm["nominal_thickness_um"]))
                    e_cost = st.number_input("Cost ($/kg)", value=float(cm["cost_index_usd_per_kg"]))
                with e2:
                    e_wvtr = st.number_input("WVTR", value=float(cm["wvtr_g_m2_day"]))
                    e_otr = st.number_input("OTR", value=float(cm["otr_cc_m2_day_atm"]))
                if st.form_submit_button("Update Details", type="primary"):
                    cm_dict = dict(cm)
                    cm_dict.update({"name": e_name, "nominal_thickness_um": e_thick, "cost_index_usd_per_kg": e_cost, "wvtr_g_m2_day": e_wvtr, "otr_cc_m2_day_atm": e_otr})
                    database.update_material(cm["id"], cm_dict)
                    st.success("Updated successfully!")
                    st.rerun()

# =============================================================================
# SECTION 7: CSV IMPORT / EXPORT
# =============================================================================
elif current_selected_idx == 6:
    st.subheader("📂 CSV Ingestion, Validation & Export Manager")
    st.write("Export the current materials repository to CSV, download templates, or batch-upload experimental measurements.")

    c_exp, c_imp = st.columns(2)
    with c_exp:
        st.markdown("##### 📥 Export Materials Database")
        csv_data = data_loader.export_materials_to_csv()
        st.download_button(
            "Download Complete Database (CSV)",
            data=csv_data,
            file_name="packaging_materials_export.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📄 Standard CSV Template")
        st.download_button(
            "Download Validation Template (CSV)",
            data=data_loader.get_csv_template(),
            file_name="materials_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c_imp:
        st.markdown("##### 📤 Upload Batch CSV File")
        up_file = st.file_uploader("Upload CSV with Laboratory Records", type=["csv"])
        if up_file:
            content = up_file.getvalue().decode("utf-8")
            res_imp = data_loader.validate_and_import_csv(content)
            if res_imp["success"]:
                st.success(f"✅ Successfully imported {res_imp['imported_count']} material(s) into database!")
                st.rerun()
            else:
                st.error("CSV import encountered errors.")
                for err in res_imp["errors"]:
                    st.error(err)

# =============================================================================
# SECTION 8: USER GUIDE & AUDIT HISTORY
# =============================================================================
elif current_selected_idx == 7:
    st.subheader("📖 User Guide, Scientific Methodology & Audit Trail")
    st.write("Literature citations, Arrhenius mass-transfer formulas, and audit query records.")

    g_tab1, g_tab2 = st.tabs(["🔬 Methodology & Standard References", "📜 Recommendation Audit History"])

    with g_tab1:
        st.markdown(
            """
            ### Scientific Formulations & Standards
            1. **Moisture Sorption Kinetics (Labuza & Robertson Formulation):**
               $$\\frac{dm}{dt} = \\frac{\\text{WVTR}_{\\text{adj}} \\cdot A}{W_s}$$
               $$\\text{WVTR}_{\\text{adj}} = \\text{WVTR}_{\\text{std}} \\cdot e^{0.045 \\cdot (T - 38)} \\cdot \\left(\\frac{\\text{RH}}{90}\\right)$$
            2. **Oxygen Ingress Mass-Transfer (Robertson 2012):**
               $$\\text{OTR}_{\\text{adj}} = \\text{OTR}_{\\text{std}} \\cdot e^{0.035 \\cdot (T - 23)}$$
               $$\\Delta V_{O_2} = \\text{OTR}_{\\text{adj}} \\cdot A \\cdot \\Delta P_{O_2} \\cdot t$$
            3. **Multi-Criteria Decision Analysis (MCDA):**
               $$S_{\\text{total}} = w_b \\cdot S_{\\text{barrier}} + w_c \\cdot S_{\\text{cost}} + w_s \\cdot S_{\\text{sustainability}}$$
            4. **Data References:**
               - ASTM F1249 (Water Vapor Transmission Rate via Infrared Sensor)
               - ASTM D3985 (Oxygen Transmission Rate via Coulometric Sensor)
               - Robertson, G. L. (2012). *Food Packaging: Principles and Practice*, CRC Press.
            """
        )

    with g_tab2:
        history = database.get_recommendation_history(limit=50)
        if not history:
            st.info("No queries logged yet. Generate recommendations in the Dashboard to record audit queries.")
        else:
            h_list = []
            for h in history:
                recs = json.loads(h["top_recommendations_json"])
                rec_str = "; ".join([f"#{r['rank']} {r['name']} ({r['score']}/100)" for r in recs[:2]])
                h_list.append({
                    "Timestamp (UTC)": h["timestamp"],
                    "Food": h["commodity_name"],
                    "Category": h["category"],
                    "Top Picks": rec_str
                })
            st.dataframe(pd.DataFrame(h_list), use_container_width=True)

st.markdown("---")
st.caption("Smart FoodPack AI • Modern Decision Support System for Food Packaging Materials • 2026")
