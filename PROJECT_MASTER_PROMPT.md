# MASTER PROMPT: Smart FoodPack AI — Intelligent Food Packaging Material Recommendation System

> **Role & Objective for the AI Assistant:**
> You are an expert Food Packaging Scientist, Senior Full-Stack Python Engineer, and Streamlit/Data Science Specialist. This document is the definitive technical master prompt and implementation manual for **Smart FoodPack AI (AI-Based Intelligent Food Packaging Material Recommendation System for Food Commodities)**.
> Use this document to understand, recreate, maintain, debug, test, extend, or refactor the application without losing any domain accuracy, architectural integrity, or UI quality.

---

## 1. Project Overview

### 1.1 Problem Statement
Selecting optimal food packaging is a complex multi-parameter engineering challenge at the intersection of food science, materials chemistry, supply chain logistics, economics, and environmental sustainability. Mismatched packaging leads to:
1. **Premature food spoilage** and microbial decay caused by excessive moisture gain/loss or oxygen permeation.
2. **Economic losses** from over-packaging (e.g., using expensive multi-layer high-barrier films on low-risk dry grains) or under-packaging (recalls and food waste).
3. **Severe environmental impacts** from single-use unrecyclable plastics and unquantified carbon footprints.
4. **Knowledge barrier** for small-to-medium food enterprises (SMEs), farmers, culinary startups, and ordinary packaging technicians who lack access to specialized barrier testing laboratories or expensive packaging consultants.

### 1.2 Objectives
The system serves as an intelligent decision-support platform designed to:
- Translate raw food commodity sensitivities (moisture, oxygen, respiration, light, fats, physical punctures) into rigorous engineering barrier thresholds.
- Filter and score commercially available packaging polymers and sustainable biopolymers using transparent, multi-criteria decision algorithms.
- Simulate package shelf-life using established first-principles mass transfer permeation equations.
- Perform environmental life-cycle and economic cost trade-off assessments.
- Provide accessible, educational visualizations and jargon-free explanations for non-technical users while preserving rigorous technical data for packaging engineers.
- Offer a seamless, dual-theme (Light & Dark Mode) and bilingual (English & Hindi) user experience.
- Generate exportable, audit-ready PDF packaging qualification reports and facilitate bidirectional CSV data management.

### 1.3 Target Audience & Practical Use Cases
- **Food Entrepreneurs & Small Businesses:** Rapidly identify compliant, cost-effective packaging for bakery, snack, coffee, spice, or fresh produce products without hiring dedicated packaging consultants.
- **Packaging Engineers & Quality Assurance:** Evaluate material alternatives (e.g., replacing BOPP/Al-foil laminates with bio-based PLA or high-barrier EVOH) and compare shelf-life impacts under variable ambient storage conditions ($T$, $\text{RH}\%$).
- **Academic Researchers & Food Technology Students:** Study permeation kinetics, polymer property trade-offs, and multi-criteria optimization algorithms with verifiable literature benchmarks.

---

## 2. Technology Stack

| Layer / Domain | Technology | Version / Spec | Justification & Role in Project |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `>=3.10, <3.15` (Verified on 3.14) | Core programming language for logic, scientific calculation, and data pipelines. |
| **Web UI Framework** | Streamlit | `>=1.35.0` | Reactive multi-page web dashboard, interactive forms, dynamic widgets, and state management. |
| **Top Navigation Bar** | streamlit-option-menu | `>=0.4.0` | Clean horizontal top navigation bar with custom Bootstrap icons and labels. |
| **Relational Database** | SQLite (`sqlite3`) | Standard Library (v3) | Serverless, zero-configuration embedded relational database storing packaging materials, food presets, and audit logs. |
| **Data Manipulation** | pandas | `>=2.0.0` | Tabular data processing, DataFrame transformations, CSV import/export parsing, and statistical aggregations. |
| **Numerical Computing** | NumPy & SciPy | `numpy>=1.24.0`, `scipy>=1.10.0` | Vectorized math, exponential Arrhenius calculations, min-max scaling, and mathematical boundary guards. |
| **Interactive Visualizations** | Plotly (`plotly.graph_objects`, `plotly.express`) | `>=5.20.0` | Publication-grade interactive charts: 6-axis Radar charts, Log-log Barrier Maps, Cost vs. Carbon bubble charts, and MCDA score breakdowns. |
| **Machine Learning** | scikit-learn | `>=1.4.0` | Unsupervised `KMeans` clustering, `StandardScaler`, and `PCA` (2D dimensionality reduction); supervised `RandomForestClassifier` baseline for feature importance demonstrations. |
| **PDF Generation Engine** | ReportLab | `>=4.0.0` | Programmatic vector PDF compilation with two-pass `NumberedCanvas` ("Page X of Y"), customized flowable tables, and executive summary formatting. |
| **Testing Suite** | pytest | `>=8.0.0` | Automated unit, regression, and integration testing for database integrity, recommendation logic, shelf-life simulation, and ML pipelines. |
| **Theme System & Design** | CSS Variables + Streamlit Theming | Custom Inject (`.streamlit/config.toml` + Injected CSS) | High-contrast Light Mode and Dark Mode with centralized CSS variables, Google Font *Plus Jakarta Sans*, rounded cards, and responsive grids. |

---

## 3. Complete Feature List & User Workflows

### 3.1 Navigation & Architecture
The application runs as a single-entry Streamlit application (`app.py`) featuring 8 main sections selectable via a horizontal top navigation bar (`streamlit-option-menu`):

1. **🌟 Dashboard & Recommendations:** The central interactive workflow. Guides users through a 4-step wizard:
   - *Step 1: Select or Customize Food Commodity* (8 large icon cards, full dropdown of 12 verified presets, and custom sensitivity tuning).
   - *Step 2: Package Geometry & Storage Conditions* (Quick-preset buttons for Cool 4°C/85% RH, Room 23°C/60% RH, Hot-Humid 38°C/90% RH; custom sliders for Temp, RH, Horizon, Area, Headspace).
   - *Step 3: Business & Environmental Priorities* (Strategy selection for Balanced, Lowest Cost, Max Moisture, Max Oxygen, Max Sustainability + fine-tuning sliders).
   - *Step 4: Real-Time Recommendations & Candidate Cards* (Score ring gauge, large predicted shelf-life number, top candidate cards with star ratings, interactive shelf-life curve with live temp slider, savings calculator, scenario comparator, expert chatbot, radar chart, score breakdown, and downloadable ReportLab PDF).
2. **⏱️ Shelf-Life Simulator:** Dedicated standalone simulation tool calculating exact days to failure from moisture gain/loss and oxygen oxidation, complete with dynamic interactive parameter sweeps across temperature and humidity.
3. **⚖️ Material Comparator:** Side-by-side technical evaluation of 2 to 4 user-selected packaging materials, featuring difference tables, percentage delta badges, and overlapping 6-axis Plotly radar profiles.
4. **🌱 Sustainability & LCA:** Dedicated environmental footprint explorer comparing cradle-to-gate carbon footprints ($kg\ \text{CO}_2/kg$), recyclability grades (A to F), biodegradability status, and a dynamic "Cost vs. Carbon Footprint" trade-off quadrant chart.
5. **🧠 ML & Material Clustering:** Unsupervised machine learning sandbox with dynamic K-Means clustering ($k=2..5$), 2D Principal Component Analysis (PCA) projection, feature variance breakdown, and an educational Random Forest feature-importance analysis.
6. **🗄️ Database Explorer & Admin:** Complete administrative dashboard to view the active polymer database, inspect material specifications, add new custom materials with real-time validation, edit existing records, or soft-delete obsolete candidates with audit logging.
7. **📂 CSV Data Import / Export:** Bulk data management suite allowing users to export the complete SQLite material repository to CSV, download standardized validation templates, and upload batch CSVs with strict schema checking.
8. **📖 User Guide & Scientific Methodology:** In-app documentation reference detailing the mathematical formulas, Arrhenius coefficients, knockout rules, literature sources, and practical usage tips.

---

## 4. Light Mode and Dark Mode Implementation & Theme Consistency

To eliminate theme clashes (such as dark buttons or black containers on light backgrounds), the application implements a centralized styling and theme architecture:

### 4.1 Base Theme Configuration (`.streamlit/config.toml`)
Streamlit's default behavior inherits the host OS theme (often dark on Windows). To prevent Streamlit from initializing dark widget styles under Light Mode, `.streamlit/config.toml` establishes a clean, high-contrast light base theme:
```toml
[theme]
primaryColor = "#0d9488"
backgroundColor = "#f8fafc"
secondaryBackgroundColor = "#ffffff"
textColor = "#0f172a"
font = "sans serif"

[server]
headless = true
enableCORS = false
enableXsrfProtection = false
```

### 4.2 Centralized CSS Variable System (`:root`)
In `app.py`, `get_custom_css(dark_mode=False)` injects a dynamic `:root` CSS variable system that consistently updates the entire UI:

- **Light Mode Palette:**
  - Page Background (`--bg-page`): `#f8fafc` (Soft slate white)
  - Card Surface (`--bg-card`): `#ffffff` (Pure white)
  - Primary Text (`--text-primary`): `#0f172a` (Deep navy charcoal, high contrast)
  - Secondary Text (`--text-secondary`): `#475569` (Readable slate grey)
  - Border Accents (`--border-color`): `#e2e8f0` (Subtle boundary line)
  - Buttons (`--btn-secondary-bg`): `#ffffff`, border `#cbd5e1`, text `#0f172a`
  - Button Hover (`--btn-hover-bg`): `#f0fdfa`, border `#0d9488`
  - Primary Button: `linear-gradient(135deg, #0d9488 0%, #0f766e 100%)`, text `#ffffff`
  - Hero Banner (`--hero-grad`): `linear-gradient(135deg, #0f766e 0%, #0d9488 45%, #059669 100%)` (Fresh, vibrant emerald-teal without dark black blocks)
  - Stepper Completed: Background `#f0fdf4`, border `#86efac`, text `#166534`
  - Metric Display (`--metric-bg`): `#f0fdfa`, border `#0d9488`

- **Dark Mode Palette:**
  - Page Background (`--bg-page`): `#0b1120` (Obsidian dark navy)
  - Card Surface (`--bg-card`): `#1e293b` (Elevated slate surface)
  - Primary Text (`--text-primary`): `#f8fafc` (Crisp off-white)
  - Secondary Text (`--text-secondary`): `#94a3b8` (Muted slate grey)
  - Border Accents (`--border-color`): `#334155`
  - Buttons (`--btn-secondary-bg`): `#1e293b`, border `#475569`, text `#f8fafc`
  - Button Hover (`--btn-hover-bg`): `#2d3b52`, border `#14b8a6`
  - Primary Button: `linear-gradient(135deg, #0d9488 0%, #0f766e 100%)`, text `#ffffff`
  - Hero Banner (`--hero-grad`): `linear-gradient(135deg, #090e17 0%, #0f3d3e 45%, #0f766e 100%)`
  - Stepper Completed: Background `#064e3b`, border `#059669`, text `#a7f3d0`
  - Metric Display (`--metric-bg`): `#132338`, border `#14b8a6`

### 4.3 Widget-Level Overrides
All standard Streamlit interactive widgets are explicitly themed using CSS variables:
- **Buttons (`.stButton > button`):** Automatically adapt their background, borders, and text to the active theme.
- **Form Inputs (`.stTextInput input`, `.stNumberInput input`, `.stTextArea textarea`):** Styled with clean background, matching border, and high-contrast text.
- **Select Boxes & Dropdowns (`.stSelectbox`, `div[data-baseweb="select"]`):** Custom backgrounds and text colors avoiding unreadable dark text on dark surfaces or vice-versa.
- **Radio Buttons & Checkboxes (`div[role="radiogroup"]`):** Contained in clean card blocks with accessible labels.
- **Tabs (`.stTabs`):** Contained in styled tab strips with active tab highlights.
- **Metrics (`div[data-testid="stMetric"]`):** Styled as rounded cards with high-contrast metric values.
- **Progress Bars (`.stProgress`):** Smooth emerald-teal gradients with theme-aware tracks.

### 4.4 Plotly Chart Theme Synchronization (`apply_chart_theme`)
Every Plotly chart in the application passes through `apply_chart_theme(fig, dark_mode=st.session_state["dark_mode"], height=...)`, ensuring:
- `paper_bgcolor` and `plot_bgcolor` are transparent (`rgba(0,0,0,0)`), integrating seamlessly into card containers.
- Fonts use *Plus Jakarta Sans* with dynamic text color (`#f8fafc` in Dark Mode, `#0f172a` in Light Mode).
- Axes gridlines use subtle translucent strokes (`rgba(255,255,255,0.08)` in Dark Mode, `rgba(0,0,0,0.06)` in Light Mode).

---

## 5. Four-Step Recommendation Wizard

The recommendation wizard provides an intuitive linear workflow with full state persistence:

```
[Step 1: Food Commodity] ──▶ [Step 2: Storage & Pack] ──▶ [Step 3: Priorities] ──▶ [Step 4: Results & Insights]
      (Icon Cards)              (Preset Buttons)            (MCDA Weights)          (Gauges & Calculations)
```

1. **Visual Stepper Component:**
   - Displays 4 distinct step cards with live status badges:
     - `✓ Completed` (Emerald green badge) for completed steps.
     - `● Current Step` (Teal badge with subtle glow) for the active step.
     - `Step N` (Subtle outlined badge) for upcoming steps.
   - Interactive buttons below each card allow jumping to any step.
2. **Step 1 — Food Commodity:**
   - Grid of 8 prominent icon cards: 🥔 Potato Chips, ☕ Coffee, 🍓 Berries, 🍞 Bread, 🍝 Pasta, 🥛 Milk Powder, 🧀 Cheese, 🥜 Almonds.
   - Selecting a card highlights it with a teal border and badge, immediately populating the sensitivity profile.
   - Full commodity dropdown containing all 12 database presets.
3. **Step 2 — Storage & Packaging Geometry:**
   - 3 Quick-Preset buttons:
     - 🧊 *Chilled / Refrigerator* (4°C, 85% RH)
     - 🏠 *Room Temperature* (23°C, 60% RH)
     - ☀️ *Hot & Humid Climate* (38°C, 90% RH)
   - Fine-tuning sliders for Temperature, Relative Humidity, Target Shelf-Life, Surface Area ($cm^2$), and Headspace ($mL$).
4. **Step 3 — Strategic Priorities:**
   - Strategy selector: Balanced, Lowest Cost, Max Moisture, Max Oxygen, Max Sustainability.
   - Expandable fine-tuning sliders for Barrier ($w_b$), Cost ($w_c$), and Eco ($w_s$) weights.
5. **Step 4 — Results & Deep Insights:**
   - Circular Plotly Score Ring gauge ($0 - 100$).
   - Large shelf-life numeric metric (e.g. `~180 Days`) with `🟢 Shelf-Life Safe` status pill.
   - Candidate cards with rank badges (Gold, Silver, Bronze), star ratings for Moisture Protection (WVTR) and Oxygen Protection (OTR), cost per kg, recyclability stream, requirements met, and limitations.
   - Distinct **Back** (secondary) and **Proceed / Download** (primary) buttons.

---

## 6. Advanced Features (Real Database Computations)

All advanced features compute exclusively from real SQLite database properties and mass-transfer physics without fabricated numbers:

### 6.1 Interactive Shelf-Life Curve (Quality vs. Days) with Live Temp Slider
- Drag a live temperature slider ($5^\circ\text{C}$ to $45^\circ\text{C}$) in the Results section.
- Recalculates Arrhenius permeation acceleration in real time using `shelf_life.estimate_moisture_shelf_life`.
- Renders an interactive Plotly Quality Retention curve showing exact days until critical spoilage.

### 6.2 Packaging Cost & $\text{CO}_2$ Savings Calculator
- User enters monthly packaging units (e.g. 50,000 units/month).
- Compares user's "Current Packaging Material" against the "Recommended Material".
- Computes package film mass: $\text{Mass} = \text{Area} \times \text{Thickness} \times \text{Density}$.
- Calculates exact annual financial savings (\$) using `cost_index_usd_per_kg`.
- Calculates exact annual greenhouse gas emissions avoided ($kg\ \text{CO}_2\text{-eq}$) using `carbon_footprint_kg_co2_per_kg`.

### 6.3 Bilingual Support (English & Hindi)
- Header button (`🌐 हिन्दी` / `🌐 English`) dynamically translates all navigation links, hero headlines, step descriptions, ratings, and technical terminology.

### 6.4 Multi-Scenario Session Comparator (Save up to 3 Scenarios)
- Button saves the current query scenario into `st.session_state["saved_scenarios"]`.
- Displays a side-by-side comparison matrix of saved foods, storage climates, top picks, match scores, shelf lives, and costs.

### 6.5 "Ask the Packaging Expert" Rule-Based Chatbot
- Zero-latency local chatbot requiring no external API keys.
- Quick prompt chips:
  - 💡 *"Why was this recommended?"*
  - 🍃 *"Can I use paper packaging?"*
  - ⏱️ *"How to extend shelf life?"*
  - 💧 *"What does WVTR mean?"*
- Contextually analyzes questions against the current food commodity sensitivities and top candidate barrier properties.

### 6.6 One-Click Demo Mode
- Top header button loads the verified Potato Chips preset ($23^\circ\text{C}, 60\%\ \text{RH}, 180\ \text{days}$) and jumps directly to Step 4 results.

---

## 7. Recommendation Engine Architecture

The recommendation engine (`recommender.py`) executes a deterministic, scientifically grounded two-stage filtering and ranking process:

```
[User Food Sensitivity & Storage Inputs] 
                  │
                  ▼
   ┌───────────────────────────────┐
   │    STAGE 1: KNOCKOUT RULES    │  ---> Disqualifies unsafe/incompatible materials
   │  - Thermal Limit Check        │       (Logs exact technical violation reason)
   │  - Respiration & EMAP Check   │
   │  - Puncture Risk vs. Strength │
   │  - Lipid/Grease Resistance    │
   │  - Food Contact Compliance    │
   └──────────────┬────────────────┘
                  │
          [Passed Candidates]
                  │
                  ▼
   ┌───────────────────────────────┐
   │ STAGE 2: MCDA SCORING ENGINE  │  ---> Multi-Criteria Decision Analysis
   │  - S_barrier (WVTR & OTR fit) │       Weighted Score (0 - 100)
   │  - S_cost (Normalized price) │       S_total = w_b*S_b + w_c*S_c + w_s*S_s
   │  - S_sustainability (LCA/Eco) │
   └──────────────┬────────────────┘
                  │
                  ▼
   [Ranked Recommendations + Explainability Dictionaries + Risk Warnings]
```

### 7.1 Stage 1: Knockout Filters (Go / No-Go Constraints)
1. **Thermal Service Limit:** $\text{If } T_{\text{target}} > T_{\text{max\_service}} \implies \mathbf{KNOCKOUT}$
2. **Respiration & EMAP Check:** $\text{If Respiration Rate} \ge 3 \text{ and } \text{OTR} < 250\ \text{cc}/\text{m}^2\cdot\text{day}\cdot\text{atm} \implies \mathbf{KNOCKOUT}$
3. **Puncture Resistance Constraint:** $\text{If Puncture Risk} = \text{High, Tensile Strength} < 35\ \text{MPa, and Puncture Resistance} = \text{False} \implies \mathbf{KNOCKOUT}$
4. **Lipid Degradation Constraint:** $\text{If Lipid Content} = \text{High and Grease Resistance} = \text{False} \implies \mathbf{KNOCKOUT}$
5. **Food Contact Compliance:** $\text{If Food Contact Approved} = \text{False} \implies \mathbf{KNOCKOUT}$

### 7.2 Stage 2: MCDA Scoring Formulation
- **Barrier Score ($S_{\text{barrier}}$):** $0.55 \cdot S_{\text{wvtr}} + 0.45 \cdot S_{\text{otr}} - \text{Penalty}_{\text{light}}$
- **Cost Score ($S_{\text{cost}}$):** $100 \times \left(1 - \frac{C - C_{\min}}{C_{\max} - C_{\min} + \epsilon}\right)$
- **Sustainability Score ($S_{\text{sust}}$):** $\text{Base}_{\text{recyclability}} + \text{Bonus}_{\text{bio}} - \text{Penalty}_{\text{carbon}}$
- **Total Composite Score:** $S_{\text{total}} = w_b \cdot S_{\text{barrier}} + w_c \cdot S_{\text{cost}} + w_s \cdot S_{\text{sust}}$

---

## 8. Database Schema & Seed Data

Managed via `database.py` with embedded SQLite storage (`packaging_knowledge.db`):

1. **`packaging_materials`:** Stores polymer properties (`name`, `structure_type`, `nominal_thickness_um`, `wvtr_g_m2_day`, `otr_cc_m2_day_atm`, `cost_index_usd_per_kg`, `recyclability_status`, `carbon_footprint_kg_co2_per_kg`, `food_contact_compliance`, `temp_range_min_c`, `temp_range_max_c`, `grease_fat_resistance`, `puncture_resistance`).
2. **`commodity_presets`:** Stores 12 food sensitivity profiles (`Potato Chips`, `Roasted Coffee Beans`, `Fresh Strawberries`, `Whole Wheat Bread`, `Roasted Almonds`, `Fresh Cut Lettuce`, `Whole Milk Powder`, `Dry Pasta`, `Raw Chicken Breast`, `Cheddar Cheese`, `Breakfast Cereal`, `Ground Spices / Paprika`).
3. **`audit_log`:** Logs all recommendation queries and administrative database mutations with UTC timestamps.

---

## 9. Scientific Calculations & Shelf-Life Simulation

Implemented in `shelf_life.py` using classical mass transfer kinetics (ASTM E96, ASTM D3985, Labuza 1984, Robertson 2012):

1. **Moisture Sorption Kinetics:**
   $$\Delta m = \frac{m_{\text{critical}} - m_{\text{initial}}}{100} \times W_s$$
   $$\text{WVTR}_{\text{adj}} = \text{WVTR}_{\text{std}} \times \exp(0.045 \times (T - 38)) \times \left(\frac{\text{RH}_{\text{ambient}}}{90}\right)$$
   $$t_{\text{shelf\_moisture}} = \frac{\Delta m}{\text{WVTR}_{\text{adj}} \times A_{\text{pack}}}$$
2. **Oxygen Ingress Kinetics:**
   $$\text{OTR}_{\text{adj}} = \text{OTR}_{\text{std}} \times \exp(0.035 \times (T - 23))$$
   $$\Delta V_{\text{O}_2} = \frac{\text{Critical } \text{O}_2\% - \text{Initial } \text{O}_2\%}{100} \times V_{\text{headspace}}$$
   $$t_{\text{shelf\_oxygen}} = \frac{\Delta V_{\text{O}_2}}{\text{OTR}_{\text{adj}} \times A_{\text{pack}} \times \Delta P_{\text{O}_2}}$$
3. **Mathematical Boundary Guards:**
   - Near-hermetic barriers ($\text{WVTR} \le 0.01$ or $\text{OTR} \le 0.01$) are capped at 1095 days with an informational notation.
   - Outward humidity gradients and pre-spoiled initial conditions are trapped with descriptive alerts.

---

## 10. PDF Report Generation Engine

Implemented in `pdf_generator.py` using ReportLab:
- Two-pass `NumberedCanvas` dynamically calculates total pages ("Page X of Y"), drawing running headers at $y=756$ pt and footers at $y=42$ pt.
- Generates Executive Summary, Commodity Inputs, Top 3 Candidate Cards, Material Comparison Matrix, and Scientific Disclaimers.
- Compiles in memory via `io.BytesIO()`, piped directly to a Streamlit download button.

---

## 11. Project Directory & File Structure

```
Food-Packaging-ai/
│
├── .streamlit/
│   └── config.toml             # Streamlit base light theme & server configuration
│
├── app.py                      # Main Streamlit web application & UI presentation layer
├── database.py                 # SQLite schema, parameterized CRUD, and literature seed data
├── recommender.py              # Two-stage knockout and MCDA recommendation algorithms
├── shelf_life.py               # First-principles mass transfer permeation & shelf-life simulation
├── analytics.py                # Publication-grade interactive Plotly visualization generators
├── ml_engine.py                # KMeans clustering, PCA projection, and baseline Random Forest
├── data_loader.py              # CSV template generation, export, and validated transactional import
├── pdf_generator.py            # ReportLab programmatic vector PDF generator with NumberedCanvas
│
├── packaging_knowledge.db      # Embedded SQLite database
├── requirements.txt            # Python dependency specifications
├── README.md                   # Project documentation and architecture overview
├── PROJECT_MASTER_PROMPT.md    # Definitive system master prompt and implementation manual
│
└── tests/                      # Automated pytest test suite (27 passing tests)
    ├── __init__.py
    ├── test_database.py        # Database creation, seeding, soft-delete, and CRUD tests (3 tests)
    ├── test_recommender.py     # Stage 1 knockout filters, MCDA scoring, and ranking tests (6 tests)
    ├── test_shelf_life.py      # Moisture, oxygen, Arrhenius, and edge-case kinetics tests (9 tests)
    ├── test_analytics_and_ml.py# Radar chart, KMeans, and Random Forest baseline tests (3 tests)
    ├── test_data_loader.py     # CSV export, template generation, and validation tests (4 tests)
    └── test_pdf_generator.py   # PDF byte generation and candidate handling tests (2 tests)
```

---

## 12. Installation, Setup & Execution Guide (Windows)

### 12.1 Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python Version:** Python 3.10 to 3.14 (Verified and active on Python 3.14).

### 12.2 Installation Commands
```powershell
cd "C:\Users\Suraj Yadav\OneDrive\Desktop\Food-Packaging-ai"
py -m venv venv
.\venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### 12.3 Launching the Web Application
```powershell
py -m streamlit run app.py
```
- **Local URL:** `http://localhost:8501`

### 12.4 Running the Test Suite
```powershell
py -m pytest tests/ -v
```

---

## 13. Testing, Verification & Real Results

### 13.1 Automated Test Execution Results
All 27 automated unit and integration tests passed cleanly:
```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\Users\Suraj Yadav\OneDrive\Desktop\Food-Packaging-ai
plugins: anyio-4.15.1
collected 27 items

tests/test_analytics_and_ml.py::test_radar_missing_data_warning PASSED   [  3%]
tests/test_analytics_and_ml.py::test_ml_clustering_and_pca PASSED        [  7%]
tests/test_analytics_and_ml.py::test_supervised_baseline_training PASSED [ 11%]
tests/test_data_loader.py::test_export_and_template PASSED               [ 14%]
tests/test_data_loader.py::test_import_missing_mandatory_columns PASSED  [ 18%]
tests/test_data_loader.py::test_import_negative_numeric_values PASSED    [ 22%]
tests/test_data_loader.py::test_import_duplicate_name_skipped PASSED     [ 25%]
tests/test_database.py::test_database_initialization PASSED              [ 29%]
tests/test_database.py::test_add_and_delete_material PASSED              [ 33%]
tests/test_database.py::test_audit_log_operations PASSED                 [ 37%]
tests/test_pdf_generator.py::test_pdf_generation_returns_valid_bytes PASSED [ 40%]
tests/test_pdf_generator.py::test_pdf_generation_empty_candidates PASSED [ 44%]
tests/test_recommender.py::test_empty_materials_list PASSED              [ 48%]
tests/test_recommender.py::test_fresh_produce_knockout PASSED            [ 51%]
tests/test_recommender.py::test_extreme_temperature_knockout PASSED      [ 55%]
tests/test_recommender.py::test_puncture_risk_knockout PASSED            [ 59%]
tests/test_recommender.py::test_weights_impact_on_ranking PASSED         [ 62%]
tests/test_recommender.py::test_explainability_payload PASSED            [ 66%]
tests/test_shelf_life.py::test_valid_moisture_shelf_life PASSED          [ 70%]
tests/test_shelf_life.py::test_invalid_moisture_thresholds PASSED        [ 74%]
tests/test_shelf_life.py::test_negative_physical_inputs PASSED           [ 77%]
tests/test_shelf_life.py::test_oxygen_ingress_simulation PASSED          [ 81%]
tests/test_shelf_life.py::test_hermetic_material_handling PASSED         [ 85%]
tests/test_shelf_life.py::test_oxygen_ingress_zero_gradient PASSED       [ 88%]
tests/test_shelf_life.py::test_oxygen_ingress_outward_gradient PASSED    [ 92%]
tests/test_shelf_life.py::test_oxygen_ingress_unreachable_threshold PASSED [ 96%]
tests/test_shelf_life.py::test_oxygen_ingress_never_negative PASSED      [100%]

============================= 27 passed in 2.62s ==============================
```

### 13.2 UI Debugging & Root Cause Analysis
Previous UI iterations suffered from several subtle root causes in Streamlit 1.66:
1. **Child Combinator Failure on Buttons:** Streamlit 1.66 wraps buttons inside intermediate containers and emotion wrappers (`stBaseButton-secondary`, `stBaseButton-primary`). CSS selectors using strict child combinators (`.stButton > button`) failed to override emotion styles, causing secondary buttons to fall back to dark charcoal (`rgb(38, 39, 48)`), appearing as black blocks in Light Mode.
2. **Text Corruption inside Buttons via Global `<p>` Selector:** Streamlit renders button labels within `<div data-testid="stMarkdownContainer"><p>Label</p></div>`. Global rules targeting `p { color: var(--text-secondary) !important; }` overrode button label colors, turning white text inside teal primary buttons into washed-out slate grey.
3. **Missing Streamlit BaseWeb CSS Variables:** Streamlit BaseWeb widgets (popovers, selectboxes, expanders) look for `--background-color`, `--secondary-background-color`, and `--text-color` on `:root, .stApp`. Explicitly binding these variables ensures native components correctly respect the light/dark palette.
4. **Redundant Title Duplication in Stepper:** Each stepper step column originally stacked an HTML card and an interactive button repeating the identical title. This was streamlined into a clean card with an action-oriented navigation button (`↩ Review Step 1`, `● Active Step 2`, `Go to Step 3 →`).
5. **Polar Coordinate Theming on Radar Charts:** Plotly radar charts use `layout.polar` axes (`radialaxis`, `angularaxis`) rather than cartesian `xaxis`/`yaxis`. Explicit polar grid color and legend font updates were incorporated into `apply_chart_theme`.
6. **Theme Defaulting in `.streamlit/config.toml`:** Created explicit light theme configuration (`backgroundColor = "#f8fafc"`, `textColor = "#0f172a"`, `primaryColor = "#0d9488"`) to prevent automatic dark-mode defaulting in environments where the operating system prefers dark mode.

### 13.3 Manual Visual Verification Checklist
Browser automation encountered sandbox driver limitations (Microsoft CDN Playwright package unavailable in container); therefore, complete visual verification is supported by this precise manual checklist:

| # | Page / Section | Elements to Verify | Light Mode Expectation | Dark Mode Expectation |
|---|----------------|-------------------|------------------------|-----------------------|
| 1 | **Header & Navigation** | App title, Demo button, Language button, Theme toggle, Horizontal Menu | White card background, crisp teal badge, readable text `#0f172a`, active tab `#0d9488` | Dark obsidian background `#1e293b`, white text, active tab `#0d9488` |
| 2 | **Dashboard & Hero** | Hero banner, Start Wizard & Demo buttons | Vibrant gradient (`#0f766e` to `#059669`), white text, no black blocks | Deep navy gradient (`#090e17` to `#0f766e`), white text |
| 3 | **Wizard Stepper** | 4-step progress bar & step navigation cards | Step 1/2/3/4 cards on white `#ffffff`, active step teal border & `#f0fdfa` tint, secondary buttons pure white, no black button boxes | Cards `#1e293b`, active step `#132338`, subtle slate borders |
| 4 | **Step 1: Food Cards** | 8 commodity cards (Potato Chips, Coffee, etc.) | Selected card has teal border + `#f0fdfa` tint, primary button teal, secondary button white | Selected card `#132338`, secondary cards `#1e293b` |
| 5 | **Step 2: Storage** | Cool, Room, Hot presets & sliders | Active preset button highlighted in teal primary style, sliders smoothly draggable | Active preset button highlighted, dark cards |
| 6 | **Step 3: Priorities** | Strategy radio list & weight sliders | Card container white with clean borders, radio labels dark `#0f172a` | Card container `#1e293b`, radio labels light `#f8fafc` |
| 7 | **Step 4: Results** | Match gauge, freshness horizon, top 3 cards, live shelf-life slider, savings calculator, chatbot, PDF download | Metric box `#f0fdfa` with teal border, candidate cards white with gold/silver badges, clean chat bubbles, download button teal | Metric box `#132338`, candidate cards `#1e293b`, dark chat bubbles |
| 8 | **Shelf-Life Simulator** | Moisture & oxygen tabs, parameter inputs, sorption curves | Pure white background, teal curve, red spoilage threshold, dark text | Dark background, teal/sky curve, light text |
| 9 | **Material Comparator** | Multiselect dropdown, radar chart, side-by-side data table | Table with light header, radar chart transparent with dark labels | Table with dark header, radar chart with light labels |
| 10 | **Sustainability & ML** | Recycling bar chart, carbon scatter plot, K-Means PCA chart, Random forest importance | All chart backgrounds transparent, gridlines subtle light grey, legend readable | All chart backgrounds transparent, gridlines subtle dark grey |
| 11 | **Database & CSV** | Material table, Add/Edit forms, CSV download & upload | Form inputs pure white with `#cbd5e1` borders, tables cleanly bordered | Form inputs `#1e293b` with `#475569` borders |
| 12 | **User Guide** | LaTeX math formulas, citation references, audit log table | Clear black LaTeX typography, clean table borders | Clear light LaTeX typography, dark table borders |

---

## 14. System Maintenance & Extension Guidelines

When asked to extend, modify, or refactor this codebase:

1. **Preserve Scientific Integrity:**
   - Never alter permeation units (WVTR: $\text{g}/\text{m}^2/\text{day}$, OTR: $\text{cc}/\text{m}^2/\text{day}/\text{atm}$).
   - Retain mathematical boundary guards (zero permeation, inverted humidity gradients, pre-spoiled initial conditions).
2. **Maintain Theme Consistency:**
   - Never introduce hardcoded black backgrounds (`#0f172a`, `#000000`) for light mode elements.
   - Always use CSS variables (`var(--bg-card)`, `var(--text-primary)`, `var(--border-color)`).
   - Route all Plotly figures through `apply_chart_theme(fig, dark_mode=st.session_state["dark_mode"])`.
   - Prevent `<p>` selectors from overriding button text colors with `button p { color: inherit !important; }`.
3. **Transactional Database Safety:**
   - When modifying database records, always record changes in `audit_log`.
   - Never perform unparameterized raw SQL string interpolation.
4. **Two-Pass PDF Canvas Safety:**
   - Retain `NumberedCanvas` in `pdf_generator.py` for accurate page-count calculation.
5. **Test-Driven Refactoring:**
   - Any modification must be verified by running `py -m pytest tests/ -v`. Ensure all 27 tests pass before concluding any task.

