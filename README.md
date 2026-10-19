# AI-Based Intelligent Food Packaging Material Recommendation System for Food Commodities

An end-to-end, physics-grounded decision support prototype designed for food science, packaging engineering, and supply chain quality optimization. Built with Python, Streamlit, Pandas, NumPy, Plotly, SQLite, and Scikit-learn.

---

🔗 **Live Demo:** https://food-packaging-ai.streamlit.app/

## 🚀 Quick Start Guide (Windows)

### Prerequisites
- Python 3.10+ (Tested on Python 3.14 on Windows)
- Git (optional)

### 1. Installation
Open PowerShell or Command Prompt in the project directory:
```powershell
# Verify Python
py --version

# Install dependencies
py -m pip install -r requirements.txt
```

### 2. Run the Web Application
```powershell
py -m streamlit run app.py
```
The application will launch automatically in your default browser at:
`http://localhost:8501`

### 3. Run Automated Unit & Integration Tests
```powershell
py -m pytest tests/ -v
```
All 27 automated tests covering recommendations, knockouts, missing data handling, database operations, physical calculations, PDF generation, and oxygen-ingress edge cases will execute.

---

## 🌟 Key Features Implemented

### 1. Professional Dashboard & Modern UI
- Clean, responsive dashboard designed with a blue-green (teal / navy / emerald) theme.
- Dynamic KPI metrics: Database count, qualified candidates, disqualified materials, and current top pick.
- Glassmorphism ranking cards with custom badge pills (#1 Recommended, #2 Alternative, #3 Budget/Eco Alternative).

### 2. Food Commodity Analyzer
- Covers 6 standard commodity categories:
  - **Cereals & Grains** (e.g. Whole Wheat Flour, Basmati Rice)
  - **Pulses** (e.g. Green Lentils, Chickpeas)
  - **Bakery Products** (e.g. White Bread, Dry Digestive Biscuits)
  - **Snacks** (e.g. Fried Potato Chips, Roasted Almonds)
  - **Fruits & Vegetables** (e.g. Fresh Cut Apple Slices, Strawberries)
  - **Dairy Products** (e.g. Whole Milk Powder, Cheddar Cheese, Salted Butter)
- Configures moisture sensitivity, lipid oxidation risk, fat content, light sensitivity, storage temperature (-40°C to 60°C), relative humidity, puncture hazards, and target shelf life.

### 3. Structured Packaging Material Knowledge Base
- Backed by persistent SQLite storage (`packaging_knowledge.db`).
- Pre-populated with 16 realistic, literature-grounded packaging materials across monolayer, multilayer, vacuum metallized, paperboard composite, bio-based, and rigid containers.
- Standard test metrics cited:
  - **WVTR** (g/m²/24h per ASTM F1249 at 38°C, 90% RH)
  - **OTR** (cc/m²/24h/atm per ASTM D3985 at 23°C, 0% RH)
  - Thickness (µm), tensile strength (MPa), seal integrity, puncture resistance
  - Cost index ($/kg estimated benchmark)
  - Recyclability status (PE, HDPE, PP, PET, Glass, Paperboard pulping, Compostable, or Multilayer Laminate)
  - Food-contact compliance evidence (US FDA 21 CFR §177, EU Regulation 10/2011, FSSAI)
  - Data citations and test standards.

### 4. Transparent Recommendation Engine
- **Stage 1 (Scientific Knockout Constraints):**
  - Thermal survivability checks.
  - Active respiration rules: Hermetic foil or zero-barrier webs are knocked out for fresh fruits and vegetables to prevent anaerobic fermentation and pathogen growth.
  - Puncture resistance checks for sharp-edged commodities.
  - High lipid resistance checks.
  - Food contact compliance verification.
- **Stage 2 (Multi-Criteria Decision Analysis - MCDA):**
  - Normalized utility scoring across Barrier fitness, Cost economy, and Sustainability.
  - Interactive preference weighting sliders ($w_{barrier}$, $w_{cost}$, $w_{sustainability}$).
  - Top 3 recommendations presented with complete transparency.
  - Discloses exact requirements satisfied, operational limitations, and unmeasured data items.

### 5. Multi-Material Radar Analytics & Comparison
- Interactive 6-axis Plotly radar chart comparing Moisture Barrier, Oxygen Barrier, Cost Economy, Recyclability, Thermal Window, and Mechanical Strength.
- Honest handling of missing property values without silent fabrication (alerts user if tensile strength or carbon footprint is unmeasured in literature).

### 6. Sustainability & Cost Trade-off Analysis
- Log-log Physical Barrier Map (WVTR vs. OTR) highlighting packaging requirement zones.
- Economic vs. Cradle-to-gate carbon footprint scatter plots.
- Recyclability distribution by circular streams.
- Documentation of the **Packaging-Food Waste Paradox** (avoiding false assumptions of universal sustainability).

### 7. Shelf-Life Permeation Simulator
- First-principles steady-state moisture sorption calculation based on Robertson (2012) and Labuza (1982).
- Arrhenius temperature and humidity gradient corrections.
- Headspace oxygen ingress kinetics.
- Clear disclaimer: Packaging permeation is a scenario model, not an empirical microbiological shelf-life guarantee.

### 8. Machine Learning Pattern Discovery
- **Unsupervised Learning:** K-Means clustering ($k=2 \dots 6$) and 2D Principal Component Analysis (PCA) projection of packaging materials in the 5D physical/economic space.
- **Supervised Baseline:** Random Forest classification model demonstrating how food attributes map to packaging archetypes, with Mean Decrease in Impurity (MDI) feature importance.
- Transparent scientific disclosure distinguishing literature-based deterministic rule models from empirical accelerated storage testing (AST) data.

### 9. Database Management & Ingestion
- Add new custom materials via UI form with input validation.
- Export entire database to CSV.
- Import external CSV datasets with schema validation, negative value detection, and duplicate skipping.
- Audit trail recording every user recommendation query.

---

## 📁 Project Structure

```
Food-Packaging-ai/
├── app.py                      # Main Streamlit web application & UI
├── database.py                 # SQLite database schema, seed data, and CRUD
├── recommender.py              # Knockout filtering, MCDA scoring, and explainability
├── shelf_life.py               # Mass-transfer permeation scenario simulator
├── analytics.py                # Plotly radar, barrier map, and trade-off charts
├── ml_engine.py                # K-Means clustering, PCA, and ML baseline
├── pdf_generator.py            # ReportLab publication-quality PDF report generation
├── data_loader.py              # CSV import/export and validation
├── packaging_knowledge.db      # Persistent SQLite database file
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & manual
└── tests/                      # Automated test suite (27 unit tests)
    ├── test_database.py
    ├── test_recommender.py
    ├── test_shelf_life.py
    ├── test_pdf_generator.py
    ├── test_data_loader.py
    └── test_analytics_and_ml.py
```

---

## 🔬 Scientific Disclaimers & Validation Scope

1. **Test Conditions vs Ambient Storage:**
   Standard literature values for WVTR are measured at 38°C and 90% RH (ASTM F1249), while OTR is measured at 23°C and 0% RH (ASTM D3985). Real warehouse conditions fluctuate. The simulator applies Arrhenius temperature adjustments, but real packaging lines must conduct humidity-controlled chamber tests.

2. **Multilayer Recycling Realities:**
   While metallized films or PET/EVOH/PE multi-layer laminates offer superior barrier performance for fatty and moisture-sensitive snacks, they cannot be recycled in standard mechanical curbside recycling streams.

3. **Determinants of Food Shelf Life:**
   Packaging permeability is only **one of several critical factors** governing shelf life. Initial product water activity ($a_w$), microbial contamination level, headspace volume, light exposure, pinholes in heat seals, and enzyme denaturation all influence the actual spoilage horizon.
