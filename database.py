"""
database.py - SQLite Database Management Layer
Provides persistent storage, schema management, and literature-referenced
packaging materials & food commodity baseline profiles.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
import pandas as pd

DB_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packaging_knowledge.db")


def get_db_connection(db_path: str = DB_FILE_PATH) -> sqlite3.Connection:
    """Create a SQLite database connection with row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_database(db_path: str = DB_FILE_PATH, force_reset: bool = False) -> None:
    """
    Initializes tables and seeds default literature data if empty or forced.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    if force_reset:
        cursor.execute("DROP TABLE IF EXISTS audit_log;")
        cursor.execute("DROP TABLE IF EXISTS commodity_presets;")
        cursor.execute("DROP TABLE IF EXISTS packaging_materials;")

    # 1. Packaging Materials Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS packaging_materials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        structure_type TEXT NOT NULL,
        polymer_code TEXT NOT NULL,
        nominal_thickness_um REAL NOT NULL,
        wvtr_g_m2_day REAL NOT NULL,
        otr_cc_m2_day_atm REAL NOT NULL,
        co2tr_cc_m2_day_atm REAL,
        light_barrier_rating TEXT NOT NULL,
        tensile_strength_mpa REAL,
        seal_integrity_rating TEXT NOT NULL,
        puncture_resistance TEXT NOT NULL,
        cost_index_usd_per_kg REAL NOT NULL,
        recyclability_status TEXT NOT NULL,
        carbon_footprint_kg_co2_per_kg REAL,
        food_contact_compliance TEXT NOT NULL,
        temp_range_min_c REAL NOT NULL,
        temp_range_max_c REAL NOT NULL,
        grease_fat_resistance TEXT NOT NULL,
        acid_resistance TEXT NOT NULL,
        data_source TEXT NOT NULL,
        test_conditions TEXT NOT NULL,
        is_verified_experimental INTEGER NOT NULL DEFAULT 0,
        image_path TEXT,
        notes TEXT
    );
    """)

    # 2. Food Commodity Presets Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS commodity_presets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        category TEXT NOT NULL,
        moisture_sensitivity TEXT NOT NULL,
        oxygen_sensitivity TEXT NOT NULL,
        fat_content_level TEXT NOT NULL,
        light_sensitivity TEXT NOT NULL,
        recommended_temp_c REAL NOT NULL,
        recommended_rh_percent REAL NOT NULL,
        target_shelf_life_days INTEGER NOT NULL,
        puncture_risk TEXT NOT NULL,
        typical_spoilage_mode TEXT NOT NULL,
        critical_aw REAL,
        image_path TEXT,
        source TEXT,
        notes TEXT
    );
    """)

    # Backward compatibility migrations for existing database files
    cursor.execute("PRAGMA table_info(commodity_presets);")
    existing_cols = {row["name"] for row in cursor.fetchall()}
    if "image_path" not in existing_cols:
        cursor.execute("ALTER TABLE commodity_presets ADD COLUMN image_path TEXT;")
    if "source" not in existing_cols:
        cursor.execute("ALTER TABLE commodity_presets ADD COLUMN source TEXT;")

    cursor.execute("PRAGMA table_info(packaging_materials);")
    existing_mat_cols = {row["name"] for row in cursor.fetchall()}
    if "image_path" not in existing_mat_cols:
        cursor.execute("ALTER TABLE packaging_materials ADD COLUMN image_path TEXT;")


    # 3. Audit Log / Recommendation History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        commodity_name TEXT NOT NULL,
        category TEXT NOT NULL,
        inputs_json TEXT NOT NULL,
        weights_json TEXT NOT NULL,
        top_recommendations_json TEXT NOT NULL
    );
    """)

    conn.commit()

    # Seed data if tables are empty
    cursor.execute("SELECT COUNT(*) FROM packaging_materials;")
    count_mats = cursor.fetchone()[0]
    if count_mats == 0:
        seed_default_materials(conn)

    cursor.execute("SELECT COUNT(*) FROM commodity_presets;")
    count_commodities = cursor.fetchone()[0]
    if count_commodities == 0:
        seed_default_commodity_presets(conn)

    conn.close()


def seed_default_materials(conn: sqlite3.Connection) -> None:
    """
    Seeds realistic packaging materials with values cited from standard packaging handbooks.
    Standard test conditions:
    - WVTR: ASTM F1249 at 38°C, 90% RH
    - OTR: ASTM D3985 / ASTM F1927 at 23°C, 0% RH
    """
    materials = [
        (
            "LDPE Monolayer Film (50 µm)",
            "Monolayer Flexible",
            "LDPE (#4)",
            50.0,
            12.5,   # WVTR: 10-15 g/m2/day
            6500.0, # OTR: Very high (permeable to oxygen)
            27000.0,
            "Low (Transparent)",
            24.0,
            "Good",
            "Medium",
            2.10,
            "Widely Recyclable (PE Stream)",
            2.05,
            "US FDA 21 CFR §177.1520; EU Regulation 10/2011; FSSAI",
            -40.0,
            80.0,
            "Moderate",
            "High",
            "Robertson (2012) Food Packaging: Principles & Practice, Ch. 2; Massey (2003)",
            "WVTR: 38°C / 90% RH (ASTM F1249); OTR: 23°C / 0% RH (ASTM D3985)",
            0,
            "Standard commodity polyolefin film. Superb water vapor barrier, poor oxygen barrier. Excellent heat sealability."
        ),
        (
            "HDPE Blown Film (50 µm)",
            "Monolayer Flexible",
            "HDPE (#2)",
            50.0,
            4.8,    # WVTR: 4-6 g/m2/day
            1800.0, # OTR: moderate-high
            7500.0,
            "Moderate (Translucent)",
            35.0,
            "Good",
            "High",
            2.25,
            "Widely Recyclable (HDPE Stream)",
            1.95,
            "US FDA 21 CFR §177.1520; EU 10/2011 compliant",
            -50.0,
            110.0,
            "High",
            "High",
            "Robertson (2012); Pira International Packaging Handbook",
            "WVTR: 38°C / 90% RH; OTR: 23°C / 0% RH",
            0,
            "High density polyethylene. Superior moisture and grease resistance compared to LDPE with higher tensile strength."
        ),
        (
            "BOPP Film (25 µm)",
            "Monolayer Biaxially Oriented",
            "PP (#5)",
            25.0,
            4.2,    # WVTR: 3.5-5.0 g/m2/day
            1350.0, # OTR
            5200.0,
            "Low (High Gloss Transparency)",
            140.0,
            "Moderate",
            "Medium",
            2.40,
            "Widely Recyclable (PP Stream)",
            1.98,
            "US FDA 21 CFR §177.1520; EU 10/2011",
            -10.0,
            125.0,
            "High",
            "High",
            "Kiran-Raj & Gontard (2014) Barrier Polymers; Massey (2003)",
            "WVTR: 38°C / 90% RH (ASTM F1249); OTR: 23°C / 0% RH (ASTM D3985)",
            0,
            "Biaxially oriented polypropylene. Excellent moisture barrier, high tensile stiffness, ideal for dry bakery and snacks."
        ),
        (
            "Metallized BOPP (Met-BOPP, 20 µm)",
            "Vacuum Metallized Film",
            "PP (#5) + Al layer (<0.04 µm)",
            20.0,
            0.65,   # WVTR: < 1.0 g/m2/day
            45.0,   # OTR: substantially reduced by aluminium vapor layer
            180.0,
            "Total Opaque (Optical Density > 2.2)",
            155.0,
            "Good (with seal layer)",
            "Medium",
            3.20,
            "Technically Recyclable (Specialized PP Stream)",
            2.65,
            "US FDA 21 CFR §175.300 / §177.1520; EU 10/2011",
            -10.0,
            120.0,
            "High",
            "Moderate",
            "Han, J. H. (2014) Innovations in Food Packaging, Elsevier",
            "WVTR: 38°C / 90% RH; OTR: 23°C / 0% RH",
            0,
            "Aluminium vapor-deposited BOPP. High moisture, oxygen, and UV light barrier. Staple inner web for fried snack foods."
        ),
        (
            "PET Film (12 µm)",
            "Biaxially Oriented Polyester",
            "PET (#1)",
            12.0,
            25.0,   # WVTR: moderate
            110.0,  # OTR: good gas barrier
            450.0,
            "Low (Clear High Gloss)",
            210.0,
            "Fair (requires seal layer)",
            "High",
            2.60,
            "Widely Recyclable (PET Bottle Stream)",
            2.30,
            "US FDA 21 CFR §177.1630; EU 10/2011",
            -60.0,
            150.0,
            "High",
            "High",
            "Massey (2003); Robertson (2012)",
            "WVTR: 38°C / 90% RH; OTR: 23°C / 0% RH",
            0,
            "High tensile strength, thermal resistance, dimensional stability. Excellent aroma barrier; moderate moisture barrier."
        ),
        (
            "Metallized PET (Met-PET, 12 µm)",
            "Vacuum Metallized Polyester",
            "PET (#1) + Al vapor",
            12.0,
            0.50,   # WVTR: very low
            1.2,    # OTR: very low
            5.0,
            "Total Opaque (OD > 2.5)",
            220.0,
            "Fair (as outer/intermediate laminate web)",
            "High",
            3.60,
            "Multi-material Laminate (Difficult Mechanical Recycling)",
            2.90,
            "US FDA 21 CFR §177.1630 / §175.300; EU 10/2011",
            -50.0,
            150.0,
            "High",
            "High",
            "Han (2014); Yam (2009) The Wiley Encyclopedia of Packaging Technology",
            "WVTR: 38°C / 90% RH; OTR: 23°C / 0% RH",
            0,
            "High barrier metallized film. Outstanding O2 and moisture protection with light blocking. Widely used in coffee and dry milk powders."
        ),
        (
            "PET / EVOH / PE Multi-layer Barrier Film (70 µm)",
            "Co-extruded / Laminated Barrier",
            "Multi-layer Composite (#7)",
            70.0,
            2.1,    # WVTR: controlled by PE
            0.8,    # OTR: EVOH core provides ultra-high gas barrier
            3.5,
            "Low (Clear Transparent)",
            65.0,
            "Excellent",
            "High",
            4.80,
            "Multi-material Non-recyclable (Curbside Incompatible)",
            3.80,
            "US FDA 21 CFR §177.1360 (EVOH) & §177.1520; EU 10/2011",
            -25.0,
            95.0,
            "Total",
            "High",
            "Ebnesajjad, S. (2013) Plastic Films in Food Packaging; Kuraray EVAL Technical Manual",
            "WVTR: 38°C / 90% RH; OTR: 23°C / 0% RH (EVOH RH-dependent)",
            0,
            "Ultra-high oxygen barrier transparent web designed for modified atmosphere packaging (MAP) of fresh cheese, processed meat, and fresh pasta."
        ),
        (
            "BOPP / Met-BOPP / PE Tri-laminate (65 µm)",
            "Adhesive Multi-layer Laminate",
            "Multi-material Laminate (#7)",
            65.0,
            0.45,
            18.0,
            75.0,
            "Total Opaque",
            95.0,
            "Excellent",
            "Medium",
            3.85,
            "Multi-material Non-recyclable (Curbside Incompatible)",
            3.40,
            "US FDA 21 CFR §177.1520; EU 10/2011",
            -10.0,
            105.0,
            "Total",
            "Moderate",
            "Packaging Technology Journal Benchmark; Frito-Lay / PepsiCo Packaging Specs (public literature)",
            "WVTR: 38°C / 90% RH; OTR: 23°C / 0% RH",
            0,
            "Benchmark commercial potato chip and extruded snack pouch structure. Complete moisture and light barrier with high sealing integrity."
        ),
        (
            "Aluminium Foil Laminate (PET / Al Foil / PE, 90 µm)",
            "Foil Barrier Multi-layer",
            "Composite Foil Pouch (#7)",
            90.0,
            0.01,   # WVTR: near zero (hermetic barrier)
            0.01,   # OTR: near zero
            0.01,
            "Total Opaque (100% Light Barrier)",
            85.0,
            "Excellent",
            "Very High",
            5.80,
            "Non-recyclable Landfill / Waste-to-Energy",
            5.20,
            "US FDA 21 CFR §178.3910 & §175.300; EU 10/2011",
            -50.0,
            135.0,
            "Total",
            "Total",
            "Robertson (2012); European Aluminium Foil Association (EAFA)",
            "WVTR: 38°C / 90% RH; OTR: 23°C / 0% RH (ASTM F1249 / ASTM D3985)",
            0,
            "Gold standard absolute barrier retort pouch. Zero transmission of gases, vapors, or light. Required for sterile shelf-stable rations and sensitive milk powders."
        ),
        (
            "BOPA / PE Laminate (Nylon / PE, 75 µm)",
            "Co-extruded Polyamide Laminate",
            "Composite PA/PE (#7)",
            75.0,
            4.5,
            25.0,   # OTR: good oxygen barrier
            110.0,
            "Low (Clear)",
            120.0,
            "Excellent",
            "Very High",
            4.30,
            "Multi-material Non-recyclable",
            3.65,
            "US FDA 21 CFR §177.1500 (Nylon) & §177.1520; EU 10/2011",
            -40.0,
            115.0,
            "High",
            "Moderate",
            "Massey (2003); AlliedSignal Nylon Barrier Technical Bulletin",
            "WVTR: 38°C / 90% RH; OTR: 23°C / 0% RH",
            0,
            "Biaxially Oriented Polyamide (Nylon-6). Outstanding puncture resistance, flex-crack resistance, and deep-draw thermoformability for sharp grains, pulses, and bone-in foods."
        ),
        (
            "PLA Bio-plastic Film (30 µm)",
            "Bio-based Aliphatic Polyester",
            "PLA (#7 Bio-based)",
            30.0,
            175.0,  # WVTR: high (poor water barrier)
            550.0,  # OTR: moderate
            2100.0,
            "Low (Clear High Clarity)",
            55.0,
            "Moderate",
            "Low",
            3.90,
            "Industrially Compostable (ASTM D6400 / EN 13432)",
            1.80,
            "US FDA GRAS Notice No. GRN 000178; EU 10/2011",
            -10.0,
            55.0,   # Low heat resistance
            "Moderate",
            "Low",
            "Auras et al. (2004) Poly(lactic acid): An overview. Macromolecular Bioscience; NatureWorks LLC",
            "WVTR: 38°C / 90% RH; OTR: 23°C / 0% RH",
            0,
            "Bio-derived thermoplastic derived from corn starch or sugarcane. Compostable in commercial facilities. Low thermal tolerance and high moisture permeability."
        ),
        (
            "Coated Cellulose NatureFlex Film (30 µm)",
            "Regenerated Bio-based Cellulose",
            "Cellulose (#7 Compostable)",
            30.0,
            18.0,   # WVTR: coated with bio-sealant
            8.0,    # OTR: excellent gas barrier
            35.0,
            "Low (Clear High Transparency)",
            60.0,
            "Good",
            "Medium",
            5.20,
            "Home & Industrially Compostable (TÜV OK Compost)",
            1.95,
            "US FDA 21 CFR §177.1200; EU 10/2011 compliant",
            -20.0,
            100.0,
            "High",
            "Moderate",
            "Futamura Chemical NatureFlex Technical Datasheet; Siracusa et al. (2008)",
            "WVTR: 38°C / 90% RH; OTR: 23°C / 0% RH",
            0,
            "Wood-pulp derived regenerated cellulose film with bio-barrier coating. High oxygen barrier and natural aroma retention; certified home compostable."
        ),
        (
            "Paperboard / PE Extrusion Coated (250 gsm / 18 µm PE)",
            "Paperboard-Polymer Composite",
            "Paper Composite (#22 / #81)",
            320.0,
            14.0,   # WVTR: determined by thin PE layer
            1200.0,
            4800.0,
            "Total Opaque",
            80.0,
            "Good",
            "High",
            1.75,
            "Recyclable in Paper Pulping Stream (Specialized Mill)",
            1.40,
            "US FDA 21 CFR §176.170 / §177.1520; BfR Recommendation XXXVI",
            -20.0,
            90.0,
            "Moderate",
            "Moderate",
            "Kirwan, M. J. (2013) Paper and Paperboard Packaging Technology, Wiley",
            "WVTR: 38°C / 90% RH; OTR: 23°C / 0% RH",
            0,
            "Solid bleached sulphate (SBS) or folding boxboard with thin LDPE coating. High rigidity, structural protection, carton containers for dry cereals and bakery."
        ),
        (
            "100% Recycled PET Film (rPET, 25 µm)",
            "Mechanically Recycled Polyester",
            "rPET (#1)",
            25.0,
            20.0,
            125.0,
            510.0,
            "Low (Slight Haze)",
            185.0,
            "Fair",
            "Medium",
            2.95,
            "Widely Recyclable (PET Bottle Stream)",
            1.15,
            "US FDA Letter of No Objection (NOL) for 100% PCR; EFSA approved",
            -40.0,
            140.0,
            "High",
            "High",
            "Welle, F. (2011) Twenty years of PET bottle to bottle recycling, Resources Conservation and Recycling",
            "WVTR: 38°C / 90% RH; OTR: 23°C / 0% RH",
            0,
            "Post-consumer recycled polyethylene terephthalate with super-clean decontamination process. Low carbon footprint, circular economy certified."
        ),
        (
            "Micro-perforated BOPP Breathable Film (30 µm)",
            "Laser Micro-perforated Film",
            "PP (#5)",
            30.0,
            180.0,  # WVTR: allows moisture venting to prevent condensation
            8500.0, # OTR: tailored gas transmission for active respiration
            32000.0,
            "Low (Clear Antifog)",
            125.0,
            "Moderate",
            "Medium",
            2.90,
            "Widely Recyclable (PP Stream)",
            2.10,
            "US FDA 21 CFR §177.1520; EU 10/2011",
            0.0,
            70.0,
            "Moderate",
            "Moderate",
            "Kader, A. A. et al. (1989) Modified Atmosphere Packaging of Fruits and Vegetables, Critical Reviews in Food Science; Fonseca et al. (2002)",
            "Tailored per perforation density (50-100 µm holes); 23°C / 85% RH",
            0,
            "Engineered equilibrium modified atmosphere packaging (EMAP) film with laser micro-perforations. Prevents anaerobic fermentation and fogging in fresh produce."
        ),
        (
            "Glass Container with Lug Closure (Rigid)",
            "Inorganic Soda-Lime Silica Glass",
            "Glass (#70)",
            3500.0,
            0.001,  # Impermeable
            0.001,  # Impermeable
            0.001,
            "Low to High (depending on flint vs amber glass)",
            70.0,
            "Excellent (Hermetic Vacuum Seal)",
            "Total",
            4.50,
            "Infinitely Recyclable (Curbside Glass Container Stream)",
            4.10,
            "US FDA GRAS §182.1; EU Regulation 1935/2004 Inert Material",
            -30.0,
            160.0,
            "Total",
            "Total",
            "Marsh & Bugusu (2007) Food Packaging—Roles, Materials, and Environmental Issues, JFS",
            "WVTR: 38°C / 90% RH (0.0); OTR: 23°C / 0% RH (0.0)",
            1,
            "Hermetic absolute barrier container. Complete chemical inertness, infinite recyclability. High transport weight and fracture risk."
        )
    ]

    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO packaging_materials (
        name, structure_type, polymer_code, nominal_thickness_um,
        wvtr_g_m2_day, otr_cc_m2_day_atm, co2tr_cc_m2_day_atm,
        light_barrier_rating, tensile_strength_mpa, seal_integrity_rating,
        puncture_resistance, cost_index_usd_per_kg, recyclability_status,
        carbon_footprint_kg_co2_per_kg, food_contact_compliance,
        temp_range_min_c, temp_range_max_c, grease_fat_resistance,
        acid_resistance, data_source, test_conditions,
        is_verified_experimental, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, materials)
    conn.commit()


def seed_default_commodity_presets(conn: sqlite3.Connection) -> None:
    """
    Seeds baseline food commodities with food science spoilage parameters.
    Attempts to seed full 36-commodity preset catalog from scripts.seed_products,
    falling back to standard baseline presets if not importable.
    """
    try:
        from scripts.seed_products import COMMODITY_DATA
        cursor = conn.cursor()
        for c in COMMODITY_DATA:
            cursor.execute("""
            INSERT OR REPLACE INTO commodity_presets (
                name, category, moisture_sensitivity, oxygen_sensitivity,
                fat_content_level, light_sensitivity, recommended_temp_c,
                recommended_rh_percent, target_shelf_life_days, puncture_risk,
                typical_spoilage_mode, critical_aw, notes, image_path, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                c["name"], c["category"], c["moisture_sensitivity"],
                c["oxygen_sensitivity"], c["fat_content_level"],
                c["light_sensitivity"], c["recommended_temp_c"],
                c["recommended_rh_percent"], c["target_shelf_life_days"],
                c["puncture_risk"], c["typical_spoilage_mode"],
                c["critical_aw"], c["notes"], c.get("image_path"),
                c.get("source")
            ))
        conn.commit()
        return
    except Exception:
        pass

    commodities = [
        (
            "Fried Potato Chips & Crisps",
            "Snacks",
            "Extreme",       # Critical moisture gain causes loss of crispness at aw > 0.45
            "Critical",      # High unsaturated lipid oxidation leads to rancidity (hexanal formation)
            "High (30-35% fat)",
            "High (Light triggers photo-oxidation)",
            23.0,
            60.0,
            180,
            "Low",
            "Rancidity (lipid oxidation) & loss of crispness (moisture absorption)",
            0.40,
            "Requires ultra-low WVTR and OTR plus opaque light barrier and nitrogen flush."
        ),
        (
            "Roasted Cashews & Almonds",
            "Snacks",
            "High",
            "High",
            "High (45-55% fat)",
            "Moderate",
            22.0,
            55.0,
            270,
            "High",          # Sharp nut edges can puncture thin films
            "Oxidative rancidity & softening; insect infestation risk",
            0.50,
            "Requires puncture-resistant barrier pouch with vacuum or gas flush."
        ),
        (
            "Whole Wheat Flour (Atta)",
            "Cereals and grains",
            "High",
            "Moderate",
            "Low-Medium (Contains wheat germ oil)",
            "Low",
            25.0,
            65.0,
            180,
            "Low",
            "Caking, mold growth at aw > 0.65, lipolytic rancidity",
            0.62,
            "Requires moisture barrier to prevent caking and tensile strength for bag handling."
        ),
        (
            "Polished Basmati Rice",
            "Cereals and grains",
            "Moderate",
            "Low",
            "Low (<1% fat)",
            "Low",
            25.0,
            65.0,
            365,
            "High",          # Sharp needle grains
            "Moisture sorption, mold germination, rice weevil infestation",
            0.65,
            "Requires high puncture resistance, aroma retention, and moisture barrier."
        ),
        (
            "Green Lentils & Chickpeas",
            "Pulses",
            "Moderate",
            "Low",
            "Low (1-5% fat)",
            "Low",
            24.0,
            60.0,
            365,
            "Very High",     # Hard angular edges
            "Harden-to-cook defect (HTCD) under high humidity/temp; beetle infestation",
            0.65,
            "Requires tough puncture-resistant film with good moisture protection."
        ),
        (
            "White Sandwich Bread",
            "Bakery products",
            "High (Moisture loss / drying)",
            "Moderate",
            "Low-Medium (3-5% fat)",
            "Low",
            22.0,
            60.0,
            7,
            "Low",
            "Starch retrogradation (staling), crumb moisture migration, Penicillium mold growth",
            0.94,
            "Short shelf life. High moisture barrier required to prevent drying; breathable enough to prevent surface mold condensation."
        ),
        (
            "Dry Digestive Biscuits / Cookies",
            "Bakery products",
            "Extreme",
            "Moderate",
            "Medium-High (15-20% fat)",
            "Moderate",
            24.0,
            55.0,
            180,
            "Medium",
            "Loss of crispness, oxidative off-flavor",
            0.35,
            "Requires tight moisture barrier (WVTR < 2) to maintain critical crispness."
        ),
        (
            "Fresh Cut Apple Slices",
            "Fruits and vegetables",
            "Low (High internal moisture)",
            "Moderate (Needs balanced O2 2-5% to prevent anaerobic browning)",
            "Zero",
            "Moderate",
            4.0,
            90.0,
            12,
            "Low",
            "Enzymatic browning (PPO enzyme), microbial softening, anaerobic off-flavor",
            0.98,
            "Requires Equilibrium Modified Atmosphere Packaging (EMAP) with micro-perforation; antifog coating."
        ),
        (
            "Strawberries & Soft Berries",
            "Fruits and vegetables",
            "Moderate (High transpiration)",
            "Moderate (High respiration rate)",
            "Zero",
            "Low",
            4.0,
            90.0,
            8,
            "Low",
            "Botrytis cinerea (grey mold), moisture condensation, tissue softening",
            0.99,
            "Requires high gas transmission or macro/micro-perforations to release CO2 and moisture venting."
        ),
        (
            "Whole Milk Powder (Full Cream)",
            "Dairy products",
            "Extreme",
            "Critical",
            "High (26-28% butterfat)",
            "High",
            24.0,
            55.0,
            365,
            "Low",
            "Lactose crystallization (caking) at aw > 0.20, lipid oxidation (tallowy flavor)",
            0.20,
            "Requires ultra-high barrier (Foil laminate or Met-PET) with nitrogen purge."
        ),
        (
            "Cheddar Cheese Block",
            "Dairy products",
            "Moderate",
            "Critical (Mold growth on surface)",
            "High (32-35% fat)",
            "High",
            4.0,
            85.0,
            90,
            "Low",
            "Surface mold, moisture loss, fat sweating, lipid photo-oxidation",
            0.95,
            "Requires vacuum skin pack or MAP with EVOH barrier to exclude oxygen and prevent mold."
        ),
        (
            "Salted Butter",
            "Dairy products",
            "Moderate",
            "High",
            "Very High (80-82% milkfat)",
            "Critical (Light activates riboflavin-sensitized oxidation)",
            4.0,
            80.0,
            120,
            "Low",
            "Photo-oxidation, surface rancidity, odor absorption",
            0.90,
            "Requires grease-proof, moisture-tight, opaque wrap (vegetable parchment / foil / HDPE)."
        )
    ]

    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO commodity_presets (
        name, category, moisture_sensitivity, oxygen_sensitivity,
        fat_content_level, light_sensitivity, recommended_temp_c,
        recommended_rh_percent, target_shelf_life_days, puncture_risk,
        typical_spoilage_mode, critical_aw, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, commodities)
    conn.commit()


def get_all_materials(db_path: str = DB_FILE_PATH) -> List[Dict[str, Any]]:
    """Fetch all packaging materials as a list of dicts."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM packaging_materials ORDER BY name ASC;")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_all_materials_df(db_path: str = DB_FILE_PATH) -> pd.DataFrame:
    """Fetch all packaging materials directly as a pandas DataFrame."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM packaging_materials ORDER BY id ASC;", conn)
    conn.close()
    return df


def get_material_by_id(material_id: int, db_path: str = DB_FILE_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve a single packaging material by ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM packaging_materials WHERE id = ?;", (material_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def add_material(mat: Dict[str, Any], db_path: str = DB_FILE_PATH) -> int:
    """Insert a new packaging material into the database with validation."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO packaging_materials (
        name, structure_type, polymer_code, nominal_thickness_um,
        wvtr_g_m2_day, otr_cc_m2_day_atm, co2tr_cc_m2_day_atm,
        light_barrier_rating, tensile_strength_mpa, seal_integrity_rating,
        puncture_resistance, cost_index_usd_per_kg, recyclability_status,
        carbon_footprint_kg_co2_per_kg, food_contact_compliance,
        temp_range_min_c, temp_range_max_c, grease_fat_resistance,
        acid_resistance, data_source, test_conditions,
        is_verified_experimental, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        mat["name"],
        mat.get("structure_type", "Flexible Film"),
        mat.get("polymer_code", "Other (#7)"),
        float(mat["nominal_thickness_um"]),
        float(mat["wvtr_g_m2_day"]),
        float(mat["otr_cc_m2_day_atm"]),
        float(mat["co2tr_cc_m2_day_atm"]) if mat.get("co2tr_cc_m2_day_atm") is not None else None,
        mat.get("light_barrier_rating", "Low"),
        float(mat["tensile_strength_mpa"]) if mat.get("tensile_strength_mpa") is not None else None,
        mat.get("seal_integrity_rating", "Good"),
        mat.get("puncture_resistance", "Medium"),
        float(mat["cost_index_usd_per_kg"]),
        mat.get("recyclability_status", "Non-recyclable"),
        float(mat["carbon_footprint_kg_co2_per_kg"]) if mat.get("carbon_footprint_kg_co2_per_kg") is not None else None,
        mat.get("food_contact_compliance", "Pending Verification"),
        float(mat.get("temp_range_min_c", -20.0)),
        float(mat.get("temp_range_max_c", 80.0)),
        mat.get("grease_fat_resistance", "Moderate"),
        mat.get("acid_resistance", "Moderate"),
        mat.get("data_source", "User Literature Input"),
        mat.get("test_conditions", "Standard Lab Conditions"),
        int(mat.get("is_verified_experimental", 0)),
        mat.get("notes", "")
    ))
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id


def update_material(material_id: int, mat: Dict[str, Any], db_path: str = DB_FILE_PATH) -> bool:
    """Update an existing packaging material in the database."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE packaging_materials SET
        name = ?,
        structure_type = ?,
        polymer_code = ?,
        nominal_thickness_um = ?,
        wvtr_g_m2_day = ?,
        otr_cc_m2_day_atm = ?,
        co2tr_cc_m2_day_atm = ?,
        light_barrier_rating = ?,
        tensile_strength_mpa = ?,
        seal_integrity_rating = ?,
        puncture_resistance = ?,
        cost_index_usd_per_kg = ?,
        recyclability_status = ?,
        carbon_footprint_kg_co2_per_kg = ?,
        food_contact_compliance = ?,
        temp_range_min_c = ?,
        temp_range_max_c = ?,
        grease_fat_resistance = ?,
        acid_resistance = ?,
        data_source = ?,
        test_conditions = ?,
        is_verified_experimental = ?,
        notes = ?
    WHERE id = ?;
    """, (
        mat["name"],
        mat.get("structure_type", "Flexible Film"),
        mat.get("polymer_code", "Other (#7)"),
        float(mat["nominal_thickness_um"]),
        float(mat["wvtr_g_m2_day"]),
        float(mat["otr_cc_m2_day_atm"]),
        float(mat["co2tr_cc_m2_day_atm"]) if mat.get("co2tr_cc_m2_day_atm") is not None else None,
        mat.get("light_barrier_rating", "Low"),
        float(mat["tensile_strength_mpa"]) if mat.get("tensile_strength_mpa") is not None else None,
        mat.get("seal_integrity_rating", "Good"),
        mat.get("puncture_resistance", "Medium"),
        float(mat["cost_index_usd_per_kg"]),
        mat.get("recyclability_status", "Non-recyclable"),
        float(mat["carbon_footprint_kg_co2_per_kg"]) if mat.get("carbon_footprint_kg_co2_per_kg") is not None else None,
        mat.get("food_contact_compliance", "Pending Verification"),
        float(mat.get("temp_range_min_c", -20.0)),
        float(mat.get("temp_range_max_c", 80.0)),
        mat.get("grease_fat_resistance", "Moderate"),
        mat.get("acid_resistance", "Moderate"),
        mat.get("data_source", "User Literature Input"),
        mat.get("test_conditions", "Standard Lab Conditions"),
        int(mat.get("is_verified_experimental", 0)),
        mat.get("notes", ""),
        material_id
    ))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def delete_material(material_id: int, db_path: str = DB_FILE_PATH) -> bool:
    """Delete a material by its ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM packaging_materials WHERE id = ?;", (material_id,))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def get_all_commodity_presets(db_path: str = DB_FILE_PATH) -> List[Dict[str, Any]]:
    """Retrieve all standard food commodity presets."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM commodity_presets ORDER BY category ASC, name ASC;")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_commodity_preset_by_name(name: str, db_path: str = DB_FILE_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve a commodity preset by its unique name."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM commodity_presets WHERE name = ?;", (name,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def log_recommendation_run(
    commodity_name: str,
    category: str,
    inputs: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, Any]] = None,
    top_recommendations: Optional[List[Dict[str, Any]]] = None,
    db_path: str = DB_FILE_PATH,
    inputs_dict: Optional[Dict[str, Any]] = None,
    weights_dict: Optional[Dict[str, Any]] = None,
    recommendations: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any
) -> int:
    """Store audit log of user recommendation queries, accepting both naming conventions safely."""
    try:
        final_inputs = inputs if inputs is not None else inputs_dict if inputs_dict is not None else {}
        final_weights = weights if weights is not None else weights_dict if weights_dict is not None else {}
        final_recs = top_recommendations if top_recommendations is not None else recommendations if recommendations is not None else []

        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO audit_log (commodity_name, category, inputs_json, weights_json, top_recommendations_json)
        VALUES (?, ?, ?, ?, ?);
        """, (
            commodity_name,
            category,
            json.dumps(final_inputs),
            json.dumps(final_weights),
            json.dumps(final_recs)
        ))
        log_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return log_id if log_id is not None else 0
    except Exception:
        return -1


def get_recommendation_history(limit: int = 50, db_path: str = DB_FILE_PATH) -> List[Dict[str, Any]]:
    """Fetch past recommendation runs from audit log."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?;", (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# Convenience alias for backward compatibility
get_all_commodities = get_all_commodity_presets

