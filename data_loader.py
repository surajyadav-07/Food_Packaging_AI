"""
data_loader.py - Data Ingestion, CSV Import/Export, and Schema Validation
Handles importing external datasets, exporting knowledge base records,
and validating input data integrity without fabricating unmeasured data.
"""

import io
from typing import List, Dict, Any, Tuple
import pandas as pd
from database import get_db_connection, add_material, get_all_materials_df

REQUIRED_CSV_COLUMNS = [
    "name",
    "structure_type",
    "polymer_code",
    "nominal_thickness_um",
    "wvtr_g_m2_day",
    "otr_cc_m2_day_atm",
    "cost_index_usd_per_kg",
    "recyclability_status",
    "food_contact_compliance"
]

NUMERIC_POSITIVE_FIELDS = [
    "nominal_thickness_um",
    "wvtr_g_m2_day",
    "otr_cc_m2_day_atm",
    "cost_index_usd_per_kg"
]


def export_materials_to_csv() -> str:
    """Exports all current database materials to a CSV string."""
    df = get_all_materials_df()
    return df.to_csv(index=False)


def get_csv_template() -> str:
    """Provides a blank sample CSV template with standard column headers and instructions."""
    template_data = {
        "name": ["Example Custom Co-ex Film (40 µm)"],
        "structure_type": ["Co-extruded Flexible"],
        "polymer_code": ["PP/EVOH/PE (#7)"],
        "nominal_thickness_um": [40.0],
        "wvtr_g_m2_day": [2.5],
        "otr_cc_m2_day_atm": [3.0],
        "co2tr_cc_m2_day_atm": [12.0],
        "light_barrier_rating": ["Low (Clear)"],
        "tensile_strength_mpa": [55.0],
        "seal_integrity_rating": ["Good"],
        "puncture_resistance": ["Medium"],
        "cost_index_usd_per_kg": [3.80],
        "recyclability_status": ["Multi-material Non-recyclable"],
        "carbon_footprint_kg_co2_per_kg": [3.10],
        "food_contact_compliance": ["US FDA 21 CFR §177.1520"],
        "temp_range_min_c": [-20.0],
        "temp_range_max_c": [95.0],
        "grease_fat_resistance": ["High"],
        "acid_resistance": ["High"],
        "data_source": ["Laboratory Pilot Line Trials 2026"],
        "test_conditions": ["WVTR: ASTM F1249 (38°C/90% RH); OTR: ASTM D3985 (23°C/0% RH)"],
        "is_verified_experimental": [1],
        "notes": ["Sample batch for college trial testing."]
    }
    df = pd.DataFrame(template_data)
    return df.to_csv(index=False)


def validate_and_import_csv(csv_content: str) -> Dict[str, Any]:
    """
    Parses and validates an uploaded CSV file.
    Ensures required fields exist, numeric ranges are valid,
    and returns a summary of successes and specific validation errors.
    """
    try:
        df = pd.read_csv(io.StringIO(csv_content))
    except Exception as e:
        return {
            "success": False,
            "imported_count": 0,
            "errors": [f"Malformed CSV file format: {str(e)}"],
            "warnings": []
        }

    # Check for required columns
    missing_cols = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
    if missing_cols:
        return {
            "success": False,
            "imported_count": 0,
            "errors": [f"Missing mandatory column(s): {', '.join(missing_cols)}"],
            "warnings": []
        }

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM packaging_materials;")
    existing_names = {row["name"].strip().lower() for row in cursor.fetchall()}
    conn.close()

    imported_count = 0
    errors = []
    warnings = []

    for idx, row in df.iterrows():
        row_num = idx + 2  # Accounting for 1-based index + header
        raw_name = str(row.get("name", "")).strip()

        if not raw_name or raw_name.lower() == "nan":
            errors.append(f"Row {row_num}: Material name is empty.")
            continue

        if raw_name.lower() in existing_names:
            warnings.append(f"Row {row_num}: '{raw_name}' already exists in database; skipped to prevent duplicates.")
            continue

        # Validate numeric positive fields
        invalid_numeric = False
        row_dict = row.to_dict()

        for field in NUMERIC_POSITIVE_FIELDS:
            val = row_dict.get(field)
            try:
                val_float = float(val)
                if val_float < 0:
                    errors.append(f"Row {row_num}: '{field}' cannot be negative ({val_float}).")
                    invalid_numeric = True
                    break
                row_dict[field] = val_float
            except (ValueError, TypeError):
                errors.append(f"Row {row_num}: '{field}' must be a valid positive numeric value (got '{val}').")
                invalid_numeric = True
                break

        if invalid_numeric:
            continue

        # Optional numeric fields
        for opt_field in ["co2tr_cc_m2_day_atm", "tensile_strength_mpa", "carbon_footprint_kg_co2_per_kg"]:
            val = row_dict.get(opt_field)
            if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
                row_dict[opt_field] = None
            else:
                try:
                    row_dict[opt_field] = float(val)
                except ValueError:
                    row_dict[opt_field] = None

        # Clean string fields
        for str_field in ["structure_type", "polymer_code", "light_barrier_rating",
                          "seal_integrity_rating", "puncture_resistance", "recyclability_status",
                          "food_contact_compliance", "grease_fat_resistance", "acid_resistance",
                          "data_source", "test_conditions", "notes"]:
            val = row_dict.get(str_field)
            row_dict[str_field] = str(val) if not pd.isna(val) else ""

        try:
            add_material(row_dict)
            existing_names.add(raw_name.lower())
            imported_count += 1
        except Exception as e:
            errors.append(f"Row {row_num}: Database insertion error for '{raw_name}': {str(e)}")

    return {
        "success": imported_count > 0 or len(errors) == 0,
        "imported_count": imported_count,
        "errors": errors,
        "warnings": warnings
    }
