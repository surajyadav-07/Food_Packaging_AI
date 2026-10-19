"""
test_data_loader.py - Tests for CSV export, template generation, and import validation
"""

import pytest
import data_loader
import database


def test_export_and_template():
    """CSV export and template generators must return valid CSV strings with headers."""
    csv_str = data_loader.export_materials_to_csv()
    assert "name,structure_type,polymer_code" in csv_str
    assert len(csv_str) > 200

    template_str = data_loader.get_csv_template()
    assert "Example Custom Co-ex Film" in template_str


def test_import_missing_mandatory_columns():
    """CSV missing required columns must be rejected with informative error."""
    bad_csv = "name,nominal_thickness_um\nTest Film,50.0"
    res = data_loader.validate_and_import_csv(bad_csv)
    assert res["success"] is False
    assert any("Missing mandatory column" in err for err in res["errors"])


def test_import_negative_numeric_values():
    """Negative thickness or WVTR must be caught by validation."""
    invalid_csv = """name,structure_type,polymer_code,nominal_thickness_um,wvtr_g_m2_day,otr_cc_m2_day_atm,cost_index_usd_per_kg,recyclability_status,food_contact_compliance
Negative Thickness Film,Monolayer,PE,-30.0,5.0,100.0,2.5,Widely Recyclable,FDA Approved
"""
    res = data_loader.validate_and_import_csv(invalid_csv)
    assert any("cannot be negative" in err for err in res["errors"])


def test_import_duplicate_name_skipped():
    """Importing a material with a name that already exists in DB must be reported in warnings."""
    existing_name = "LDPE Monolayer Film (50 µm)"
    duplicate_csv = f"""name,structure_type,polymer_code,nominal_thickness_um,wvtr_g_m2_day,otr_cc_m2_day_atm,cost_index_usd_per_kg,recyclability_status,food_contact_compliance
{existing_name},Monolayer,PE,50.0,12.0,6500.0,2.1,Widely Recyclable,FDA Approved
"""
    res = data_loader.validate_and_import_csv(duplicate_csv)
    assert any("already exists" in warn for warn in res["warnings"])
