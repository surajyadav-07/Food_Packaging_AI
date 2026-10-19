"""
test_database.py - Unit tests for SQLite database operations
"""

import os
import tempfile
import pytest
import sqlite3
import database


@pytest.fixture
def temp_db():
    """Creates an isolated temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database.init_database(db_path=path, force_reset=True)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_database_initialization(temp_db):
    """Test that schema is initialized and seeded with default materials and presets."""
    mats = database.get_all_materials(db_path=temp_db)
    assert len(mats) >= 15
    presets = database.get_all_commodity_presets(db_path=temp_db)
    assert len(presets) >= 10


def test_add_and_delete_material(temp_db):
    """Test adding a custom material, retrieving it, and deleting it."""
    test_mat = {
        "name": "Custom Test Nanocomposite Barrier Film",
        "structure_type": "Nanocomposite Flexible",
        "polymer_code": "PLA/Nano-clay (#7)",
        "nominal_thickness_um": 25.0,
        "wvtr_g_m2_day": 1.2,
        "otr_cc_m2_day_atm": 4.5,
        "co2tr_cc_m2_day_atm": 18.0,
        "light_barrier_rating": "Moderate",
        "tensile_strength_mpa": 85.0,
        "seal_integrity_rating": "Good",
        "puncture_resistance": "High",
        "cost_index_usd_per_kg": 4.50,
        "recyclability_status": "Industrially Compostable",
        "carbon_footprint_kg_co2_per_kg": 2.10,
        "food_contact_compliance": "US FDA GRAS compliant",
        "temp_range_min_c": -15.0,
        "temp_range_max_c": 60.0,
        "grease_fat_resistance": "High",
        "acid_resistance": "Moderate",
        "data_source": "Lab Trial 2026",
        "test_conditions": "ASTM F1249; ASTM D3985",
        "is_verified_experimental": 1,
        "notes": "Testing addition"
    }

    new_id = database.add_material(test_mat, db_path=temp_db)
    assert new_id > 0

    fetched = database.get_material_by_id(new_id, db_path=temp_db)
    assert fetched is not None
    assert fetched["name"] == "Custom Test Nanocomposite Barrier Film"
    assert fetched["nominal_thickness_um"] == 25.0

    # Test Update material
    test_mat["nominal_thickness_um"] = 30.0
    test_mat["cost_index_usd_per_kg"] = 5.00
    updated = database.update_material(new_id, test_mat, db_path=temp_db)
    assert updated is True

    fetched_updated = database.get_material_by_id(new_id, db_path=temp_db)
    assert fetched_updated["nominal_thickness_um"] == 30.0
    assert fetched_updated["cost_index_usd_per_kg"] == 5.00

    # Delete material
    deleted = database.delete_material(new_id, db_path=temp_db)
    assert deleted is True
    assert database.get_material_by_id(new_id, db_path=temp_db) is None


def test_audit_log_operations(temp_db):
    """Test audit log recording and retrieval."""
    inputs = {"moisture_sensitivity": "High", "oxygen_sensitivity": "Critical"}
    weights = {"barrier": 0.5, "cost": 0.25, "sustainability": 0.25}
    top = [{"rank": 1, "name": "Aluminium Foil Laminate"}]

    log_id = database.log_recommendation_run(
        commodity_name="Test Dry Powder",
        category="Dairy products",
        inputs=inputs,
        weights=weights,
        top_recommendations=top,
        db_path=temp_db
    )
    assert log_id > 0

    history = database.get_recommendation_history(limit=10, db_path=temp_db)
    assert len(history) >= 1
    assert history[0]["commodity_name"] == "Test Dry Powder"
