"""
test_pdf_generator.py - Unit tests for ReportLab PDF report generation
"""

import pytest
import database
import recommender
import pdf_generator


@pytest.fixture
def sample_data():
    mats = database.get_all_materials()
    comm = database.get_commodity_preset_by_name("Fried Potato Chips & Crisps")
    weights = {"barrier": 0.45, "cost": 0.25, "sustainability": 0.30}
    res = recommender.rank_recommendations(mats, comm, weights)
    return comm, res, weights


def test_pdf_generation_returns_valid_bytes(sample_data):
    """PDF generator must produce non-empty bytes with valid PDF header."""
    comm, res, weights = sample_data
    pdf_bytes = pdf_generator.generate_recommendation_pdf(comm, res, weights)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF")


def test_pdf_generation_empty_candidates():
    """PDF generator must handle empty candidates gracefully without crashing."""
    comm = {
        "name": "Hypothetical Impossible Food",
        "category": "Snacks",
        "moisture_sensitivity": "Extreme",
        "oxygen_sensitivity": "Critical",
        "storage_temp_c": 200.0,  # Impossible temp will knock out all materials
        "target_shelf_life_days": 365
    }
    weights = {"barrier": 0.5, "cost": 0.25, "sustainability": 0.25}
    mats = database.get_all_materials()
    res = recommender.rank_recommendations(mats, comm, weights)

    pdf_bytes = pdf_generator.generate_recommendation_pdf(comm, res, weights)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
