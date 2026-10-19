"""
test_recommender.py - Tests for filtering, knockout logic, MCDA ranking, and explainability
"""

import pytest
import database
import recommender


@pytest.fixture
def materials():
    """Returns standard default materials from database."""
    return database.get_all_materials()


def test_empty_materials_list():
    """Engine must gracefully handle an empty materials dataset."""
    profile = {
        "category": "Snacks",
        "moisture_sensitivity": "High",
        "oxygen_sensitivity": "High",
        "storage_temp_c": 23.0
    }
    result = recommender.rank_recommendations([], profile)
    assert result["status"] == "NO_QUALIFIED_MATERIALS"
    assert len(result["top_candidates"]) == 0


def test_fresh_produce_knockout(materials):
    """Hermetic high-barrier films must be knocked out for fresh produce."""
    profile = {
        "name": "Fresh Strawberries",
        "category": "Fruits and vegetables",
        "moisture_sensitivity": "Moderate",
        "oxygen_sensitivity": "Moderate",
        "fat_content_level": "Zero",
        "storage_temp_c": 4.0,
        "target_shelf_life_days": 7
    }
    result = recommender.rank_recommendations(materials, profile)
    assert result["status"] == "SUCCESS"
    assert len(result["top_candidates"]) > 0

    # Ensure hermetic foil is knocked out
    knocked_names = [m["name"] for m in result["knocked_out"]]
    assert any("Foil" in name for name in knocked_names)

    # Top recommendation should be breathable / micro-perforated film
    top_1 = result["top_candidates"][0]
    assert "Micro-perforated" in top_1["name"] or "Paperboard" in top_1["name"] or "LDPE" in top_1["name"]


def test_extreme_temperature_knockout(materials):
    """Storage temperatures exceeding polymer ratings must trigger knockout."""
    profile = {
        "name": "Deep-Freeze Meat / Seafood",
        "category": "Dairy products",
        "moisture_sensitivity": "High",
        "oxygen_sensitivity": "High",
        "storage_temp_c": -35.0,  # Below PLA's -10°C limit
        "target_shelf_life_days": 180
    }
    result = recommender.rank_recommendations(materials, profile)
    knocked_names = [m["name"] for m in result["knocked_out"]]
    assert any("PLA" in name for name in knocked_names)


def test_puncture_risk_knockout(materials):
    """Sharp food commodities must knock out low-puncture thin films."""
    profile = {
        "name": "Hard Pulses",
        "category": "Pulses",
        "moisture_sensitivity": "Moderate",
        "oxygen_sensitivity": "Low",
        "puncture_risk": "Very High",
        "storage_temp_c": 22.0,
        "target_shelf_life_days": 365
    }
    result = recommender.rank_recommendations(materials, profile)
    for cand in result["top_candidates"]:
        punc = str(cand.get("puncture_resistance", "")).lower()
        assert punc != "low"


def test_weights_impact_on_ranking(materials):
    """Altering user preference weights between cost and sustainability must affect ranks."""
    profile = {
        "name": "Dry Biscuits",
        "category": "Bakery products",
        "moisture_sensitivity": "High",
        "oxygen_sensitivity": "Moderate",
        "storage_temp_c": 24.0,
        "target_shelf_life_days": 180
    }
    # Extreme sustainability weighting
    sust_weights = {"barrier": 0.20, "cost": 0.05, "sustainability": 0.75}
    res_sust = recommender.rank_recommendations(materials, profile, weights=sust_weights)

    # Extreme cost weighting
    cost_weights = {"barrier": 0.20, "cost": 0.75, "sustainability": 0.05}
    res_cost = recommender.rank_recommendations(materials, profile, weights=cost_weights)

    assert res_sust["status"] == "SUCCESS"
    assert res_cost["status"] == "SUCCESS"
    top_sust = res_sust["top_candidates"][0]
    top_cost = res_cost["top_candidates"][0]

    # Sustainability score of top_sust should be >= top_cost sustainability score
    assert top_sust["sustainability_score"] >= top_cost["sustainability_score"]
    # Cost score of top_cost should be >= top_sust cost score
    assert top_cost["cost_score"] >= top_sust["cost_score"]


def test_explainability_payload(materials):
    """Candidates must contain explanation with satisfied requirements, limitations, and missing data."""
    profile = {
        "name": "Fried Potato Chips",
        "category": "Snacks",
        "moisture_sensitivity": "Extreme",
        "oxygen_sensitivity": "Critical",
        "fat_content_level": "High (30% fat)",
        "light_sensitivity": "High",
        "storage_temp_c": 23.0,
        "target_shelf_life_days": 180
    }
    result = recommender.rank_recommendations(materials, profile)
    top_candidate = result["top_candidates"][0]

    assert "explanation" in top_candidate
    expl = top_candidate["explanation"]
    assert "rationale" in expl
    assert len(expl["requirements_met"]) > 0
    assert len(expl["limitations"]) > 0
    assert len(expl["missing_data"]) > 0
