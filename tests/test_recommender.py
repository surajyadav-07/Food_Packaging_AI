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


def test_missing_compliance_status_knockout():
    """Missing or empty food-contact compliance must fail closed when compliance is required."""
    profile = {
        "name": "General Food",
        "category": "General",
        "storage_temp_c": 20.0,
        "require_food_contact_compliance": True,
    }
    sample_mats = [
        {
            "id": 101,
            "name": "Unspecified Polymer A",
            "food_contact_compliance": None,
            "temp_range_min_c": -10.0,
            "temp_range_max_c": 80.0,
            "nominal_thickness_um": 50.0,
        },
        {
            "id": 102,
            "name": "Unspecified Polymer B",
            "food_contact_compliance": "   ",
            "temp_range_min_c": -10.0,
            "temp_range_max_c": 80.0,
            "nominal_thickness_um": 50.0,
        },
    ]
    qualified, knocked = recommender.filter_knockouts(sample_mats, profile)
    assert len(qualified) == 0
    assert len(knocked) == 2
    for m in knocked:
        assert any("Food-contact compliance status is missing" in r for r in m["knockout_reasons"])

    # When compliance requirement is disabled, materials should not be knocked out for compliance
    profile_no_req = dict(profile, require_food_contact_compliance=False)
    qualified_no_req, _ = recommender.filter_knockouts(sample_mats, profile_no_req)
    assert len(qualified_no_req) == 2


def test_unknown_and_unapproved_compliance_status():
    """Pending, unknown, unapproved, or non-compliant statuses must be rejected."""
    profile = {
        "name": "Dry Goods",
        "category": "General",
        "storage_temp_c": 20.0,
        "require_food_contact_compliance": True,
    }
    unverified_samples = [
        {"name": "Mat Pending", "food_contact_compliance": "Pending FDA review"},
        {"name": "Mat Unapproved", "food_contact_compliance": "Unapproved for direct food contact"},
        {"name": "Mat Unknown", "food_contact_compliance": "Unknown status"},
        {"name": "Mat Not Compliant", "food_contact_compliance": "Non-compliant with EU 10/2011"},
        {"name": "Mat Untested", "food_contact_compliance": "Untested material"},
        {"name": "Mat Arbitrary", "food_contact_compliance": "Some random text string"},
    ]
    for s in unverified_samples:
        s["temp_range_min_c"] = -10.0
        s["temp_range_max_c"] = 80.0
        s["nominal_thickness_um"] = 50.0

    qualified, knocked = recommender.filter_knockouts(unverified_samples, profile)
    assert len(qualified) == 0
    assert len(knocked) == len(unverified_samples)
    for k in knocked:
        assert any("Food-contact compliance status is missing, unapproved" in r for r in k["knockout_reasons"])


def test_valid_compliance_status_acceptance():
    """Affirmative compliance labels and recognized regulatory frameworks must qualify."""
    profile = {
        "name": "Dry Goods",
        "category": "General",
        "storage_temp_c": 20.0,
        "require_food_contact_compliance": True,
    }
    valid_samples = [
        {"name": "Polymer FDA", "food_contact_compliance": "US FDA 21 CFR §177.1520; EU 10/2011"},
        {"name": "Polymer Compliant", "food_contact_compliance": "Compliant"},
        {"name": "Polymer Approved", "food_contact_compliance": "FDA Approved"},
        {"name": "Polymer GRAS", "food_contact_compliance": "US FDA GRAS Notice No. GRN 000178"},
        {"name": "Polymer BfR", "food_contact_compliance": "BfR Recommendation XXXVI"},
        {"name": "Polymer FSSAI", "food_contact_compliance": "FSSAI certified"},
    ]
    for s in valid_samples:
        s["temp_range_min_c"] = -10.0
        s["temp_range_max_c"] = 80.0
        s["nominal_thickness_um"] = 50.0

    qualified, knocked = recommender.filter_knockouts(valid_samples, profile)
    assert len(qualified) == len(valid_samples)
    assert len(knocked) == 0


def test_fresh_produce_synthetic_knockout_behavior():
    """Fresh produce must knock out low-OTR films and require gas transmission."""
    profile = {
        "name": "Fresh Salad Greens",
        "category": "Fruits and vegetables",
        "storage_temp_c": 4.0,
        "target_shelf_life_days": 5,
        "require_food_contact_compliance": True,
    }
    test_mats = [
        {
            "name": "Hermetic Barrier Film",
            "food_contact_compliance": "Compliant",
            "temp_range_min_c": 0.0,
            "temp_range_max_c": 50.0,
            "otr_cc_m2_day_atm": 2.0,  # Below 100 cc/m2/day
            "wvtr_g_m2_day": 1.0,
        },
        {
            "name": "Micro-perforated Breathable Film",
            "food_contact_compliance": "Compliant",
            "temp_range_min_c": 0.0,
            "temp_range_max_c": 50.0,
            "otr_cc_m2_day_atm": 5000.0,
            "wvtr_g_m2_day": 80.0,
        },
        {
            "name": "Permeable Non-perforated Film",
            "food_contact_compliance": "Compliant",
            "temp_range_min_c": 0.0,
            "temp_range_max_c": 50.0,
            "otr_cc_m2_day_atm": 2500.0,
            "wvtr_g_m2_day": 40.0,
        },
        {
            "name": "Missing OTR Film",
            "food_contact_compliance": "Compliant",
            "temp_range_min_c": 0.0,
            "temp_range_max_c": 50.0,
            "otr_cc_m2_day_atm": None,
            "wvtr_g_m2_day": 10.0,
        },
    ]
    qualified, knocked = recommender.filter_knockouts(test_mats, profile)
    qualified_names = [m["name"] for m in qualified]
    knocked_names = [m["name"] for m in knocked]

    assert "Micro-perforated Breathable Film" in qualified_names
    assert "Permeable Non-perforated Film" in qualified_names
    assert "Hermetic Barrier Film" in knocked_names
    assert "Missing OTR Film" in knocked_names


def test_temperature_boundary_rejections():
    """Materials exceeding or falling below thermal tolerance boundaries must be rejected."""
    mats = [
        {
            "name": "Room Temp Polymer",
            "food_contact_compliance": "Compliant",
            "temp_range_min_c": 0.0,
            "temp_range_max_c": 40.0,
        },
        {
            "name": "Deep Freeze Polymer",
            "food_contact_compliance": "Compliant",
            "temp_range_min_c": -50.0,
            "temp_range_max_c": 100.0,
        },
    ]
    # Cold test (-25°C)
    profile_cold = {"storage_temp_c": -25.0, "category": "Frozen Food"}
    q_cold, k_cold = recommender.filter_knockouts(mats, profile_cold)
    assert any("Room Temp" in m["name"] for m in k_cold)
    assert any("Deep Freeze" in m["name"] for m in q_cold)

    # Hot test (80°C)
    profile_hot = {"storage_temp_c": 80.0, "category": "Hot Fill"}
    q_hot, k_hot = recommender.filter_knockouts(mats, profile_hot)
    assert any("Room Temp" in m["name"] for m in k_hot)
    assert any("Deep Freeze" in m["name"] for m in q_hot)


def test_puncture_risk_synthetic_rejection():
    """Commodities with high puncture risk must reject thin or low-puncture materials."""
    profile = {
        "name": "Sharp Dried Bone-in Commodity",
        "category": "Meat",
        "puncture_risk": "High",
        "storage_temp_c": 20.0,
    }
    mats = [
        {
            "name": "Thin Delicate Film (12 µm)",
            "food_contact_compliance": "Compliant",
            "temp_range_min_c": -10.0,
            "temp_range_max_c": 60.0,
            "nominal_thickness_um": 12.0,
            "puncture_resistance": "High",
        },
        {
            "name": "Low Resistance Thick Film (50 µm)",
            "food_contact_compliance": "Compliant",
            "temp_range_min_c": -10.0,
            "temp_range_max_c": 60.0,
            "nominal_thickness_um": 50.0,
            "puncture_resistance": "Low",
        },
        {
            "name": "Robust Heavy Pouch (80 µm)",
            "food_contact_compliance": "Compliant",
            "temp_range_min_c": -10.0,
            "temp_range_max_c": 60.0,
            "nominal_thickness_um": 80.0,
            "puncture_resistance": "High",
        },
    ]
    qualified, knocked = recommender.filter_knockouts(mats, profile)
    assert len(qualified) == 1
    assert qualified[0]["name"] == "Robust Heavy Pouch (80 µm)"
    assert len(knocked) == 2


def test_mechanical_score_contribution():
    """Mechanical properties must influence overall score when mechanical weight > 0."""
    profile = {
        "name": "Dry Food",
        "category": "Bakery products",
        "storage_temp_c": 22.0,
        "puncture_risk": "High",
    }
    mat_strong = {
        "name": "Mat Mechanically Strong",
        "food_contact_compliance": "Compliant",
        "temp_range_min_c": -20.0,
        "temp_range_max_c": 100.0,
        "nominal_thickness_um": 50.0,
        "wvtr_g_m2_day": 5.0,
        "otr_cc_m2_day_atm": 50.0,
        "cost_index_usd_per_kg": 3.0,
        "recyclability_status": "Widely Recyclable (PE Stream)",
        "seal_integrity_rating": "Excellent",
        "puncture_resistance": "High",
        "tensile_strength_mpa": 160.0,
    }
    mat_weak = {
        "name": "Mat Mechanically Moderate",
        "food_contact_compliance": "Compliant",
        "temp_range_min_c": -20.0,
        "temp_range_max_c": 100.0,
        "nominal_thickness_um": 50.0,
        "wvtr_g_m2_day": 5.0,
        "otr_cc_m2_day_atm": 50.0,
        "cost_index_usd_per_kg": 3.0,
        "recyclability_status": "Widely Recyclable (PE Stream)",
        "seal_integrity_rating": "Poor",
        "puncture_resistance": "Moderate",
        "tensile_strength_mpa": None,
    }

    # Weight including mechanical
    mech_weights = {"barrier": 0.25, "cost": 0.25, "sustainability": 0.25, "mechanical": 0.25}
    res = recommender.rank_recommendations([mat_strong, mat_weak], profile, weights=mech_weights)
    assert res["status"] == "SUCCESS"
    top_cand = res["top_candidates"][0]
    second_cand = res["top_candidates"][1]

    assert top_cand["name"] == "Mat Mechanically Strong"
    assert top_cand["mechanical_score"] > second_cand["mechanical_score"]
    assert top_cand["total_mcda_score"] > second_cand["total_mcda_score"]


def test_missing_cost_handling():
    """Missing cost data must not disqualify material, but cost score is None and remaining criteria renormalized."""
    profile = {
        "name": "Dry Snack",
        "category": "Snacks",
        "storage_temp_c": 22.0,
    }
    mat_no_cost = {
        "name": "Mat Without Cost",
        "food_contact_compliance": "Compliant",
        "temp_range_min_c": -20.0,
        "temp_range_max_c": 100.0,
        "nominal_thickness_um": 50.0,
        "wvtr_g_m2_day": 2.0,
        "otr_cc_m2_day_atm": 20.0,
        "cost_index_usd_per_kg": None,
        "recyclability_status": "Widely Recyclable (PE Stream)",
    }
    mat_with_cost = {
        "name": "Mat With Cost",
        "food_contact_compliance": "Compliant",
        "temp_range_min_c": -20.0,
        "temp_range_max_c": 100.0,
        "nominal_thickness_um": 50.0,
        "wvtr_g_m2_day": 2.0,
        "otr_cc_m2_day_atm": 20.0,
        "cost_index_usd_per_kg": 3.5,
        "recyclability_status": "Widely Recyclable (PE Stream)",
    }

    res = recommender.rank_recommendations([mat_no_cost, mat_with_cost], profile)
    assert res["status"] == "SUCCESS"
    candidates = {c["name"]: c for c in res["all_ranked"]}

    c_no_cost = candidates["Mat Without Cost"]
    assert c_no_cost["cost_score"] is None
    assert "cost" in c_no_cost["missing_score_criteria"]
    assert c_no_cost["total_mcda_score"] > 0
    assert any("Cost index is missing" in m for m in c_no_cost["explanation"]["missing_data"])

    c_with_cost = candidates["Mat With Cost"]
    assert c_with_cost["cost_score"] is not None
    assert "cost" not in c_with_cost["missing_score_criteria"]


def test_three_weight_configuration_compatibility():
    """Existing 3-weight dictionaries (barrier, cost, sustainability) must work seamlessly and normalize safely."""
    profile = {
        "name": "Bakery",
        "category": "Bakery products",
        "storage_temp_c": 20.0,
    }
    mat = {
        "name": "Sample Film",
        "food_contact_compliance": "Compliant",
        "temp_range_min_c": -10.0,
        "temp_range_max_c": 90.0,
        "nominal_thickness_um": 40.0,
        "wvtr_g_m2_day": 4.0,
        "otr_cc_m2_day_atm": 100.0,
        "cost_index_usd_per_kg": 2.5,
        "recyclability_status": "Widely Recyclable (PE Stream)",
    }

    three_weights = {"barrier": 0.45, "cost": 0.25, "sustainability": 0.30}
    res = recommender.rank_recommendations([mat], profile, weights=three_weights)
    assert res["status"] == "SUCCESS"
    weights_used = res["weights_used"]
    assert round(sum(weights_used.values()), 4) == 1.0
    assert weights_used["barrier"] == 0.45
    assert weights_used["cost"] == 0.25
    assert weights_used["sustainability"] == 0.30
    assert weights_used["mechanical"] == 0.0


def test_empty_candidate_results_safety():
    """Impossible profile conditions must safely yield NO_QUALIFIED_MATERIALS without IndexError or crashing."""
    profile = {
        "name": "Superheated Food",
        "category": "Snacks",
        "storage_temp_c": 999.0,  # Impossible temperature
    }
    mat = {
        "name": "Standard Film",
        "food_contact_compliance": "Compliant",
        "temp_range_min_c": -20.0,
        "temp_range_max_c": 80.0,
        "nominal_thickness_um": 30.0,
    }
    result = recommender.rank_recommendations([mat], profile)
    assert result["status"] == "NO_QUALIFIED_MATERIALS"
    assert result["top_candidates"] == []
    assert result["all_ranked"] == []
    assert len(result["knocked_out"]) == 1
    assert "message" in result
    assert result["total_qualified"] == 0
    # Verify iteration over empty candidates causes no error
    names = [c["name"] for c in result["top_candidates"]]
    assert names == []


def test_explanation_accuracy_and_missing_data_warnings():
    """Explanations must honestly report unmeasured properties and avoid falsely claiming certification."""
    profile = {
        "name": "Dry Product",
        "category": "General",
        "storage_temp_c": 20.0,
    }
    mat = {
        "name": "Literature Pouch",
        "food_contact_compliance": "US FDA 21 CFR §177.1520",
        "temp_range_min_c": -10.0,
        "temp_range_max_c": 80.0,
        "nominal_thickness_um": 40.0,
        "wvtr_g_m2_day": 3.0,
        "otr_cc_m2_day_atm": 50.0,
        "cost_index_usd_per_kg": 3.0,
        "recyclability_status": "Widely Recyclable (PE Stream)",
        "co2tr_cc_m2_day_atm": None,
        "tensile_strength_mpa": None,
        "carbon_footprint_kg_co2_per_kg": None,
        "is_verified_experimental": 0,
    }
    expl = recommender.generate_explanation(mat, profile)
    assert len(expl["requirements_met"]) > 0
    assert len(expl["missing_data"]) >= 3
    # Verify honest wording without false certification claim
    assert any("This label alone does not establish independent regulatory verification" in r for r in expl["requirements_met"])
    assert any("Carbon dioxide transmission rate (CO2TR) is not documented" in m for m in expl["missing_data"])
    assert any("Tensile strength is not documented" in m for m in expl["missing_data"])
    assert any("carbon footprint data is unavailable" in m for m in expl["missing_data"])
