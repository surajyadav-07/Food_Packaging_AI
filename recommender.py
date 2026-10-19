"""
recommender.py - Transparent Multi-Criteria Recommendation Engine
Implements:
1. Stage-1 Hard Knockout Constraints (Physical, Thermal, Biological & Food Safety Feasibility)
2. Stage-2 Multi-Criteria Decision Analysis (MCDA) Scoring (Barrier, Cost, Sustainability, Mechanical)
3. Transparent Explainability Engine with Requirements Met, Trade-offs, and Missing Data Disclosures.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


# Recyclability baseline scores (0-100 scale) based on circular packaging literature
RECYCLABILITY_SCORE_MAP = {
    "Widely Recyclable (PE Stream)": 95.0,
    "Widely Recyclable (HDPE Stream)": 95.0,
    "Widely Recyclable (PP Stream)": 90.0,
    "Widely Recyclable (PET Bottle Stream)": 95.0,
    "Infinitely Recyclable (Curbside Glass Container Stream)": 98.0,
    "Recyclable in Paper Pulping Stream (Specialized Mill)": 80.0,
    "Technically Recyclable (Specialized PP Stream)": 65.0,
    "Industrially Compostable (ASTM D6400 / EN 13432)": 75.0,
    "Home & Industrially Compostable (TÜV OK Compost)": 85.0,
    "Multi-material Laminate (Difficult Mechanical Recycling)": 30.0,
    "Multi-material Non-recyclable (Curbside Incompatible)": 20.0,
    "Multi-material Non-recyclable": 20.0,
    "Non-recyclable Landfill / Waste-to-Energy": 10.0,
}


def filter_knockouts(
    materials: List[Dict[str, Any]],
    commodity_profile: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Applies scientifically grounded Stage-1 elimination constraints:
    - Temperature survivability
    - Active respiration / breathability for fresh produce
    - Puncture resistance for sharp commodities
    - Extreme moisture sensitivity vs high-permeability materials
    - Grease / fat resistance for lipid-rich foods
    - Food contact compliance check

    Returns:
        (qualified_materials, knocked_out_materials_with_reasons)
    """
    qualified = []
    knocked_out = []

    storage_temp = float(commodity_profile.get("storage_temp_c", 23.0))
    category = str(commodity_profile.get("category", "")).lower()
    moisture_sens = str(commodity_profile.get("moisture_sensitivity", "Moderate")).lower()
    oxygen_sens = str(commodity_profile.get("oxygen_sensitivity", "Moderate")).lower()
    fat_content = str(commodity_profile.get("fat_content_level", "Low")).lower()
    puncture_risk = str(commodity_profile.get("puncture_risk", "Low")).lower()
    target_shelf_life = int(commodity_profile.get("target_shelf_life_days", 90))
    require_compliance = bool(commodity_profile.get("require_food_contact_compliance", True))

    is_fresh_produce = "fruit" in category or "vegetable" in category or "fresh cut" in category

    for mat in materials:
        reasons = []

        # 1. Thermal tolerance check
        t_min = mat.get("temp_range_min_c", -40.0)
        t_max = mat.get("temp_range_max_c", 100.0)
        if storage_temp < t_min:
            reasons.append(f"Storage temperature ({storage_temp}°C) is below minimum safe rating ({t_min}°C); risk of cold embrittlement / seal shattering.")
        if storage_temp > t_max:
            reasons.append(f"Storage temperature ({storage_temp}°C) exceeds maximum safe rating ({t_max}°C); risk of polymer thermal deformation / seal creep.")

        # 2. Fresh Produce / Active Respiration Rule
        # Fresh respiring produce requires gas transmission or micro-perforations to avoid anaerobic spoilage.
        # Absolute zero-barrier foils or impermeable multi-layer laminates without ventilation are unsafe.
        if is_fresh_produce:
            mat_name = mat.get("name", "").lower()
            otr = mat.get("otr_cc_m2_day_atm", 100.0)
            if "micro-perforated" not in mat_name and otr < 100.0:
                reasons.append("Hermetic high-barrier film is hazardous for fresh respiring produce; triggers anaerobic respiration, off-flavors, and pathogen proliferation.")

        # 3. Puncture resistance for sharp commodities
        if puncture_risk in ["high", "very high"]:
            punc_res = str(mat.get("puncture_resistance", "Medium")).lower()
            thick = float(mat.get("nominal_thickness_um", 30.0))
            if punc_res == "low" or thick < 20.0:
                reasons.append(f"Inadequate puncture resistance ({mat.get('puncture_resistance')}, {thick} µm) for sharp-edged food; high risk of pinholing and seal breach.")

        # 4. Extreme moisture sensitivity + long shelf-life knockout
        if "extreme" in moisture_sens and target_shelf_life >= 60:
            wvtr = float(mat.get("wvtr_g_m2_day", 10.0))
            if wvtr > 35.0:
                reasons.append(f"Water vapor transmission rate ({wvtr} g/m²/day) is excessively high for extreme moisture-sensitive commodity requiring {target_shelf_life} days.")

        # 5. Critical oxygen sensitivity + long shelf-life knockout
        if "critical" in oxygen_sens and target_shelf_life >= 90:
            otr = float(mat.get("otr_cc_m2_day_atm", 1000.0))
            if otr > 3000.0:
                reasons.append(f"Oxygen transmission rate ({otr} cc/m²/day) is too permeable for critical lipid-oxidation prone commodity across {target_shelf_life} days.")

        # 6. High fat / lipid resistance
        if ("high" in fat_content or "very high" in fat_content) and "zero" not in fat_content:
            grease_res = str(mat.get("grease_fat_resistance", "Moderate")).lower()
            if grease_res in ["low", "poor"]:
                reasons.append(f"Inadequate grease/fat resistance ({mat.get('grease_fat_resistance')}); fatty acids may plasticize or delaminate film.")

        # 7. Food contact compliance check
        if require_compliance:
            compliance = str(mat.get("food_contact_compliance", "")).lower()
            if "pending" in compliance or "unapproved" in compliance:
                reasons.append("Lacks certified food contact compliance evidence (FDA / EU / national standard).")

        if reasons:
            knocked_copy = dict(mat)
            knocked_copy["knockout_reasons"] = reasons
            knocked_out.append(knocked_copy)
        else:
            qualified.append(mat)

    return qualified, knocked_out


def compute_barrier_score(
    material: Dict[str, Any],
    commodity_profile: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes a transparent 0-100 barrier protection utility score:
    Evaluates WVTR fitness and OTR fitness according to food degradation sensitivity.
    """
    wvtr = float(material.get("wvtr_g_m2_day", 10.0))
    otr = float(material.get("otr_cc_m2_day_atm", 1000.0))
    category = str(commodity_profile.get("category", "")).lower()
    is_fresh = "fruit" in category or "vegetable" in category

    moisture_sens = str(commodity_profile.get("moisture_sensitivity", "Moderate")).lower()
    oxygen_sens = str(commodity_profile.get("oxygen_sensitivity", "Moderate")).lower()
    light_sens = str(commodity_profile.get("light_sensitivity", "Low")).lower()

    # Target thresholds based on food science benchmarks
    if "extreme" in moisture_sens:
        target_wvtr = 1.0
    elif "high" in moisture_sens:
        target_wvtr = 5.0
    elif "moderate" in moisture_sens:
        target_wvtr = 15.0
    else:
        target_wvtr = 50.0

    if "critical" in oxygen_sens:
        target_otr = 15.0
    elif "high" in oxygen_sens:
        target_otr = 100.0
    elif "moderate" in oxygen_sens:
        target_otr = 1500.0
    else:
        target_otr = 7000.0

    # For fresh respiring produce: balanced EMAP transmission is rewarded, hermetic zero barrier is penalized
    if is_fresh:
        if 1000.0 <= otr <= 10000.0 and wvtr >= 30.0:
            barrier_score = 95.0
        elif otr > 10000.0:
            barrier_score = 80.0
        else:
            barrier_score = 45.0
        wvtr_fit = 90.0 if wvtr >= 30.0 else 50.0
        otr_fit = 95.0 if 1000.0 <= otr <= 10000.0 else 40.0
    else:
        # Standard foods: lower transmission is better or equal to target
        # WVTR score
        if wvtr <= target_wvtr:
            wvtr_fit = 100.0
        else:
            ratio = target_wvtr / (wvtr + 1e-6)
            wvtr_fit = max(10.0, 100.0 * float(np.sqrt(np.clip(ratio, 0.0, 1.0))))

        # OTR score
        if otr <= target_otr:
            otr_fit = 100.0
        else:
            ratio = target_otr / (otr + 1e-6)
            otr_fit = max(10.0, 100.0 * float(np.sqrt(np.clip(ratio, 0.0, 1.0))))

        barrier_score = 0.55 * wvtr_fit + 0.45 * otr_fit

    # Light barrier adjustment
    light_rating = str(material.get("light_barrier_rating", "Low")).lower()
    if "critical" in light_sens or "high" in light_sens:
        if "opaque" in light_rating or "total" in light_rating or "high" in light_rating:
            barrier_score = min(100.0, barrier_score + 5.0)
        else:
            barrier_score = max(15.0, barrier_score - 15.0)

    details = {
        "wvtr_actual": wvtr,
        "wvtr_target": target_wvtr,
        "wvtr_fit_score": round(wvtr_fit, 1),
        "otr_actual": otr,
        "otr_target": target_otr,
        "otr_fit_score": round(otr_fit, 1),
        "barrier_overall": round(barrier_score, 1)
    }
    return float(np.clip(barrier_score, 0.0, 100.0)), details


def compute_sustainability_score(material: Dict[str, Any]) -> float:
    """
    Computes sustainability utility score (0-100) based on recyclability tier
    and cradle-to-gate carbon footprint index.
    """
    recyclability_status = material.get("recyclability_status", "")
    base_score = RECYCLABILITY_SCORE_MAP.get(recyclability_status, 40.0)

    # Carbon footprint adjustment (lower footprint gets slight bonus)
    co2 = material.get("carbon_footprint_kg_co2_per_kg")
    if co2 is not None:
        co2_val = float(co2)
        if co2_val <= 1.5:
            base_score += 5.0
        elif co2_val > 4.5:
            base_score -= 8.0

    return float(np.clip(base_score, 5.0, 100.0))


def compute_mechanical_score(material: Dict[str, Any], commodity_profile: Dict[str, Any]) -> float:
    """
    Evaluates tensile strength, seal integrity, and puncture tolerance.
    """
    score = 70.0
    seal = str(material.get("seal_integrity_rating", "Good")).lower()
    if "excellent" in seal:
        score += 15.0
    elif "good" in seal:
        score += 8.0
    elif "fair" in seal:
        score -= 10.0

    punc = str(material.get("puncture_resistance", "Medium")).lower()
    punc_req = str(commodity_profile.get("puncture_risk", "Low")).lower()
    if punc_req in ["high", "very high"]:
        if punc in ["very high", "high", "total"]:
            score += 15.0
        else:
            score -= 20.0

    tensile = material.get("tensile_strength_mpa")
    if tensile is not None and float(tensile) >= 100.0:
        score += 5.0

    return float(np.clip(score, 10.0, 100.0))


def rank_recommendations(
    materials: List[Dict[str, Any]],
    commodity_profile: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Executes complete recommendation pipeline:
    1. Filter knockouts
    2. Normalize cost index
    3. Multi-criteria utility scoring
    4. Top candidates extraction & explainability breakdown
    """
    if weights is None:
        weights = {"barrier": 0.45, "cost": 0.25, "sustainability": 0.30}

    # Normalize user weights to sum to 1.0
    w_sum = sum(weights.values()) or 1.0
    w_barrier = weights.get("barrier", 0.45) / w_sum
    w_cost = weights.get("cost", 0.25) / w_sum
    w_sust = weights.get("sustainability", 0.30) / w_sum

    qualified_mats, knocked_out_mats = filter_knockouts(materials, commodity_profile)

    if not qualified_mats:
        return {
            "top_candidates": [],
            "all_ranked": [],
            "knocked_out": knocked_out_mats,
            "weights_used": {"barrier": w_barrier, "cost": w_cost, "sustainability": w_sust},
            "status": "NO_QUALIFIED_MATERIALS",
            "message": "All evaluated materials were disqualified by hard physical, biological, or safety constraints."
        }

    # Determine cost bounds for normalization among qualified candidates
    costs = [float(m.get("cost_index_usd_per_kg", 3.0)) for m in qualified_mats]
    min_cost = min(costs)
    max_cost = max(costs)
    cost_range = max_cost - min_cost if max_cost > min_cost else 1.0

    scored_candidates = []
    for mat in qualified_mats:
        b_score, b_details = compute_barrier_score(mat, commodity_profile)
        s_score = compute_sustainability_score(mat)
        m_score = compute_mechanical_score(mat, commodity_profile)

        # Cost utility: Lower cost = higher score (0-100)
        c_actual = float(mat.get("cost_index_usd_per_kg", 3.0))
        c_score = 100.0 * (1.0 - ((c_actual - min_cost) / cost_range))

        # Composite MCDA Total Utility Score
        total_score = (
            w_barrier * b_score +
            w_cost * c_score +
            w_sust * s_score
        )

        candidate = dict(mat)
        candidate["barrier_score"] = round(b_score, 1)
        candidate["cost_score"] = round(c_score, 1)
        candidate["sustainability_score"] = round(s_score, 1)
        candidate["mechanical_score"] = round(m_score, 1)
        candidate["total_mcda_score"] = round(total_score, 1)
        candidate["barrier_details"] = b_details

        # Generate transparent explainability payload
        candidate["explanation"] = generate_explanation(candidate, commodity_profile)

        scored_candidates.append(candidate)

    # Sort descending by total score
    scored_candidates.sort(key=lambda x: x["total_mcda_score"], reverse=True)

    # Assign rank numbers
    for i, c in enumerate(scored_candidates):
        c["rank"] = i + 1

    top_candidates = scored_candidates[:3]

    return {
        "top_candidates": top_candidates,
        "all_ranked": scored_candidates,
        "knocked_out": knocked_out_mats,
        "weights_used": {"barrier": w_barrier, "cost": w_cost, "sustainability": w_sust},
        "status": "SUCCESS",
        "total_evaluated": len(materials),
        "total_qualified": len(qualified_mats),
        "total_knocked_out": len(knocked_out_mats)
    }


def generate_explanation(candidate: Dict[str, Any], commodity_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs a scientifically clear justification, satisfied requirements,
    known technical limitations, and unmeasured data items.
    """
    name = candidate.get("name", "Material")
    b_score = candidate.get("barrier_score", 0.0)
    wvtr = candidate.get("wvtr_g_m2_day")
    otr = candidate.get("otr_cc_m2_day_atm")
    cost = candidate.get("cost_index_usd_per_kg")
    recyclability = candidate.get("recyclability_status", "Unspecified")

    requirements_met = []
    limitations = []
    missing_data = []

    # Satisfied requirements
    requirements_met.append(f"Water vapor transmission: {wvtr} g/m²/day under 38°C/90% RH standard test conditions.")
    requirements_met.append(f"Oxygen transmission rate: {otr} cc/m²/day/atm under 23°C/0% RH.")
    requirements_met.append(f"Food contact compliance: {candidate.get('food_contact_compliance', 'Compliant')}")
    requirements_met.append(f"Thermal tolerance verified across range: {candidate.get('temp_range_min_c')}°C to {candidate.get('temp_range_max_c')}°C.")

    # Technical limitations & sustainability trade-offs
    if "non-recyclable" in recyclability.lower() or "laminate" in recyclability.lower():
        limitations.append("Multi-material composite structure inhibits mechanical reprocessing in standard domestic curbside recycling streams.")
    if cost > 4.0:
        limitations.append(f"Higher commercial cost index (${cost:.2f}/kg); represents a premium packaging tier.")
    if "pla" in name.lower() or "bio" in name.lower():
        limitations.append("High sensitivity to moisture and low thermal distortion temperature (<55°C); requires climate-controlled storage.")
    if "glass" in name.lower():
        limitations.append("High gross tare weight significantly elevates transportation fuel consumption and carbon footprint.")
    if "ldpe" in name.lower() and candidate.get("otr_cc_m2_day_atm", 0) > 4000:
        limitations.append("High oxygen permeability makes monolayer film unsuitable for extended oxidation-sensitive storage.")

    # Missing / unmeasured data checks (Transparent science)
    if candidate.get("co2tr_cc_m2_day_atm") is None:
        missing_data.append("Carbon dioxide transmission rate (CO2TR) not empirically documented in current reference entry.")
    if candidate.get("tensile_strength_mpa") is None:
        missing_data.append("Tensile yield strength value not reported; mechanical safety factor cannot be fully verified.")
    if candidate.get("carbon_footprint_kg_co2_per_kg") is None:
        missing_data.append("Complete cradle-to-gate Life Cycle Assessment (LCA) carbon footprint data unavailable.")

    selection_rationale = (
        f"{name} achieves a composite MCDA utility score of {candidate.get('total_mcda_score')}/100. "
        f"It provides a barrier match score of {b_score}/100 and a circularity/sustainability rating of "
        f"{candidate.get('sustainability_score')}/100 for '{commodity_profile.get('name', 'Selected Commodity')}'."
    )

    return {
        "rationale": selection_rationale,
        "requirements_met": requirements_met,
        "limitations": limitations if limitations else ["No severe operational limitations flagged for current target scenario."],
        "missing_data": missing_data if missing_data else ["All primary physical barrier attributes are documented."]
    }
