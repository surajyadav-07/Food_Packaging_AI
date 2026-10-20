
"""
Food Packaging AI — Transparent Multi-Criteria Recommendation Engine

Features:
1. Stage-1 hard knockout constraints.
2. Stage-2 multi-criteria decision analysis (MCDA).
3. Barrier, cost, sustainability and mechanical scoring.
4. Explainable recommendations and trade-offs.
5. Explicit missing-data disclosures.
6. Conservative food-contact compliance handling.

Important:
Scores are decision-support estimates, not proof of food safety,
regulatory approval, or experimentally validated shelf life.
"""

from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np


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


def _number(
    value: Any,
    default: Optional[float] = None
) -> Optional[float]:
    """Safely parse finite numeric values."""
    if value is None or isinstance(value, bool):
        return default

    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _text(value: Any, default: str = "") -> str:
    """Convert a value to normalized lowercase text."""
    if value is None:
        return default.lower()

    result = str(value).strip().lower()
    if result in {"", "none", "nan", "n/a", "null"}:
        return default.lower()

    return result


def _has_verified_compliance_status(material: Dict[str, Any]) -> bool:
    """
    Check whether the material records an affirmative food-contact compliance
    status or recognized regulatory standard citation.

    This checks recorded database/catalog status fields only. It does NOT
    independently verify certificates, regulatory approval, or supporting documentation.
    Missing, blank, unknown, pending, or unapproved statuses fail closed.
    """
    raw_status = material.get("food_contact_compliance")
    if raw_status is None:
        return False

    status = _text(raw_status)
    if not status:
        return False

    # Negative / unverified indicators that must immediately fail closed.
    unverified_tokens = (
        "pending",
        "unapproved",
        "unverified",
        "unknown",
        "not compliant",
        "non-compliant",
        "non compliant",
        "not approved",
        "not certified",
        "not available",
        "missing",
        "untested",
        "failed",
        "rejected",
        "prohibited",
        "restricted",
        "none",
        "nan",
        "n/a",
        "null",
        "ineligible",
        "non-food",
    )

    for token in unverified_tokens:
        if token in status:
            return False

    # Recognized affirmative statuses and regulatory standards.
    # 1. Direct affirmative status keywords
    affirmative_keywords = (
        "compliant",
        "approved",
        "verified",
        "certified",
        "food-contact compliant",
        "food contact compliant",
        "food safe",
        "food-grade",
        "food grade",
    )
    for kw in affirmative_keywords:
        if kw in status:
            return True

    # 2. Recognized regulatory standard citations & frameworks
    # (e.g. US FDA 21 CFR §177.xxx, EU 10/2011, EU 1935/2004, EFSA, BfR, FSSAI, FDA GRAS)
    regulatory_frameworks = (
        "fda",
        "21 cfr",
        "cfr",
        "gras",
        "letter of no objection",
        "nol",
        "eu 10/2011",
        "10/2011",
        "1935/2004",
        "efsa",
        "bfr",
        "fssai",
        "gb 9685",
        "gb 4806",
        "is 9845",
        "bis",
    )
    for rf in regulatory_frameworks:
        if rf in status:
            return True

    return False


def filter_knockouts(
    materials: List[Dict[str, Any]],
    commodity_profile: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Apply screening constraints for temperature, fresh produce,
    puncture risk, moisture, oxygen, grease and food-contact status.

    Returns:
        (qualified_materials, knocked_out_materials_with_reasons)
    """
    qualified = []
    knocked_out = []

    storage_temp = _number(
        commodity_profile.get("storage_temp_c"), 23.0
    )
    category = _text(commodity_profile.get("category"))
    moisture_sens = _text(
        commodity_profile.get("moisture_sensitivity"), "moderate"
    )
    oxygen_sens = _text(
        commodity_profile.get("oxygen_sensitivity"), "moderate"
    )
    fat_content = _text(
        commodity_profile.get("fat_content_level"), "low"
    )
    puncture_risk = _text(
        commodity_profile.get("puncture_risk"), "low"
    )
    target_shelf_life = _number(
        commodity_profile.get("target_shelf_life_days"), 90.0
    )
    require_compliance = commodity_profile.get(
        "require_food_contact_compliance", True
    )

    # Avoid treating a string such as "false" as True.
    if isinstance(require_compliance, str):
        require_compliance = (
            require_compliance.strip().lower()
            in {"true", "1", "yes"}
        )
    else:
        require_compliance = bool(require_compliance)

    is_fresh_produce = any(
        term in category
        for term in ("fruit", "vegetable", "fresh cut")
    )

    for mat in materials:
        reasons = []

        t_min = _number(mat.get("temp_range_min_c"))
        t_max = _number(mat.get("temp_range_max_c"))

        if t_min is not None and storage_temp < t_min:
            reasons.append(
                f"Storage temperature ({storage_temp}°C) is below "
                f"the listed minimum ({t_min}°C)."
            )

        if t_max is not None and storage_temp > t_max:
            reasons.append(
                f"Storage temperature ({storage_temp}°C) exceeds "
                f"the listed maximum ({t_max}°C)."
            )

        # Fresh produce usually requires suitable gas exchange.
        if is_fresh_produce:
            mat_name = _text(mat.get("name"))
            otr = _number(mat.get("otr_cc_m2_day_atm"))

            if "micro-perforated" not in mat_name and (
                otr is None or otr < 100.0
            ):
                reasons.append(
                    "Gas transmission data is insufficient or below "
                    "the screening threshold for fresh produce. "
                    "Confirm produce-specific respiration and packaging "
                    "requirements before selecting this material."
                )

        if puncture_risk in {"high", "very high"}:
            punc_res = _text(mat.get("puncture_resistance"))
            thick = _number(mat.get("nominal_thickness_um"))

            if punc_res in {"low", "poor"}:
                reasons.append(
                    "Listed puncture resistance is inadequate for "
                    "the selected high-puncture-risk commodity."
                )

            if thick is not None and thick < 20.0:
                reasons.append(
                    f"Listed thickness ({thick} µm) is below the "
                    "configured 20 µm screening threshold."
                )

            if punc_res in {"", "unknown"} or thick is None:
                reasons.append(
                    "Puncture resistance or thickness data is missing "
                    "for a high-puncture-risk commodity."
                )

        if "extreme" in moisture_sens and target_shelf_life >= 60:
            wvtr = _number(mat.get("wvtr_g_m2_day"))

            if wvtr is None:
                reasons.append(
                    "WVTR data is missing for an extreme "
                    "moisture-sensitive commodity."
                )
            elif wvtr > 35.0:
                reasons.append(
                    f"WVTR ({wvtr} g/m²/day) exceeds the configured "
                    "screening threshold of 35."
                )

        if "critical" in oxygen_sens and target_shelf_life >= 90:
            otr = _number(mat.get("otr_cc_m2_day_atm"))

            if otr is None:
                reasons.append(
                    "OTR data is missing for a critical "
                    "oxygen-sensitive commodity."
                )
            elif otr > 3000.0:
                reasons.append(
                    f"OTR ({otr} cc/m²/day/atm) exceeds the configured "
                    "screening threshold of 3000."
                )

        if "high" in fat_content:
            grease_res = _text(
                mat.get("grease_fat_resistance"), "unknown"
            )

            if grease_res in {"low", "poor"}:
                reasons.append(
                    "Listed grease/fat resistance is inadequate "
                    "for a high-fat commodity."
                )
            elif grease_res in {"", "unknown"}:
                reasons.append(
                    "Grease/fat resistance data is missing for "
                    "a high-fat commodity."
                )

        # Fail closed when food-contact status is missing or unverified.
        if require_compliance and not _has_verified_compliance_status(mat):
            raw_c = mat.get("food_contact_compliance")
            status_label = f" ('{raw_c}')" if raw_c and str(raw_c).strip() else ""
            reasons.append(
                f"Food-contact compliance status is missing, unapproved, "
                f"or unverified{status_label}. Valid regulatory compliance evidence "
                f"is required before treating this material as compliant."
            )

        if reasons:
            rejected = dict(mat)
            rejected["knockout_reasons"] = reasons
            knocked_out.append(rejected)
        else:
            qualified.append(mat)

    return qualified, knocked_out


def compute_barrier_score(
    material: Dict[str, Any],
    commodity_profile: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """Compute a heuristic 0–100 barrier utility score."""

    wvtr = _number(material.get("wvtr_g_m2_day"))
    otr = _number(material.get("otr_cc_m2_day_atm"))

    category = _text(commodity_profile.get("category"))
    is_fresh = any(
        term in category
        for term in ("fruit", "vegetable", "fresh cut")
    )

    moisture_sens = _text(
        commodity_profile.get("moisture_sensitivity"), "moderate"
    )
    oxygen_sens = _text(
        commodity_profile.get("oxygen_sensitivity"), "moderate"
    )
    light_sens = _text(
        commodity_profile.get("light_sensitivity"), "low"
    )

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

    def transmission_fit(
        actual: Optional[float],
        target: float
    ) -> float:
        if actual is None or actual < 0:
            return 0.0
        if actual <= target:
            return 100.0
        return max(
            10.0,
            100.0 * math.sqrt(target / actual)
        )

    wvtr_fit = transmission_fit(wvtr, target_wvtr)
    otr_fit = transmission_fit(otr, target_otr)

    if is_fresh:
        # A simplified screening heuristic, not a validated MAP model.
        if wvtr is not None and otr is not None:
            if 1000.0 <= otr <= 10000.0 and wvtr >= 30.0:
                barrier_score = 95.0
            elif otr > 10000.0:
                barrier_score = 80.0
            else:
                barrier_score = 45.0
        else:
            barrier_score = 0.0
    else:
        # Missing barrier values contribute zero rather than being
        # silently replaced with optimistic defaults.
        barrier_score = 0.55 * wvtr_fit + 0.45 * otr_fit

    light_rating = _text(
        material.get("light_barrier_rating"), "unknown"
    )

    if "critical" in light_sens or "high" in light_sens:
        if any(
            term in light_rating
            for term in ("opaque", "total", "high")
        ):
            barrier_score = min(100.0, barrier_score + 5.0)
        elif light_rating == "unknown":
            barrier_score = max(0.0, barrier_score - 10.0)
        else:
            barrier_score = max(15.0, barrier_score - 15.0)

    details = {
        "wvtr_actual": wvtr,
        "wvtr_target": target_wvtr,
        "wvtr_fit_score": round(wvtr_fit, 1),
        "otr_actual": otr,
        "otr_target": target_otr,
        "otr_fit_score": round(otr_fit, 1),
        "barrier_overall": round(
            float(np.clip(barrier_score, 0.0, 100.0)), 1
        ),
        "data_complete": wvtr is not None and otr is not None,
        "score_is_heuristic": True,
    }

    return (
        float(np.clip(barrier_score, 0.0, 100.0)),
        details
    )


def compute_sustainability_score(
    material: Dict[str, Any]
) -> float:
    """
    Estimate sustainability utility using the configured recyclability
    map and a small carbon-footprint adjustment.

    This is not a full life-cycle assessment.
    """
    recyclability_status = str(
        material.get("recyclability_status", "")
    ).strip()

    base_score = RECYCLABILITY_SCORE_MAP.get(
        recyclability_status, 40.0
    )

    co2 = _number(
        material.get("carbon_footprint_kg_co2_per_kg")
    )

    if co2 is not None and co2 >= 0:
        if co2 <= 1.5:
            base_score += 5.0
        elif co2 > 4.5:
            base_score -= 8.0

    return float(np.clip(base_score, 5.0, 100.0))


def compute_mechanical_score(
    material: Dict[str, Any],
    commodity_profile: Dict[str, Any]
) -> float:
    """Compute a heuristic mechanical utility score."""

    score = 70.0

    seal = _text(
        material.get("seal_integrity_rating"), "unknown"
    )

    if "excellent" in seal:
        score += 15.0
    elif "good" in seal:
        score += 8.0
    elif "fair" in seal:
        score -= 10.0
    elif seal in {"poor", "low"}:
        score -= 20.0
    else:
        score -= 10.0

    punc = _text(
        material.get("puncture_resistance"), "unknown"
    )
    punc_req = _text(
        commodity_profile.get("puncture_risk"), "low"
    )

    if punc_req in {"high", "very high"}:
        if punc in {"very high", "high", "total"}:
            score += 15.0
        elif punc in {"medium", "moderate"}:
            score -= 5.0
        else:
            score -= 20.0

    tensile = _number(material.get("tensile_strength_mpa"))

    if tensile is not None and tensile >= 100.0:
        score += 5.0
    elif tensile is None:
        score -= 5.0

    return float(np.clip(score, 10.0, 100.0))


def rank_recommendations(
    materials: List[Dict[str, Any]],
    commodity_profile: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Run screening, score qualified materials, and return transparent
    ranked candidates.

    Default weights sum to 1.0:
    barrier 0.40, cost 0.20, sustainability 0.25, mechanical 0.15.
    """
    if commodity_profile is None and "commodity" in kwargs:
        commodity_profile = kwargs["commodity"]
    if weights is None and "user_weights" in kwargs:
        weights = kwargs["user_weights"]
    if commodity_profile is None:
        commodity_profile = {}
    if weights is None:
        weights = {
            "barrier": 0.40,
            "cost": 0.20,
            "sustainability": 0.25,
            "mechanical": 0.15,
        }

    criteria = (
        "barrier",
        "cost",
        "sustainability",
        "mechanical",
    )

    clean_weights = {}
    for criterion in criteria:
        value = _number(weights.get(criterion), 0.0)
        clean_weights[criterion] = max(0.0, value or 0.0)

    weight_sum = sum(clean_weights.values())

    if weight_sum <= 0:
        clean_weights = {
            "barrier": 0.40,
            "cost": 0.20,
            "sustainability": 0.25,
            "mechanical": 0.15,
        }
        weight_sum = sum(clean_weights.values())

    normalized_weights = {
        key: value / weight_sum
        for key, value in clean_weights.items()
    }

    qualified_mats, knocked_out_mats = filter_knockouts(
        materials, commodity_profile
    )

    if not qualified_mats:
        return {
            "top_candidates": [],
            "all_ranked": [],
            "knocked_out": knocked_out_mats,
            "weights_used": normalized_weights,
            "leading_tradeoffs": {},
            "status": "NO_QUALIFIED_MATERIALS",
            "message": (
                "No materials passed the configured screening rules. "
                "Review knockout reasons and input data. Do not weaken "
                "food-safety requirements just to obtain a result."
            ),
            "total_evaluated": len(materials),
            "total_qualified": 0,
            "total_knocked_out": len(knocked_out_mats),
            "scoring_note": (
                "Scores are heuristic decision-support estimates. "
                "Missing criteria are excluded and remaining weights "
                "are renormalized for each candidate."
            ),
        }

    costs = [
        _number(mat.get("cost_index_usd_per_kg"))
        for mat in qualified_mats
    ]
    valid_costs = [value for value in costs if value is not None]

    min_cost = min(valid_costs) if valid_costs else None
    max_cost = max(valid_costs) if valid_costs else None

    scored_candidates = []

    for mat in qualified_mats:
        barrier_score, barrier_details = compute_barrier_score(
            mat, commodity_profile
        )
        sustainability_score = compute_sustainability_score(mat)
        mechanical_score = compute_mechanical_score(
            mat, commodity_profile
        )

        cost_actual = _number(mat.get("cost_index_usd_per_kg"))

        if cost_actual is None or cost_actual < 0:
            cost_score = None
        elif min_cost is None or max_cost is None:
            cost_score = None
        elif max_cost == min_cost:
            cost_score = 100.0
        else:
            cost_score = 100.0 * (
                1.0 - (cost_actual - min_cost) / (max_cost - min_cost)
            )

        # Re-normalize weights across available criteria for each material.
        # Missing cost is not silently treated as a cheap default.
        available_scores = {
            "barrier": barrier_score,
            "cost": cost_score,
            "sustainability": sustainability_score,
            "mechanical": mechanical_score,
        }

        active_weight_sum = sum(
            normalized_weights[key]
            for key, value in available_scores.items()
            if value is not None
        )

        if active_weight_sum > 0:
            total_score = sum(
                normalized_weights[key] * value
                for key, value in available_scores.items()
                if value is not None
            ) / active_weight_sum
        else:
            total_score = 0.0

        candidate = dict(mat)
        candidate["barrier_score"] = round(barrier_score, 1)
        candidate["cost_score"] = (
            round(cost_score, 1)
            if cost_score is not None
            else None
        )
        candidate["sustainability_score"] = round(
            sustainability_score, 1
        )
        candidate["mechanical_score"] = round(
            mechanical_score, 1
        )
        candidate["total_mcda_score"] = round(
            float(total_score), 1
        )
        candidate["barrier_details"] = barrier_details
        candidate["scoring_weights_used"] = normalized_weights
        candidate["missing_score_criteria"] = [
            key for key, value in available_scores.items()
            if value is None
        ]
        candidate["explanation"] = generate_explanation(
            candidate, commodity_profile
        )

        scored_candidates.append(candidate)

    scored_candidates.sort(
        key=lambda item: item["total_mcda_score"],
        reverse=True
    )

    for index, candidate in enumerate(scored_candidates, start=1):
        candidate["rank"] = index

    top_candidates = scored_candidates[:3]
    leading_tradeoffs = explain_top_pick_tradeoffs(
        top_candidates[0], top_candidates[1:]
    ) if len(top_candidates) > 0 else {}

    return {
        "top_candidates": top_candidates,
        "all_ranked": scored_candidates,
        "knocked_out": knocked_out_mats,
        "weights_used": normalized_weights,
        "leading_tradeoffs": leading_tradeoffs,
        "status": "SUCCESS",
        "total_evaluated": len(materials),
        "total_qualified": len(qualified_mats),
        "total_knocked_out": len(knocked_out_mats),
        "scoring_note": (
            "Scores are heuristic decision-support estimates. "
            "Missing criteria are excluded and remaining weights "
            "are renormalized for each candidate."
        ),
    }


def generate_explanation(
    candidate: Dict[str, Any],
    commodity_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Explain recorded data, limitations, and missing information.
    Avoid asserting that unverified requirements have been satisfied.
    """
    name = candidate.get("name", "Material")
    barrier_score = candidate.get("barrier_score", 0.0)
    sustainability_score = candidate.get(
        "sustainability_score", 0.0
    )
    total_score = candidate.get("total_mcda_score")
    recyclability = str(
        candidate.get("recyclability_status", "Unspecified")
    )

    requirements_met = []
    limitations = []
    missing_data = []

    wvtr = _number(candidate.get("wvtr_g_m2_day"))
    otr = _number(candidate.get("otr_cc_m2_day_atm"))
    cost = _number(candidate.get("cost_index_usd_per_kg"))

    if wvtr is not None:
        requirements_met.append(
            f"Recorded WVTR: {wvtr} g/m²/day. "
            "Verify the source and test conditions."
        )
    else:
        missing_data.append("Water vapor transmission rate is missing.")

    if otr is not None:
        requirements_met.append(
            f"Recorded OTR: {otr} cc/m²/day/atm. "
            "Verify the source and test conditions."
        )
    else:
        missing_data.append("Oxygen transmission rate is missing.")

    compliance = candidate.get("food_contact_compliance")
    if compliance is not None and str(compliance).strip():
        requirements_met.append(
            f"Recorded food-contact status: {compliance}. "
            "This label alone does not establish independent "
            "regulatory verification."
        )
    else:
        missing_data.append("Food-contact compliance status is missing.")

    t_min = candidate.get("temp_range_min_c")
    t_max = candidate.get("temp_range_max_c")

    if t_min is not None and t_max is not None:
        requirements_met.append(
            f"Listed temperature range: {t_min}°C to {t_max}°C. "
            "Supporting test evidence should be confirmed."
        )
    else:
        missing_data.append(
            "Complete temperature-range data is missing."
        )

    if cost is None:
        missing_data.append(
            "Cost index is missing; cost was excluded from this "
            "candidate's composite score."
        )

    if "non-recyclable" in recyclability.lower() or (
        "laminate" in recyclability.lower()
    ):
        limitations.append(
            "The listed structure may be difficult to recycle in "
            "standard curbside systems; check local facilities."
        )

    if cost is not None and cost > 4.0:
        limitations.append(
            f"Higher recorded cost index (${cost:.2f}/kg); "
            "confirm local supplier pricing."
        )

    if "pla" in name.lower() or "bio" in name.lower():
        limitations.append(
            "Bio-based composition does not by itself guarantee "
            "home compostability or suitable thermal performance."
        )

    if "glass" in name.lower():
        limitations.append(
            "Glass can have higher transport weight; compare full "
            "life-cycle impacts and reuse assumptions."
        )

    if "ldpe" in name.lower() and otr is not None and otr > 4000:
        limitations.append(
            "The recorded OTR is high; suitability for oxygen-sensitive "
            "foods needs product-specific validation."
        )

    if candidate.get("co2tr_cc_m2_day_atm") is None:
        missing_data.append(
            "Carbon dioxide transmission rate (CO2TR) is not documented."
        )

    if candidate.get("tensile_strength_mpa") is None:
        missing_data.append(
            "Tensile strength is not documented."
        )

    if candidate.get("carbon_footprint_kg_co2_per_kg") is None:
        missing_data.append(
            "Cradle-to-gate carbon footprint data is unavailable; "
            "the score is not a complete life-cycle assessment."
        )

    if not candidate.get("is_verified_experimental"):
        missing_data.append(
            "Independent experimental laboratory verification is not recorded; "
            "values are handbook/literature baselines."
        )

    if not missing_data:
        missing_data.append(
            "Batch-specific certificate of analysis (CoA) and real-time shelf-life "
            "validation data under actual logistics conditions are not documented."
        )

    if not limitations:
        limitations.append(
            "No additional rule-based limitation was flagged. "
            "This does not mean the material is proven safe or optimal."
        )

    if not requirements_met:
        requirements_met.append(
            "No recorded baseline properties met the screening profile criteria."
        )

    if total_score is None:
        rationale = (
            f"{name} received a barrier score of {barrier_score}/100 "
            f"and sustainability score of {sustainability_score}/100. "
            "Validate the available data before making a final selection."
        )
    else:
        rationale = (
            f"{name} received a heuristic MCDA score of "
            f"{total_score}/100, a barrier score of {barrier_score}/100, "
            f"and a sustainability score of {sustainability_score}/100 "
            f"for '{commodity_profile.get('name', 'Selected Commodity')}'. "
            "These scores are comparative estimates, not safety "
            "certification or experimental validation."
        )

    return {
        "rationale": rationale,
        "requirements_met": requirements_met,
        "limitations": limitations,
        "missing_data": missing_data,
    }


POLYMER_DENSITIES_G_CM3 = {
    "pp": 0.905,
    "bopp": 0.905,
    "ldpe": 0.92,
    "hdpe": 0.95,
    "pet": 1.38,
    "rpet": 1.38,
    "pla": 1.25,
    "nylon": 1.14,
    "bopa": 1.14,
    "cellulose": 1.45,
    "natureflex": 1.45,
    "paper": 0.80,
    "paperboard": 0.80,
    "aluminium": 2.70,
    "aluminum": 2.70,
    "foil": 1.70,
    "glass": 2.50,
}


def get_material_density(material: Dict[str, Any]) -> float:
    """Estimates physical material density in g/cm3 for packaging mass estimation."""
    poly_code = _text(material.get("polymer_code"))
    name = _text(material.get("name"))
    structure = _text(material.get("structure_type"))

    for key, density in POLYMER_DENSITIES_G_CM3.items():
        if key in poly_code or key in name or key in structure:
            return density
    return 1.15


def calculate_unit_packaging_economics(
    material: Dict[str, Any],
    package_surface_area_cm2: float = 400.0,
    monthly_volume: int = 50000
) -> Dict[str, Any]:
    """
    Computes first-principles package piece mass, unit cost, and annualized economics.

    Mass = Surface Area (cm2) * Thickness (cm) * Density (g/cm3)
    """
    thickness_um = _number(material.get("nominal_thickness_um"), 40.0) or 40.0
    density = get_material_density(material)
    cost_per_kg = _number(material.get("cost_index_usd_per_kg"))
    co2_per_kg = _number(material.get("carbon_footprint_kg_co2_per_kg"))

    thickness_cm = thickness_um / 10000.0
    unit_mass_g = package_surface_area_cm2 * thickness_cm * density

    unit_cost_usd = (unit_mass_g / 1000.0 * cost_per_kg) if cost_per_kg is not None else None
    annual_spend_usd = (unit_cost_usd * monthly_volume * 12.0) if unit_cost_usd is not None else None

    unit_co2_kg = (unit_mass_g / 1000.0 * co2_per_kg) if co2_per_kg is not None else None
    annual_co2_kg = (unit_co2_kg * monthly_volume * 12.0) if unit_co2_kg is not None else None

    return {
        "material_name": material.get("name", "Material"),
        "density_g_cm3": round(density, 2),
        "nominal_thickness_um": thickness_um,
        "unit_mass_grams": round(float(unit_mass_g), 2),
        "unit_cost_usd": round(float(unit_cost_usd), 4) if unit_cost_usd is not None else None,
        "cost_per_thousand_usd": round(float(unit_cost_usd * 1000.0), 2) if unit_cost_usd is not None else None,
        "annual_spend_usd": round(float(annual_spend_usd), 2) if annual_spend_usd is not None else None,
        "unit_carbon_kg_co2": round(float(unit_co2_kg), 4) if unit_co2_kg is not None else None,
        "annual_carbon_kg_co2": round(float(annual_co2_kg), 1) if annual_co2_kg is not None else None,
        "package_surface_area_cm2": package_surface_area_cm2,
        "monthly_volume": monthly_volume
    }


def explain_top_pick_tradeoffs(
    top_candidate: Dict[str, Any],
    alternatives: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generates comparative insights explaining why the leading recommendation
    ranked higher than runner-up alternatives.
    """
    if not top_candidate:
        return {
            "headline": "No candidates qualified under the configured screening constraints.",
            "advantages": [],
            "concessions": [],
            "verdict": "Adjust temperature, shelf-life, or compliance screening thresholds to evaluate candidates."
        }

    top_name = top_candidate.get("name", "Rank #1 Material")
    top_score = top_candidate.get("total_mcda_score", 0.0)
    top_b = top_candidate.get("barrier_score", 0.0)
    top_c = top_candidate.get("cost_score")
    top_s = top_candidate.get("sustainability_score", 0.0)
    top_m = top_candidate.get("mechanical_score", 0.0)
    top_wvtr = top_candidate.get("wvtr_g_m2_day")
    top_otr = top_candidate.get("otr_cc_m2_day_atm")
    top_cost_val = top_candidate.get("cost_index_usd_per_kg")

    advantages = []
    concessions = []

    if alternatives:
        runner_up = alternatives[0]
        alt_name = runner_up.get("name", "Runner-up")
        alt_score = runner_up.get("total_mcda_score", 0.0)
        alt_b = runner_up.get("barrier_score", 0.0)
        alt_c = runner_up.get("cost_score")
        alt_s = runner_up.get("sustainability_score", 0.0)
        alt_wvtr = runner_up.get("wvtr_g_m2_day")
        alt_otr = runner_up.get("otr_cc_m2_day_atm")
        alt_cost_val = runner_up.get("cost_index_usd_per_kg")

        score_margin = round(float(top_score - alt_score), 1)
        headline = f"Why {top_name} Ranks #{top_candidate.get('rank', 1)} (+{score_margin} pts above {alt_name})"

        # Barrier comparison
        if top_b > alt_b:
            b_diff = round(top_b - alt_b, 1)
            advantages.append(
                f"Superior Barrier Protection (+{b_diff} pts): Recorded WVTR of {top_wvtr} g/m²/day vs {alt_wvtr} g/m²/day for {alt_name}."
            )
        elif alt_b > top_b:
            concessions.append(
                f"{alt_name} has higher barrier score ({alt_b}/100 vs {top_b}/100), but {top_name} compensates with cost and sustainability efficiency."
            )

        # Cost comparison
        if top_c is not None and alt_c is not None:
            if top_c > alt_c:
                c_diff = round(top_c - alt_c, 1)
                advantages.append(
                    f"Higher Economic Utility (+{c_diff} pts): Listed at ${top_cost_val:.2f}/kg compared to ${alt_cost_val:.2f}/kg for {alt_name}."
                )
            elif alt_c > top_c:
                concessions.append(
                    f"{alt_name} is more economical (${alt_cost_val:.2f}/kg vs ${top_cost_val:.2f}/kg), representing a budget-focused alternative."
                )

        # Sustainability comparison
        if top_s > alt_s:
            s_diff = round(top_s - alt_s, 1)
            advantages.append(
                f"Circularity Advantage (+{s_diff} pts): Ranked for circular packaging compatibility ({top_candidate.get('recyclability_status', 'Standard')})."
            )
        elif alt_s > top_s:
            concessions.append(
                f"{alt_name} scored higher on circularity ({alt_s}/100 vs {top_s}/100), but lacks equivalent barrier performance."
            )

        # Mechanical comparison
        if top_m > runner_up.get("mechanical_score", 0.0):
            advantages.append(
                f"Mechanical Resilience: Offers {top_candidate.get('seal_integrity_rating', 'Good')} seal integrity and {top_candidate.get('puncture_resistance', 'Medium')} puncture resistance."
            )

        verdict = (
            f"{top_name} emerges as the optimal multi-criteria compromise with an overall utility of {top_score}/100, "
            f"offering the most balanced trade-off across moisture protection, economic cost, and recycling feasibility."
        )
    else:
        headline = f"Single Best Fit: {top_name}"
        advantages.append(f"Achieves composite score of {top_score}/100 satisfying all critical screening thresholds.")
        verdict = f"{top_name} is the sole material meeting all constraints for this food profile."

    return {
        "headline": headline,
        "advantages": advantages,
        "concessions": concessions,
        "verdict": verdict
    }
