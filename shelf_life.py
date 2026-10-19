"""
shelf_life.py - Mass-Transfer & Permeation Scenario Shelf-Life Module
Provides first-principles permeation modeling for moisture gain and oxygen ingress.
Prominently notes that packaging barrier is only one determinant among microbial,
enzymatic, headspace, and storage history factors.
"""

from typing import Dict, Any, List
import numpy as np


def estimate_moisture_shelf_life(
    wvtr_standard: float,           # g/m2/day at standard 38°C, 90% RH
    package_surface_area_m2: float, # e.g. 0.05 m2 (standard pouch)
    product_dry_weight_g: float,    # e.g. 150 g
    initial_moisture_percent: float,# e.g. 2.0% (dry basis or wet basis)
    critical_moisture_percent: float,# e.g. 5.5% (crispness loss threshold)
    ambient_temp_c: float = 25.0,
    ambient_rh_percent: float = 65.0
) -> Dict[str, Any]:
    """
    Simulates moisture gain through package web using steady-state mass transfer
    derived from Robertson (2012) and Labuza (1982) Food Packaging Shelf-Life Fundamentals.

    dm/dt = (WVTR_actual * A * Delta_RH) / (100 * W_s)
    """
    if critical_moisture_percent <= initial_moisture_percent:
        return {
            "error": "Critical moisture threshold must be greater than initial moisture.",
            "estimated_days": 0.0
        }
    if package_surface_area_m2 <= 0 or product_dry_weight_g <= 0 or wvtr_standard < 0:
        return {
            "error": "Physical parameters (surface area, weight, WVTR) must be strictly positive.",
            "estimated_days": 0.0
        }

    # Temperature adjustment: Permeation increases with temperature according to Arrhenius behavior.
    # Standard WVTR test is at 38°C. For lower temperatures, permeation is typically lower.
    # Typical activation energy for polyolefin water permeation is ~35-45 kJ/mol (Robertson 2012).
    # Approximate temperature correction factor:
    delta_t = ambient_temp_c - 38.0
    temp_factor = np.exp(0.045 * delta_t)  # drops by ~50% at 23°C relative to 38°C
    wvtr_adjusted = max(0.0001, wvtr_standard * temp_factor * (ambient_rh_percent / 90.0))

    # Moisture mass to gain (grams of H2O)
    delta_moisture_fraction = (critical_moisture_percent - initial_moisture_percent) / 100.0
    critical_water_gain_g = delta_moisture_fraction * product_dry_weight_g

    # Daily moisture transfer rate (grams H2O / day) through package area A
    daily_gain_g = wvtr_adjusted * package_surface_area_m2

    if daily_gain_g <= 1e-7:
        estimated_days = 9999.0
    else:
        estimated_days = critical_water_gain_g / daily_gain_g

    # Generate daily progression curve for scenario plotting (up to estimated_days or 365 days)
    plot_days = min(365, max(30, int(estimated_days * 1.2)))
    days_array = np.linspace(0, plot_days, 50)
    moisture_curve = initial_moisture_percent + (daily_gain_g / product_dry_weight_g * 100.0) * days_array

    return {
        "estimated_days": round(float(estimated_days), 1),
        "wvtr_standard": wvtr_standard,
        "wvtr_adjusted": round(float(wvtr_adjusted), 3),
        "daily_gain_grams": round(float(daily_gain_g), 4),
        "critical_water_gain_g": round(float(critical_water_gain_g), 2),
        "days_array": days_array.tolist(),
        "moisture_curve": [round(float(v), 2) for v in moisture_curve],
        "critical_threshold": critical_moisture_percent,
        "is_hermetic": bool(wvtr_standard < 0.02)
    }


def estimate_oxygen_ingress(
    otr_standard: float,            # cc/m2/day/atm at 23°C, 0% RH
    package_surface_area_m2: float, # e.g. 0.05 m2
    headspace_volume_cc: float,     # e.g. 100 cc
    initial_headspace_o2_percent: float = 0.5, # e.g. 0.5% with N2 flush
    critical_headspace_o2_percent: float = 2.0, # e.g. 2.0% maximum allowable
    ambient_temp_c: float = 23.0,
    ambient_o2_percent: float = 21.0 # Atmospheric oxygen concentration (~0.21 atm)
) -> Dict[str, Any]:
    """
    Simulates cumulative oxygen ingress into package headspace under steady-state permeation.
    Safely handles inward, zero, and outward oxygen partial-pressure gradients.
    """
    # 1. Physical input validity
    if package_surface_area_m2 <= 0 or headspace_volume_cc <= 0 or otr_standard < 0:
        return {
            "error": "Surface area, headspace volume, and OTR must be strictly positive.",
            "estimated_days": 0.0,
            "status": "INVALID_INPUTS"
        }

    # 2. Critical threshold vs initial oxygen validity
    if critical_headspace_o2_percent <= initial_headspace_o2_percent:
        return {
            "error": "Critical oxygen threshold must exceed initial headspace oxygen.",
            "estimated_days": 0.0,
            "status": "INVALID_THRESHOLD"
        }

    # 3. Temperature adjustment on OTR (standard test is 23°C)
    delta_t = ambient_temp_c - 23.0
    temp_factor = np.exp(0.035 * delta_t)
    otr_adjusted = max(0.0001, otr_standard * temp_factor)

    # 4. Oxygen partial-pressure driving force (atm)
    # Atmospheric oxygen partial pressure ~ (ambient_o2_percent / 100.0) atm
    ambient_p = ambient_o2_percent / 100.0
    initial_p = initial_headspace_o2_percent / 100.0
    delta_p = ambient_p - initial_p

    # Edge Case: Outward gradient (initial package O2 > ambient O2, e.g. 70-80% O2 high-oxygen MAP)
    if delta_p < -1e-6:
        return {
            "error": (
                f"Initial package oxygen ({initial_headspace_o2_percent:.1f}%) exceeds ambient oxygen concentration "
                f"({ambient_o2_percent:.1f}%). The oxygen partial-pressure gradient points outward; oxygen permeates "
                "out of the package rather than inward. An outward permeation or gas-depletion model is required."
            ),
            "status": "OUTWARD_GRADIENT",
            "gradient_direction": "outward",
            "estimated_days": None,
            "otr_standard": otr_standard,
            "otr_adjusted": round(float(otr_adjusted), 3),
            "delta_p": round(float(delta_p), 4),
            "is_hermetic": bool(otr_standard < 0.05)
        }

    # Edge Case: Zero gradient (initial package O2 == ambient O2, equilibrium air packaging)
    if abs(delta_p) <= 1e-6:
        return {
            "error": (
                f"Initial package oxygen ({initial_headspace_o2_percent:.1f}%) is in equilibrium with ambient oxygen "
                f"concentration ({ambient_o2_percent:.1f}%). The partial-pressure gradient is zero; the current "
                "inward-ingress model cannot calculate a positive ingress shelf-life estimate under zero-gradient assumptions."
            ),
            "status": "ZERO_GRADIENT",
            "gradient_direction": "zero",
            "estimated_days": None,
            "otr_standard": otr_standard,
            "otr_adjusted": round(float(otr_adjusted), 3),
            "delta_p": 0.0,
            "is_hermetic": bool(otr_standard < 0.05)
        }

    # Edge Case: Critical threshold >= ambient oxygen (unreachable via passive atmospheric diffusion)
    if critical_headspace_o2_percent >= ambient_o2_percent:
        return {
            "error": (
                f"Critical oxygen threshold ({critical_headspace_o2_percent:.1f}%) is greater than or equal to ambient oxygen "
                f"concentration ({ambient_o2_percent:.1f}%). Under passive atmospheric diffusion, headspace oxygen "
                f"asymptotically approaches ambient concentration ({ambient_o2_percent:.1f}%) and will never exceed it from permeation alone."
            ),
            "status": "UNREACHABLE_THRESHOLD",
            "gradient_direction": "inward",
            "estimated_days": None,
            "otr_standard": otr_standard,
            "otr_adjusted": round(float(otr_adjusted), 3),
            "delta_p": round(float(delta_p), 4),
            "is_hermetic": bool(otr_standard < 0.05)
        }

    # 5. Normal Inward Ingress Case (0 <= initial < critical < ambient)
    daily_o2_ingress_cc = otr_adjusted * package_surface_area_m2 * delta_p

    delta_o2_percent = critical_headspace_o2_percent - initial_headspace_o2_percent
    allowable_o2_cc = (delta_o2_percent / 100.0) * headspace_volume_cc

    # Check for hermetic material or near-zero transfer
    if otr_standard < 0.05 or daily_o2_ingress_cc <= 1e-7:
        estimated_days = 9999.0
        is_herm = True
    else:
        estimated_days = allowable_o2_cc / daily_o2_ingress_cc
        is_herm = False

    plot_days = min(365, max(30, int(estimated_days * 1.2)))
    days_array = np.linspace(0, plot_days, 50)
    o2_curve = initial_headspace_o2_percent + (daily_o2_ingress_cc / headspace_volume_cc * 100.0) * days_array

    return {
        "estimated_days": round(float(estimated_days), 1),
        "otr_standard": otr_standard,
        "otr_adjusted": round(float(otr_adjusted), 3),
        "daily_o2_ingress_cc": round(float(daily_o2_ingress_cc), 4),
        "allowable_o2_cc": round(float(allowable_o2_cc), 3),
        "days_array": days_array.tolist(),
        "o2_curve": [round(float(v), 2) for v in o2_curve],
        "critical_threshold": critical_headspace_o2_percent,
        "is_hermetic": is_herm,
        "status": "SUCCESS",
        "gradient_direction": "inward"
    }


def estimate_combined_shelf_life(
    wvtr_standard: float,
    otr_standard: float,
    package_surface_area_m2: float,
    product_dry_weight_g: float,
    initial_moisture_percent: float,
    critical_moisture_percent: float,
    headspace_volume_cc: float,
    initial_headspace_o2_percent: float = 0.5,
    critical_headspace_o2_percent: float = 2.0,
    ambient_temp_c: float = 23.0,
    ambient_rh_percent: float = 60.0,
    ambient_o2_percent: float = 21.0
) -> Dict[str, Any]:
    """
    Evaluates coupled moisture absorption and oxygen ingress degradation mechanisms.

    Determines the primary limiting degradation mode (first failure threshold reached),
    conservative shelf-life estimate, and diagnostic comparison.
    """
    m_res = estimate_moisture_shelf_life(
        wvtr_standard=wvtr_standard,
        package_surface_area_m2=package_surface_area_m2,
        product_dry_weight_g=product_dry_weight_g,
        initial_moisture_percent=initial_moisture_percent,
        critical_moisture_percent=critical_moisture_percent,
        ambient_temp_c=ambient_temp_c,
        ambient_rh_percent=ambient_rh_percent
    )

    o_res = estimate_oxygen_ingress(
        otr_standard=otr_standard,
        package_surface_area_m2=package_surface_area_m2,
        headspace_volume_cc=headspace_volume_cc,
        initial_headspace_o2_percent=initial_headspace_o2_percent,
        critical_headspace_o2_percent=critical_headspace_o2_percent,
        ambient_temp_c=ambient_temp_c,
        ambient_o2_percent=ambient_o2_percent
    )

    m_days = m_res.get("estimated_days") if "error" not in m_res else None
    o_days = o_res.get("estimated_days") if o_res.get("status") == "SUCCESS" else None

    # Determine limiting degradation factor
    if m_days is not None and o_days is not None:
        if m_days < o_days:
            limiting_days = m_days
            limiting_factor = "Moisture Absorption (Texture Staling / Caking)"
            secondary_days = o_days
            secondary_factor = "Oxygen Ingress (Lipid Oxidation / Rancidity)"
        elif o_days < m_days:
            limiting_days = o_days
            limiting_factor = "Oxygen Ingress (Lipid Oxidation / Rancidity)"
            secondary_days = m_days
            secondary_factor = "Moisture Absorption (Texture Staling / Caking)"
        else:
            limiting_days = m_days
            limiting_factor = "Synchronous Degradation (Moisture & Oxygen Limits Reached Concurrently)"
            secondary_days = o_days
            secondary_factor = limiting_factor
    elif m_days is not None:
        limiting_days = m_days
        limiting_factor = "Moisture Absorption (Texture Staling / Caking)"
        secondary_days = None
        secondary_factor = o_res.get("status", "Oxygen Model Inactive")
    elif o_days is not None:
        limiting_days = o_days
        limiting_factor = "Oxygen Ingress (Lipid Oxidation / Rancidity)"
        secondary_days = None
        secondary_factor = m_res.get("error", "Moisture Model Inactive")
    else:
        limiting_days = 0.0
        limiting_factor = "Simulation Undetermined (Check Moisture and Oxygen Thresholds)"
        secondary_days = None
        secondary_factor = None

    return {
        "limiting_days": round(float(limiting_days), 1) if limiting_days is not None else 0.0,
        "limiting_factor": limiting_factor,
        "secondary_days": round(float(secondary_days), 1) if secondary_days is not None else None,
        "secondary_factor": secondary_factor,
        "moisture_result": m_res,
        "oxygen_result": o_res,
        "ambient_temp_c": ambient_temp_c,
        "ambient_rh_percent": ambient_rh_percent,
        "scientific_disclaimer": (
            "Model operates on steady-state Fickian diffusion with Arrhenius temperature scaling. "
            "It estimates packaging barrier limitation only and does not model microbial proliferation, "
            "enzymatic staling, or physical package punctures."
        )
    }
