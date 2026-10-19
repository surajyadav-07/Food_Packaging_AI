"""
test_shelf_life.py - Tests for mass-transfer permeation scenario calculations and error handling
"""

import pytest
import shelf_life


def test_valid_moisture_shelf_life():
    """Valid inputs must compute positive estimated days and valid moisture curves."""
    res = shelf_life.estimate_moisture_shelf_life(
        wvtr_standard=1.5,
        package_surface_area_m2=0.04,
        product_dry_weight_g=100.0,
        initial_moisture_percent=2.0,
        critical_moisture_percent=5.0,
        ambient_temp_c=25.0,
        ambient_rh_percent=60.0
    )
    assert "error" not in res
    assert res["estimated_days"] > 0
    assert len(res["moisture_curve"]) > 0
    assert res["moisture_curve"][-1] >= res["moisture_curve"][0]


def test_invalid_moisture_thresholds():
    """Critical moisture lower than initial moisture must trigger explicit error."""
    res = shelf_life.estimate_moisture_shelf_life(
        wvtr_standard=1.5,
        package_surface_area_m2=0.04,
        product_dry_weight_g=100.0,
        initial_moisture_percent=5.0,
        critical_moisture_percent=3.0  # Invalid!
    )
    assert "error" in res
    assert res["estimated_days"] == 0.0


def test_negative_physical_inputs():
    """Negative surface area or weight must trigger error."""
    res = shelf_life.estimate_moisture_shelf_life(
        wvtr_standard=1.5,
        package_surface_area_m2=-0.04,
        product_dry_weight_g=100.0,
        initial_moisture_percent=2.0,
        critical_moisture_percent=5.0
    )
    assert "error" in res
    assert res["estimated_days"] == 0.0


def test_oxygen_ingress_simulation():
    """Valid OTR inputs must return positive days and ingress curve."""
    res = shelf_life.estimate_oxygen_ingress(
        otr_standard=25.0,
        package_surface_area_m2=0.05,
        headspace_volume_cc=120.0,
        initial_headspace_o2_percent=0.5,
        critical_headspace_o2_percent=2.5,
        ambient_temp_c=23.0
    )
    assert "error" not in res
    assert res["estimated_days"] > 0
    assert len(res["o2_curve"]) > 0
    assert res["o2_curve"][-1] >= res["o2_curve"][0]


def test_hermetic_material_handling():
    """Near-zero transmission materials (like foil or glass) must be flagged as hermetic."""
    res_m = shelf_life.estimate_moisture_shelf_life(
        wvtr_standard=0.005,
        package_surface_area_m2=0.05,
        product_dry_weight_g=150.0,
        initial_moisture_percent=2.0,
        critical_moisture_percent=6.0
    )
    assert res_m["is_hermetic"] is True

    res_o = shelf_life.estimate_oxygen_ingress(
        otr_standard=0.005,
        package_surface_area_m2=0.05,
        headspace_volume_cc=100.0,
        initial_headspace_o2_percent=0.2,
        critical_headspace_o2_percent=1.5
    )
    assert res_o["is_hermetic"] is True


def test_oxygen_ingress_zero_gradient():
    """When package oxygen equals ambient oxygen, zero-gradient must be cleanly reported."""
    res = shelf_life.estimate_oxygen_ingress(
        otr_standard=25.0,
        package_surface_area_m2=0.05,
        headspace_volume_cc=120.0,
        initial_headspace_o2_percent=21.0,
        critical_headspace_o2_percent=21.0,
        ambient_temp_c=23.0
    )
    # Critical must exceed initial, so if equal, INVALID_THRESHOLD is caught
    assert "error" in res

    # If critical is specified higher (e.g. 21.5%), zero gradient check triggers
    res2 = shelf_life.estimate_oxygen_ingress(
        otr_standard=25.0,
        package_surface_area_m2=0.05,
        headspace_volume_cc=120.0,
        initial_headspace_o2_percent=21.0,
        critical_headspace_o2_percent=22.0,
        ambient_temp_c=23.0,
        ambient_o2_percent=21.0
    )
    assert res2["status"] == "ZERO_GRADIENT"
    assert res2["estimated_days"] is None
    assert "zero" in res2["error"].lower()
    assert res2["delta_p"] == 0.0


def test_oxygen_ingress_outward_gradient():
    """When package oxygen exceeds ambient, outward permeation must be cleanly reported."""
    res = shelf_life.estimate_oxygen_ingress(
        otr_standard=30.0,
        package_surface_area_m2=0.05,
        headspace_volume_cc=150.0,
        initial_headspace_o2_percent=70.0, # High-oxygen MAP (e.g. red meat)
        critical_headspace_o2_percent=75.0,
        ambient_temp_c=23.0,
        ambient_o2_percent=21.0
    )
    assert res["status"] == "OUTWARD_GRADIENT"
    assert res["estimated_days"] is None
    assert "outward" in res["error"].lower()
    assert res["delta_p"] < 0


def test_oxygen_ingress_unreachable_threshold():
    """Thresholds greater than ambient cannot be reached via passive atmospheric diffusion."""
    res = shelf_life.estimate_oxygen_ingress(
        otr_standard=25.0,
        package_surface_area_m2=0.05,
        headspace_volume_cc=120.0,
        initial_headspace_o2_percent=2.0,
        critical_headspace_o2_percent=25.0, # Impossible to reach from 21% air
        ambient_temp_c=23.0,
        ambient_o2_percent=21.0
    )
    assert res["status"] == "UNREACHABLE_THRESHOLD"
    assert res["estimated_days"] is None
    assert "asymptotically" in res["error"].lower()


def test_oxygen_ingress_never_negative():
    """Across any valid or invalid input permutation, estimated_days must never be negative."""
    test_cases = [
        (0.5, 2.0, 21.0),
        (20.0, 20.5, 21.0),
        (21.0, 22.0, 21.0),
        (50.0, 60.0, 21.0),
        (80.0, 85.0, 21.0)
    ]
    for init_o2, crit_o2, amb_o2 in test_cases:
        res = shelf_life.estimate_oxygen_ingress(
            otr_standard=20.0,
            package_surface_area_m2=0.04,
            headspace_volume_cc=100.0,
            initial_headspace_o2_percent=init_o2,
            critical_headspace_o2_percent=crit_o2,
            ambient_o2_percent=amb_o2
        )
        if res.get("estimated_days") is not None:
            assert res["estimated_days"] >= 0.0, f"Failed for init={init_o2}, crit={crit_o2}: got {res['estimated_days']}"


def test_estimate_combined_shelf_life_moisture_limiting():
    """When moisture spoilage occurs before oxygen limit, moisture must be identified as limiting factor."""
    res = shelf_life.estimate_combined_shelf_life(
        wvtr_standard=15.0,  # High moisture transmission -> rapid moisture staling
        otr_standard=0.5,    # Very high oxygen barrier -> slow oxygen ingress
        package_surface_area_m2=0.04,
        product_dry_weight_g=100.0,
        initial_moisture_percent=2.0,
        critical_moisture_percent=4.0,
        headspace_volume_cc=100.0,
        initial_headspace_o2_percent=0.5,
        critical_headspace_o2_percent=3.0,
        ambient_temp_c=25.0,
        ambient_rh_percent=75.0
    )
    assert res["limiting_days"] > 0
    assert "Moisture Absorption" in res["limiting_factor"]
    assert res["secondary_days"] is not None
    assert res["secondary_days"] > res["limiting_days"]


def test_estimate_combined_shelf_life_oxygen_limiting():
    """When oxygen limit is reached first, oxygen ingress must be identified as limiting factor."""
    res = shelf_life.estimate_combined_shelf_life(
        wvtr_standard=0.1,    # High moisture barrier
        otr_standard=50.0,   # Moderate oxygen permeability
        package_surface_area_m2=0.04,
        product_dry_weight_g=100.0,
        initial_moisture_percent=2.0,
        critical_moisture_percent=10.0,
        headspace_volume_cc=100.0,
        initial_headspace_o2_percent=0.5,
        critical_headspace_o2_percent=2.5,
        ambient_temp_c=25.0,
        ambient_rh_percent=50.0
    )
    assert res["limiting_days"] > 0
    assert "Oxygen Ingress" in res["limiting_factor"]
    assert res["secondary_days"] is not None
    assert res["secondary_days"] > res["limiting_days"]



def test_estimate_combined_shelf_life_structure_and_disclaimer():
    """Combined shelf life return dict must contain honest scientific disclaimer and valid keys."""
    res = shelf_life.estimate_combined_shelf_life(
        wvtr_standard=2.0,
        otr_standard=25.0,
        package_surface_area_m2=0.04,
        product_dry_weight_g=150.0,
        initial_moisture_percent=2.0,
        critical_moisture_percent=5.0,
        headspace_volume_cc=120.0,
        ambient_temp_c=23.0
    )
    assert "limiting_days" in res
    assert "limiting_factor" in res
    assert "scientific_disclaimer" in res
    assert "Fickian diffusion" in res["scientific_disclaimer"]
    assert "moisture_result" in res
    assert "oxygen_result" in res

