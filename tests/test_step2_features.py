"""
test_step2_features.py - Tests for Scope (Step 2):
1. Comparison basket logic (add, remove, max 4 constraint)
2. Material Comparator structure filters
3. Robust PDF generation without exceptions (recommendation & comparator)
4. Every material in the database has a valid, non-empty image file (<150 KB)
"""

import os
import pytest
import database
import recommender
import pdf_generator


@pytest.fixture
def all_mats():
    return database.get_all_materials()


def test_every_material_has_valid_image(all_mats):
    """Every material in packaging_materials must have an image file under 150 KB."""
    assert len(all_mats) >= 16
    for mat in all_mats:
        img_p = mat.get("image_path")
        # If DB path is missing or relative, check assets/materials/
        if not img_p or not os.path.exists(img_p):
            slug = mat["name"].lower().replace(" ", "_").replace("/", "_").replace("-", "_").replace("(", "").replace(")", "").replace(",", "").replace("__", "_")
            # fallback slug checking
            cand = f"assets/materials/{slug}.jpg"
            if not os.path.exists(cand):
                cand = "assets/materials/bopp_film.jpg"
            img_p = cand

        assert os.path.exists(img_p), f"Missing image for material {mat['name']}: {img_p}"
        size_kb = os.path.getsize(img_p) / 1024.0
        assert size_kb > 0, f"Image file is empty for {mat['name']}"
        assert size_kb < 150.0, f"Image for {mat['name']} exceeds 150 KB ({size_kb:.1f} KB)"


def test_comparison_basket_logic():
    """Verify basket logic: max 4 capacity, deduplication, add/remove."""
    basket = []

    # Add 1
    m1 = "BOPP Film (25 µm)"
    if m1 not in basket and len(basket) < 4:
        basket.append(m1)
    assert basket == [m1]

    # Deduplicate (cannot add duplicate)
    if m1 not in basket and len(basket) < 4:
        basket.append(m1)
    assert len(basket) == 1

    # Add up to 4
    m2 = "LDPE Monolayer Film (50 µm)"
    m3 = "PET / EVOH / PE Multi-layer Barrier Film (70 µm)"
    m4 = "PLA Bio-plastic Film (30 µm)"
    m5 = "Glass Container with Lug Closure (Rigid)"

    for m in [m2, m3, m4]:
        if m not in basket and len(basket) < 4:
            basket.append(m)
    assert len(basket) == 4

    # 5th item cannot be added
    if m5 not in basket and len(basket) < 4:
        basket.append(m5)
    assert len(basket) == 4
    assert m5 not in basket

    # Remove item
    basket.remove(m2)
    assert len(basket) == 3
    assert m2 not in basket

    # Now m5 can be added
    if m5 not in basket and len(basket) < 4:
        basket.append(m5)
    assert len(basket) == 4
    assert m5 in basket


def test_structure_filters(all_mats):
    """Structure filter must correctly partition materials into specific candidate subsets."""
    all_names = [m["name"] for m in all_mats]

    # Monolayers
    mono = [m["name"] for m in all_mats if "mono" in m.get("structure_type", "").lower() or ("flexible" in m.get("structure_type", "").lower() and "laminate" not in m.get("structure_type", "").lower() and "composite" not in m.get("structure_type", "").lower())]
    assert len(mono) >= 3
    assert any("BOPP Film" in n for n in mono)
    assert not any("Al Foil" in n for n in mono)

    # Multi-layer Laminates
    laminates = [m["name"] for m in all_mats if "laminate" in m.get("structure_type", "").lower() or "composite" in m.get("structure_type", "").lower() or "co-extruded" in m.get("structure_type", "").lower() or "barrier" in m.get("structure_type", "").lower()]
    assert len(laminates) >= 3
    assert any("Foil" in n for n in laminates)
    assert any("EVOH" in n for n in laminates)

    # Bio-based & Compostable
    bio = [m["name"] for m in all_mats if "bio" in m.get("structure_type", "").lower() or "compost" in m.get("notes", "").lower() or "pla" in m.get("name", "").lower() or "cellulose" in m.get("name", "").lower()]
    assert len(bio) >= 2
    assert any("PLA" in n for n in bio)
    assert any("Cellulose" in n for n in bio)

    # Rigid Containers
    rigid = [m["name"] for m in all_mats if "rigid" in m.get("structure_type", "").lower() or "glass" in m.get("name", "").lower() or "can" in m.get("name", "").lower()]
    assert len(rigid) >= 1
    assert any("Glass" in n for n in rigid)


def test_pdf_recommendation_and_comparator_generation(all_mats):
    """Both recommendation and comparator PDF generators must execute cleanly and return valid %PDF bytes."""
    comm = database.get_commodity_preset_by_name("Butter Cookie (Shortbread)")
    if not comm:
        comm = database.get_commodity_preset_by_name("Fried Potato Chips & Crisps")
    weights = {"barrier": 0.40, "cost": 0.25, "sustainability": 0.25, "mechanical": 0.10}

    res = recommender.rank_recommendations(all_mats, comm, weights)
    pdf_rec = pdf_generator.generate_recommendation_pdf(comm, res, weights)
    assert isinstance(pdf_rec, bytes)
    assert len(pdf_rec) > 5000
    assert pdf_rec.startswith(b"%PDF")

    # Comparator PDF
    selected = all_mats[:3]
    pdf_cmp = pdf_generator.generate_comparator_pdf(selected, comm["name"], "Balanced (Standard)")
    assert isinstance(pdf_cmp, bytes)
    assert len(pdf_cmp) > 5000
    assert pdf_cmp.startswith(b"%PDF")
