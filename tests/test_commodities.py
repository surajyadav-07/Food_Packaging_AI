"""
tests/test_commodities.py - Quality & Integrity Tests for Food Commodity Presets
Validates commodity profile count, sector coverage, required fields, image asset validity,
and fallback behavior.
"""

import os
import sqlite3
import pytest
from database import DB_FILE_PATH, get_all_commodity_presets
from app import get_food_image_path

REQUIRED_CATEGORIES = [
    "Bakery products",
    "Cereals and grains",
    "Dairy products",
    "Fruits and vegetables",
    "Pulses",
    "Snacks"
]

REQUIRED_FIELDS = [
    "name",
    "category",
    "moisture_sensitivity",
    "oxygen_sensitivity",
    "fat_content_level",
    "recommended_temp_c",
    "recommended_rh_percent",
    "target_shelf_life_days",
    "typical_spoilage_mode",
    "critical_aw"
]


@pytest.fixture
def commodity_list():
    """Load all commodity presets from the SQLite database."""
    items = get_all_commodity_presets(DB_FILE_PATH)
    assert len(items) > 0, "No commodity presets found in database"
    return items


def test_commodity_total_count(commodity_list):
    """Ensure at least 30 commodity presets are present."""
    assert len(commodity_list) >= 30, f"Expected at least 30 commodities, found {len(commodity_list)}"


def test_no_duplicate_commodity_names(commodity_list):
    """Ensure there are no duplicate commodity names."""
    names = [c["name"].strip().lower() for c in commodity_list]
    assert len(names) == len(set(names)), "Duplicate commodity names detected in database"


def test_required_categories_presence_and_minimum(commodity_list):
    """Ensure all 6 required categories exist and contain at least 4-5 commodities."""
    categories_found = {c["category"] for c in commodity_list}
    for req_cat in REQUIRED_CATEGORIES:
        assert req_cat in categories_found, f"Missing required food category: '{req_cat}'"
        cat_items = [c for c in commodity_list if c["category"] == req_cat]
        assert len(cat_items) >= 4, f"Category '{req_cat}' has fewer than 4 commodities ({len(cat_items)})"


def test_commodity_required_fields(commodity_list):
    """Ensure all required food-science fields are non-null and valid."""
    for c in commodity_list:
        name = c.get("name")
        for field in REQUIRED_FIELDS:
            val = c.get(field)
            assert val is not None, f"Commodity '{name}' has null value for required field '{field}'"
            if isinstance(val, str):
                assert len(val.strip()) > 0, f"Commodity '{name}' has empty string for field '{field}'"

        # Validate numeric ranges
        assert -30.0 <= c["recommended_temp_c"] <= 60.0, f"Invalid recommended_temp_c for '{name}': {c['recommended_temp_c']}"
        assert 10 <= c["recommended_rh_percent"] <= 100, f"Invalid recommended_rh_percent for '{name}': {c['recommended_rh_percent']}"
        assert 1 <= c["target_shelf_life_days"] <= 1000, f"Invalid target_shelf_life_days for '{name}': {c['target_shelf_life_days']}"
        if c.get("critical_aw") is not None:
            assert 0.05 <= c["critical_aw"] <= 1.0, f"Invalid critical_aw for '{name}': {c['critical_aw']}"


def test_commodity_image_files_exist_and_size(commodity_list):
    """Verify that every product's image file exists on disk and is under 150 KB."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for c in commodity_list:
        img_path = c.get("image_path")
        assert img_path is not None and len(img_path.strip()) > 0, f"Commodity '{c['name']}' has empty image_path"
        
        full_path = os.path.join(base_dir, img_path) if not os.path.isabs(img_path) else img_path
        assert os.path.exists(full_path), f"Image file not found on disk for '{c['name']}': {full_path}"
        
        file_size = os.path.getsize(full_path)
        assert file_size > 100, f"Image file for '{c['name']}' is suspiciously small ({file_size} bytes)"
        assert file_size <= 150 * 1024, f"Image file for '{c['name']}' exceeds 150 KB ({file_size} bytes)"


def test_image_lookup_and_fallback(commodity_list):
    """Test get_food_image_path resolves correctly for all presets and falls back gracefully."""
    for c in commodity_list:
        resolved = get_food_image_path(c["category"], c["name"], c.get("image_path", ""))
        assert resolved and os.path.exists(resolved), f"Failed to resolve image path for '{c['name']}'"

    # Test unknown fallback
    fallback = get_food_image_path("NonExistentCategory", "UnknownFoodProduct")
    assert fallback and os.path.exists(fallback), "Fallback image failed to resolve for unknown product"


def test_credits_file_exists_and_covers_products(commodity_list):
    """Verify CREDITS.md exists and documents the image assets."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    credits_path = os.path.join(base_dir, "CREDITS.md")
    assert os.path.exists(credits_path), "CREDITS.md does not exist"
    
    with open(credits_path, "r", encoding="utf-8") as f:
        credits_content = f.read()
    
    for c in commodity_list:
        assert c["name"] in credits_content, f"Commodity '{c['name']}' is not documented in CREDITS.md"
