"""
test_analytics_and_ml.py - Tests for analytics visualizations, clustering, and ML pipelines
"""

import pytest
import database
import analytics
import ml_engine


@pytest.fixture
def sample_materials():
    return database.get_all_materials()[:3]


def test_radar_missing_data_warning(sample_materials):
    """Radar chart generation should flag unmeasured mechanical data without failing."""
    # Temporarily remove tensile strength from first material
    mats = [dict(m) for m in sample_materials]
    mats[0]["tensile_strength_mpa"] = None

    fig, warnings = analytics.create_radar_comparison_chart(mats)
    assert fig is not None
    assert len(warnings) >= 1
    assert any("Tensile strength data is unmeasured" in w for w in warnings)


def test_ml_clustering_and_pca():
    """K-Means clustering and PCA must run on the materials database and return clusters."""
    df = database.get_all_materials_df()
    clustered_df, fig, stats = ml_engine.perform_material_clustering(df, n_clusters=3)

    assert "cluster" in clustered_df.columns
    assert "pca_1" in clustered_df.columns
    assert "pca_2" in clustered_df.columns
    assert stats["total_variance_explained"] > 50.0
    assert fig is not None


def test_supervised_baseline_training():
    """Supervised baseline classifier must fit and return feature importances."""
    clf, fig, summary = ml_engine.build_supervised_baseline()

    assert clf is not None
    assert summary["train_accuracy"] > 70.0
    assert len(summary["classes"]) >= 3
    assert fig is not None


def test_pca_interpretations_present():
    """PCA stats dictionary must contain plain-language physical axis interpretations."""
    df = database.get_all_materials_df()
    _, _, stats = ml_engine.perform_material_clustering(df, n_clusters=3)

    assert "pc1_interpretation" in stats
    assert "pc2_interpretation" in stats
    assert len(stats["pc1_interpretation"]) > 20
    assert len(stats["pc2_interpretation"]) > 20
    assert "barrier" in stats["pc1_interpretation"].lower()


def test_supervised_baseline_holdout_accuracy():
    """Supervised baseline must report holdout test accuracy and transparent split ratio."""
    _, _, summary = ml_engine.build_supervised_baseline()

    assert "test_accuracy" in summary
    assert "train_accuracy" in summary
    assert "split_ratio" in summary
    assert "disclaimer" in summary
    assert 0.0 <= summary["test_accuracy"] <= 100.0
    assert "80% Train" in summary["split_ratio"]

