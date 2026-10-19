"""
ml_engine.py - Machine Learning Archetype Clustering and Supervised Classification
Implements:
1. Unsupervised K-Means clustering and PCA dimensionality reduction on material barrier features.
2. Supervised baseline classification with feature importance analysis and transparent disclosures.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from recommender import RECYCLABILITY_SCORE_MAP


def perform_material_clustering(materials_df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, go.Figure, Dict[str, Any]]:
    """
    Applies K-Means clustering and 2D PCA on materials' physical and economic feature vectors.
    """
    df = materials_df.copy()

    # Feature extraction with log transformation for skewed transmission rates
    df["log_wvtr"] = np.log10(np.clip(df["wvtr_g_m2_day"], 0.001, 1000.0))
    df["log_otr"] = np.log10(np.clip(df["otr_cc_m2_day_atm"], 0.001, 50000.0))
    df["recyclability_num"] = df["recyclability_status"].map(lambda x: RECYCLABILITY_SCORE_MAP.get(x, 40.0))

    feature_cols = ["log_wvtr", "log_otr", "nominal_thickness_um", "cost_index_usd_per_kg", "recyclability_num"]
    X = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=min(n_clusters, len(df)), random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    df["cluster_label"] = df["cluster"].apply(lambda c: f"Cluster Archetype {c+1}")

    # 2D PCA Projection
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df["pca_1"] = pca_coords[:, 0]
    df["pca_2"] = pca_coords[:, 1]
    var_explained = pca.explained_variance_ratio_

    # Plotly Scatter Plot
    fig = px.scatter(
        df,
        x="pca_1",
        y="pca_2",
        color="cluster_label",
        hover_name="name",
        hover_data={
            "structure_type": True,
            "wvtr_g_m2_day": ":.2f",
            "otr_cc_m2_day_atm": ":.1f",
            "cost_index_usd_per_kg": ":$.2f",
            "recyclability_status": True,
            "pca_1": False,
            "pca_2": False
        },
        title=f"PCA Material Space & K-Means Clusters (Explained Variance: {var_explained.sum()*100:.1f}%)",
        labels={
            "pca_1": f"Principal Component 1 ({var_explained[0]*100:.1f}% variance)",
            "pca_2": f"Principal Component 2 ({var_explained[1]*100:.1f}% variance)"
        }
    )

    fig.update_traces(marker=dict(size=14, line=dict(width=1.5, color="#334155")))
    fig.update_layout(
        height=480,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )

    stats = {
        "n_clusters": min(n_clusters, len(df)),
        "explained_variance_ratio": [round(float(v), 4) for v in var_explained],
        "total_variance_explained": round(float(var_explained.sum() * 100.0), 2),
        "pc1_interpretation": (
            "PC1 (Horizontal Axis): Primary barrier transmission gradient. "
            "Separates high-barrier metallized films/foils (negative values) from permeable polyolefins and EMAP films (positive values)."
        ),
        "pc2_interpretation": (
            "PC2 (Vertical Axis): Structural stiffness and mass density. "
            "Separates thin flexible films from thick rigid glass and composite paperboard structures."
        )
    }

    return df, fig, stats


def build_supervised_baseline() -> Tuple[RandomForestClassifier, go.Figure, Dict[str, Any]]:
    """
    Constructs a baseline Random Forest classifier mapping commodity sensitivity profiles
    to packaging category archetypes.
    Prominently documents the synthetic rule-matrix generation process.
    """
    np.random.seed(42)

    # Generate transparent rule-aligned scenario dataset (200 instances)
    # Features:
    # 0: Moisture Sensitivity (1: Low, 2: Mod, 3: High, 4: Extreme)
    # 1: Oxygen Sensitivity (1: Low, 2: Mod, 3: High, 4: Critical)
    # 2: Fat Content (0: Zero, 1: Low, 2: Med, 3: High, 4: Very High)
    # 3: Target Shelf Life Days (7 to 365)
    # 4: Storage Temperature °C (-18 to 38)
    # 5: Is Fresh Produce (0 or 1)

    records = []
    labels = []

    for _ in range(250):
        is_fresh = np.random.choice([0, 1], p=[0.85, 0.15])
        if is_fresh == 1:
            m_sens = 1
            o_sens = np.random.choice([1, 2])
            fat = 0
            shelf_life = np.random.randint(5, 21)
            temp = np.random.choice([4.0, 10.0])
            label = "Breathable EMAP Film"
        else:
            m_sens = np.random.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2])
            o_sens = np.random.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15])
            fat = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.3, 0.2, 0.1])
            shelf_life = np.random.choice([30, 90, 180, 365])
            temp = np.random.choice([-18.0, 4.0, 23.0, 35.0])

            # Deterministic food science logic
            if m_sens >= 4 and o_sens >= 4:
                label = "Ultra-Barrier Foil / Retort Laminate"
            elif o_sens >= 3 and shelf_life >= 90:
                label = "High Oxygen Barrier (EVOH / Met-PET)"
            elif m_sens >= 3:
                label = "Moisture-Barrier Polyolefin (HDPE / BOPP)"
            elif shelf_life <= 30 and m_sens <= 2:
                label = "Bio-based / Compostable Film"
            else:
                label = "Standard Commodity Monolayer (LDPE / Paper)"

        records.append([m_sens, o_sens, fat, shelf_life, temp, is_fresh])
        labels.append(label)

    feature_names = [
        "Moisture Sensitivity",
        "Oxygen Sensitivity",
        "Fat Content",
        "Target Shelf Life (Days)",
        "Storage Temp (°C)",
        "Is Fresh Respiring"
    ]

    X = pd.DataFrame(records, columns=feature_names)
    y = np.array(labels)

    # Rigorous 80/20 train/test split to evaluate true holdout generalization
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = float(clf.score(X_train, y_train) * 100.0)
    test_acc = float(clf.score(X_test, y_test) * 100.0)

    # Feature Importance Plot
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Supervised Baseline: Feature Importance (Mean Decrease in Impurity)",
        color="Importance",
        color_continuous_scale="Teal"
    )
    fig.update_layout(
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        coloraxis_showscale=False
    )

    training_summary = {
        "n_samples": len(records),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": len(np.unique(y)),
        "classes": list(np.unique(y)),
        "model_type": "Random Forest Classifier (50 Trees, max_depth=5)",
        "train_accuracy": round(train_acc, 2),
        "test_accuracy": round(test_acc, 2),
        "split_ratio": "80% Train / 20% Holdout Test",
        "disclaimer": (
            "Baseline classifier trained on rule-derived food sensitivity instances to validate "
            "feature hierarchy. Does not replace physical laboratory barrier measurement."
        )
    }

    return clf, fig, training_summary
