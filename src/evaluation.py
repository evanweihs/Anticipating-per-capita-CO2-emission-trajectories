# ============================================================
# Evaluation utilities: tables & plots (from notebook only)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ============================================================
# ===================== METRICS ===============================
# ============================================================

def eval_model(y_true, y_pred, label):
    """
    Print RMSE, MAE and R2 (as in notebook).
    """
    print(label)
    print("RMSE :", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE  :", mean_absolute_error(y_true, y_pred))
    print("R2   :", r2_score(y_true, y_pred))
    print("-" * 40)


# ============================================================
# ====================== TABLES ===============================
# ============================================================

# ------------------------------------------------------------
# 1. PCA variance explained table
# ------------------------------------------------------------
def pca_variance_table(eigvals, n_components=2):
    """
    Table of variance explained by PCA components.
    """
    var_exp = eigvals / eigvals.sum()
    df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(var_exp))],
        "Variance_explained": var_exp
    })
    return df.head(n_components)


# ------------------------------------------------------------
# 2. Country-level prediction table (RF or GB)
# ------------------------------------------------------------
def country_prediction_table(df, y_true_col, y_pred_col):
    """
    Country-level mean table: y_true vs y_pred.
    """
    return (
        df
        .groupby(["Country", "ISO3"], as_index=False)
        .agg(
            y_true=(y_true_col, "mean"),
            y_pred=(y_pred_col, "mean")
        )
    )


# ------------------------------------------------------------
# 3. RF vs GB comparison table
# ------------------------------------------------------------
def rf_gb_comparison_table(df):
    """
    Table with country-level mean predictions:
    y_true, y_pred_rf, y_pred_gb
    """
    return (
        df
        .groupby(["Country", "ISO3"], as_index=False)
        .agg(
            y_true=("co2_per_capita", "mean"),
            y_pred_rf=("co2_pred_rf", "mean"),
            y_pred_gb=("co2_pred_gb", "mean")
        )
    )


# ------------------------------------------------------------
# 4. K-Means cluster table
# ------------------------------------------------------------
def kmeans_cluster_table(df_country):
    """
    Final clustering table: Country, co2_mean, cluster, group label.
    """
    return df_country[[
        "Country",
        "co2_mean",
        "cluster_mode",
        "co2_group"
    ]].copy()


# ============================================================
# ======================= PLOTS ===============================
# ============================================================

# ------------------------------------------------------------
# 5. PCA scatter – all observations
# ------------------------------------------------------------
def plot_pca_scatter(df_pca):
    plt.figure(figsize=(10, 7))
    plt.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (standardized) – all observations")
    plt.grid(True)
    plt.show()


# ------------------------------------------------------------
# 6. PCA scatter – country means
# ------------------------------------------------------------
def plot_pca_country_means(df_pca_mean):
    plt.figure(figsize=(12, 8))
    plt.scatter(df_pca_mean["PC1"], df_pca_mean["PC2"], s=80)

    for i in range(len(df_pca_mean)):
        plt.text(
            df_pca_mean["PC1"].iloc[i],
            df_pca_mean["PC2"].iloc[i],
            df_pca_mean["Country"].iloc[i],
            fontsize=9
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (standardized) – country means")
    plt.grid(True)
    plt.show()


# ------------------------------------------------------------
# 7. K-Means clusters on PCA
# ------------------------------------------------------------
def plot_kmeans_pca(df_pca_mean):
    plt.figure(figsize=(12, 8))
    plt.scatter(
        df_pca_mean["PC1"],
        df_pca_mean["PC2"],
        c=df_pca_mean["cluster"],
        s=80
    )

    for i in range(len(df_pca_mean)):
        plt.text(
            df_pca_mean["PC1"].iloc[i],
            df_pca_mean["PC2"].iloc[i],
            df_pca_mean["Country"].iloc[i],
            fontsize=9
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("K-Means clustering on PCA")
    plt.grid(True)
    plt.show()
