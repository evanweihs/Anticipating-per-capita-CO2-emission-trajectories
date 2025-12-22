# ============================================================
# Results utilities
# Tables & figures for analysis and reporting
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# ======================= TABLES ===============================
# ============================================================

def country_prediction_table(df, y_true_col, y_pred_col):
    """
    Country-level mean prediction table.

    Parameters
    ----------
    df : pd.DataFrame
    y_true_col : str
    y_pred_col : str

    Returns
    -------
    pd.DataFrame
        Country, ISO3, y_true, y_pred
    """
    return (
        df
        .groupby(["Country", "ISO3"], as_index=False)
        .agg(
            y_true=(y_true_col, "mean"),
            y_pred=(y_pred_col, "mean"),
        )
    )


def rf_gb_comparison_table(df_rf, df_gb):
    """
    Country-level RF vs GB comparison table.

    Parameters
    ----------
    df_rf : pd.DataFrame
        Must contain Country, ISO3, y_true, co2_pred_rf
    df_gb : pd.DataFrame
        Must contain Country, ISO3, co2_per_capita, co2_predicted

    Returns
    -------
    pd.DataFrame
    """
    df = df_rf.merge(
        df_gb,
        on=["Country", "ISO3"],
        how="inner",
        suffixes=("_rf", "_gb")
    )

    return df[[
        "Country",
        "ISO3",
        "y_true",
        "co2_pred_rf",
        "co2_predicted"
    ]].rename(columns={
        "co2_pred_rf": "y_pred_rf",
        "co2_predicted": "y_pred_gb"
    })


def kmeans_cluster_table(df_country):
    """
    Final clustering table.

    Returns
    -------
    pd.DataFrame
    """
    return df_country[[
        "Country",
        "co2_mean",
        "cluster_mode",
        "co2_group"
    ]].copy()


# ============================================================
# ======================= FIGURES ==============================
# ============================================================

def plot_pca_scatter(df_pca, save_path=None):
    """
    PCA scatter plot – all observations.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (standardized) – all observations")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_pca_country_means(df_pca_mean, save_path=None):
    """
    PCA scatter plot – country means.
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(df_pca_mean["PC1"], df_pca_mean["PC2"], s=80)

    for _, row in df_pca_mean.iterrows():
        plt.text(row["PC1"], row["PC2"], row["Country"], fontsize=9)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (standardized) – country means")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_kmeans_pca(df_pca_mean, save_path=None):
    """
    K-Means clusters on PCA space.
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(
        df_pca_mean["PC1"],
        df_pca_mean["PC2"],
        c=df_pca_mean["cluster"],
        s=80
    )

    for _, row in df_pca_mean.iterrows():
        plt.text(row["PC1"], row["PC2"], row["Country"], fontsize=9)

    plt.xlabel("PC1")
    plt.ylabel("PC2
