# ============================================================
# Results utilities
# Tables & figures saved to /results
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ======================= TABLES ===============================
# ============================================================

def save_table(df: pd.DataFrame, filename: str):
    """
    Save a DataFrame to results/tables.
    """
    path = TABLES_DIR / filename
    df.to_csv(path, index=False)
    print(f"Table saved: {path}")


def country_prediction_table(df, y_true_col, y_pred_col):
    """
    Country-level mean prediction table.
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

def plot_pca_scatter(df_pca):
    """
    PCA scatter plot – all observations.
    """
    path = FIGURES_DIR / "pca_scatter_all.png"

    plt.figure(figsize=(10, 7))
    plt.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (standardized) – all observations")
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved: {path}")


def plot_pca_country_means(df_pca_mean):
    """
    PCA scatter plot – country means.
    """
    path = FIGURES_DIR / "pca_country_means.png"

    plt.figure(figsize=(12, 8))
    plt.scatter(df_pca_mean["PC1"], df_pca_mean["PC2"], s=80)

    for _, row in df_pca_mean.iterrows():
        plt.text(row["PC1"], row["PC2"], row["Country"], fontsize=9)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (standardized) – country means")
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved: {path}")


def plot_kmeans_pca(df_pca_kmeans):
    """
    K-Means clusters on PCA space.

    Expected columns:
    - Country
    - PC1, PC2
    - cluster_mode
    """
    path = FIGURES_DIR / "kmeans_pca_clusters.png"

    plt.figure(figsize=(12, 8))
    plt.scatter(
        df_pca_kmeans["PC1"],
        df_pca_kmeans["PC2"],
        c=df_pca_kmeans["cluster_mode"],
        s=80
    )

    for _, row in df_pca_kmeans.iterrows():
        plt.text(row["PC1"], row["PC2"], row["Country"], fontsize=9)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("K-Means clustering on PCA")
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved: {path}")


def plot_emission_trajectories_by_group(
    df_forecast,
    year_ref=2024,
    n_countries=8,
):
    """
    Plot and save CO2 per capita trajectories (2024–2030)
    for low, medium and high emitting countries.
    """

    # ---------------------------------------
    # Rank countries by reference year
    # ---------------------------------------
    ranking = (
        df_forecast[df_forecast["Year"] == year_ref]
        .sort_values("co2_pred")
    )

    groups = {
        "low_emitters": ranking.head(n_countries)["Country"].tolist(),
        "medium_emitters": ranking.iloc[n_countries:2*n_countries]["Country"].tolist(),
        "high_emitters": ranking.tail(n_countries)["Country"].tolist(),
    }

    # ---------------------------------------
    # Plot each group
    # ---------------------------------------
    for group_name, countries in groups.items():

        plt.figure(figsize=(8, 5))

        for country in countries:
            d = (
                df_forecast[df_forecast["Country"] == country]
                .sort_values("Year")
            )
            plt.plot(d["Year"], d["co2_pred"], marker="o", label=country)

        plt.title(f"{group_name.replace('_', ' ').title()} (2024–2030)")
        plt.xlabel("Year")
        plt.ylabel("CO₂ per capita (predicted)")
        plt.grid(True)
        plt.legend(fontsize=6)

        path = FIGURES_DIR / f"{group_name}_trajectories.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Figure saved: {path}")
