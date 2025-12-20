# =====================================================
# K-Means clustering on PCA (exploratory analysis)
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# =====================================================
# 1. Load data
# =====================================================

df_final = pd.read_csv("data/clean/df_final.csv")
df_pca_mean = pd.read_csv("data/pca/df_pca_mean.csv")
co2_mean_country = pd.read_csv("data/clean/co2_mean_country.csv")


# =====================================================
# 2. K-Means on PCA space
# =====================================================

kmeans = KMeans(n_clusters=3, random_state=42)

df_pca_mean["cluster"] = kmeans.fit_predict(
    df_pca_mean[["PC1", "PC2"]]
)

df_pca_mean = df_pca_mean.merge(
    co2_mean_country,
    on="Country",
    how="left"
)


# =====================================================
# 3. Visualisation (clusters on PCA)
# =====================================================

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
plt.title("K-means clustering on PCA (PC1 vs PC2)")
plt.grid(True)
plt.savefig("results/kmeans_pca_clusters.png", dpi=300, bbox_inches="tight")
plt.show()


# =====================================================
# 4. Cluster assignment over full PCA trajectories
# =====================================================

df_pca = pd.read_csv("data/pca/df_pca.csv")

kmeans = KMeans(n_clusters=3, random_state=42)
df_pca["cluster"] = kmeans.fit_predict(df_pca[["PC1", "PC2"]])


# =====================================================
# 5. Country-level aggregation
# =====================================================

df_country = (
    df_pca
    .groupby("Country")
    .agg(
        co2_mean=("co2_per_capita", "mean"),
        cluster_mode=("cluster", lambda x: x.mode()[0])
    )
    .reset_index()
)


# =====================================================
# 6. Non-arbitrary cluster labeling
# =====================================================

cluster_order = (
    df_country
    .groupby("cluster_mode")["co2_mean"]
    .mean()
    .sort_values()
)

cluster_mapping = {
    cluster_order.index[0]: "Low CO2",
    cluster_order.index[1]: "Medium CO2",
    cluster_order.index[2]: "High CO2",
}

df_country["co2_group"] = df_country["cluster_mode"].map(cluster_mapping)


# =====================================================
# 7. Save results
# =====================================================

df_country.to_csv(
    "results/country_co2_clusters.csv",
    index=False
)
