# ============================================================
# Evaluation utilities (STRICT)
# Quantitative evaluation only (no plots, no saving)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. Regression metrics
# ============================================================

def regression_metrics(y_true, y_pred):
    """
    Compute regression performance metrics.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        rmse, mae, r2
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),}


# ============================================================
# 2. PCA variance explained
# ============================================================

def pca_variance_table(eigvals, n_components=2):
    """
    Build a table of variance explained by PCA components.

    Parameters
    ----------
    eigvals : array-like
        Eigenvalues from PCA.
    n_components : int
        Number of components to keep in the output table.

    Returns
    -------
    pd.DataFrame
        Columns: Component, Variance_explained
    """
    eigvals = np.asarray(eigvals, dtype=float)
    var_exp = eigvals / eigvals.sum()

    return pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(var_exp))],
        "Variance_explained": var_exp
    }).head(n_components)


# ============================================================
# 3. RF vs GB comparison (country level)
# ============================================================

def rf_gb_comparison_table(df_rf_country, df_gb_country):
    """
    Country-level comparison table:
    y_true, y_pred_rf, y_pred_gb

    Parameters
    ----------
    df_rf_country : pd.DataFrame
        Must contain: Country, ISO3, y_true, co2_pred_rf
    df_gb_country : pd.DataFrame
        Must contain: Country, ISO3, co2_per_capita, co2_predicted

    Returns
    -------
    pd.DataFrame
        Columns: Country, ISO3, y_true, y_pred_rf, y_pred_gb
    """
    rf_required = {"Country", "ISO3", "y_true", "co2_pred_rf"}
    gb_required = {"Country", "ISO3", "co2_per_capita", "co2_predicted"}

    if not rf_required.issubset(df_rf_country.columns):
        missing = sorted(list(rf_required - set(df_rf_country.columns)))
        raise ValueError(f"df_rf_country missing columns: {missing}")

    if not gb_required.issubset(df_gb_country.columns):
        missing = sorted(list(gb_required - set(df_gb_country.columns)))
        raise ValueError(f"df_gb_country missing columns: {missing}")

    df_rf = df_rf_country.rename(columns={"co2_pred_rf": "y_pred_rf"}).copy()
    df_gb = df_gb_country.rename(columns={
        "co2_per_capita": "y_true_gb",
        "co2_predicted": "y_pred_gb"
    }).copy()

    # Merge on Country/ISO3
    out = df_rf.merge(
        df_gb[["Country", "ISO3", "y_pred_gb"]],
        on=["Country", "ISO3"],
        how="inner"
    )

    # Final column order
    out = out[["Country", "ISO3", "y_true", "y_pred_rf", "y_pred_gb"]]

    return out
