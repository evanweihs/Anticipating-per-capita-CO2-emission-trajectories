# =====================================================
# Dynamic Random Forest with lagged variables
# Future CO2 per capita projection
# =====================================================

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# -----------------------------------------------------
# Utilities
# -----------------------------------------------------

def _build_lags(df, vars_lag):
    df = df.sort_values(["ISO3", "Year"]).copy()
    for v in vars_lag:
        df[f"{v}_lag1"] = df.groupby("ISO3")[v].shift(1)
    return df


def _classify_trajectory(delta, eps):
    if delta > eps:
        return "increase"
    elif delta < -eps:
        return "decrease"
    else:
        return "stable"


# -----------------------------------------------------
# Main function
# -----------------------------------------------------

def run_future_projection(
    df_final,
    start_year=2024,
    end_year=2030,
    epsilon=0.02,
    random_state=42,
):
    """
    Dynamic Random Forest with lagged variables and recursive projection.

    Parameters
    ----------
    df_final : pd.DataFrame
        Clean country-year panel (up to 2023).
    start_year, end_year : int
        Projection horizon.
    epsilon : float
        Threshold for trajectory classification.
    random_state : int
        Random seed.

    Returns
    -------
    df_forecast : pd.DataFrame
        Long-format projections (ISO3, Country, Year, co2_pred).
    df_trajectory : pd.DataFrame
        Country-level trajectory classification.
    """

    # -------------------------------------------------
    # 1. Select available variables and build lags
    # -------------------------------------------------
    lag_vars_all = [
        "co2_per_capita", "gdp_per_capita",
        "Coal", "Oil", "Gas", "Nuclear",
        "Hydro", "Wind", "Solar", "Other"
    ]

    lag_vars = [v for v in lag_vars_all if v in df_final.columns]

    if len(lag_vars) < 2:
        raise ValueError("Not enough variables available for lagged model.")

    df_ml = _build_lags(df_final, lag_vars)
    lag_cols = [f"{v}_lag1" for v in lag_vars]

    df_ml = df_ml.dropna(subset=lag_cols)

    X_cols = lag_cols
    y_col = "co2_per_capita"

    # -------------------------------------------------
    # 2. Temporal split (strict)
    # -------------------------------------------------
    train = df_ml[df_ml["Year"] <= 2018]
    val   = df_ml[(df_ml["Year"] >= 2019) & (df_ml["Year"] <= 2021)]
    test  = df_ml[df_ml["Year"] >= 2022]

    X_train, y_train = train[X_cols], train[y_col]
    X_val,   y_val   = val[X_cols],   val[y_col]
    X_test,  y_test  = test[X_cols],  test[y_col]

    # -------------------------------------------------
    # 3. Train dynamic Random Forest
    # -------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # -------------------------------------------------
    # 4. Recursive projection
    # -------------------------------------------------
    cols_last_obs = ["Country"] + lag_vars

    last_obs = (
        df_final[df_final["Year"] == 2023]
        .set_index("ISO3")[cols_last_obs]
    )

    state = last_obs.copy()
    results = []

    for year in range(start_year, end_year + 1):
        X_y = pd.DataFrame(
            {f"{v}_lag1": state[v] for v in lag_vars},
            index=state.index
        )[X_cols]

        co2_pred = rf.predict(X_y)

        results.append(
            pd.DataFrame({
                "ISO3": state.index,
                "Country": state["Country"].values,
                "Year": year,
                "co2_pred": co2_pred
            })
        )

        state = state.copy()
        state["co2_per_capita"] = co2_pred

    df_forecast = pd.concat(results, ignore_index=True)

    # -------------------------------------------------
    # 5. Trajectory classification (2024 → 2030)
    # -------------------------------------------------
    c_start = (
        df_forecast[df_forecast["Year"] == start_year][["Country", "co2_pred"]]
        .rename(columns={"co2_pred": "co2_start"})
    )

    c_end = (
        df_forecast[df_forecast["Year"] == end_year][["Country", "co2_pred"]]
        .rename(columns={"co2_pred": "co2_end"})
    )

    df_trajectory = c_start.merge(c_end, on="Country", how="inner")
    df_trajectory["delta_pct"] = (
        (df_trajectory["co2_end"] - df_trajectory["co2_start"])
        / df_trajectory["co2_start"]
    )

    df_trajectory["trajectory"] = df_trajectory["delta_pct"].apply(
        lambda d: _classify_trajectory(d, epsilon)
    )

    df_trajectory = df_trajectory.sort_values("delta_pct")

    return df_forecast, df_trajectory
