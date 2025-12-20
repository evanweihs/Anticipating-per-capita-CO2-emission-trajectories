# =====================================================
# Recursive future projection 2024–2030
# =====================================================

import pandas as pd


# =====================================================
# 1. Base state (last observed year)
# =====================================================

df_final = pd.read_csv("data/clean/df_final.csv")
df_proj_base = df_final.sort_values(["ISO3", "Year"]).copy()

last_obs = (
    df_proj_base[df_proj_base["Year"] == 2023]
    .set_index("ISO3")
)[[
    "Country", "co2_per_capita", "gdp_per_capita",
    "Coal", "Oil", "Gas", "Nuclear",
    "Hydro", "Wind", "Solar", "Other"
]]

state = last_obs.copy()


# =====================================================
# 2. Recursive projection
# =====================================================

years_future = list(range(2024, 2031))
results = []

for year in years_future:

    X_y = pd.DataFrame({
        "co2_per_capita_lag1": state["co2_per_capita"],
        "gdp_per_capita_lag1": state["gdp_per_capita"],
        "Coal_lag1": state["Coal"],
        "Oil_lag1": state["Oil"],
        "Gas_lag1": state["Gas"],
        "Nuclear_lag1": state["Nuclear"],
        "Hydro_lag1": state["Hydro"],
        "Wind_lag1": state["Wind"],
        "Solar_lag1": state["Solar"],
        "Other_lag1": state["Other"],
    }, index=state.index)

    X_y = X_y[features_lag]

    co2_pred = rf.predict(X_y)

    tmp = pd.DataFrame({
        "ISO3": state.index,
        "Country": state["Country"].values,
        "Year": year,
        "co2_pred": co2_pred
    })

    results.append(tmp)

    state = state.copy()
    state["co2_per_capita"] = co2_pred


df_future = pd.concat(results, ignore_index=True)
df_future.to_csv("results/future_projection_rf_2024_2030.csv", index=False)
