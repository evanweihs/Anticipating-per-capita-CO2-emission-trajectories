# =====================================================
# Random Forest – CO2 per capita prediction
# =====================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =====================================================
# 1. Load clean data
# =====================================================

df_final = pd.read_csv("data/clean/df_final.csv")


# =====================================================
# 2. Lagged features construction
# =====================================================

df_ml = df_final.sort_values(["ISO3", "Year"]).copy()

lag_vars = [
    "co2_per_capita", "gdp_per_capita",
    "Coal", "Oil", "Gas", "Nuclear",
    "Hydro", "Wind", "Solar", "Other"
]

for var in lag_vars:
    df_ml[f"{var}_lag1"] = (
        df_ml
        .groupby("ISO3")[var]
        .shift(1)
    )

df_ml = df_ml.dropna(subset=[f"{v}_lag1" for v in lag_vars])


# =====================================================
# 3. Features and target
# =====================================================

features_lag = [
    "co2_per_capita_lag1", "gdp_per_capita_lag1",
    "Coal_lag1", "Oil_lag1", "Gas_lag1", "Nuclear_lag1",
    "Hydro_lag1", "Wind_lag1", "Solar_lag1", "Other_lag1"
]

X = df_ml[features_lag]
y = df_ml["co2_per_capita"]


# =====================================================
# 4. Temporal split (STRICT)
# =====================================================

train = df_ml[df_ml["Year"] <= 2018]
val   = df_ml[(df_ml["Year"] >= 2019) & (df_ml["Year"] <= 2021)]
test  = df_ml[df_ml["Year"] >= 2022]

X_train, y_train = train[features_lag], train["co2_per_capita"]
X_val,   y_val   = val[features_lag],   val["co2_per_capita"]
X_test,  y_test  = test[features_lag],  test["co2_per_capita"]


# =====================================================
# 5. Model training
# =====================================================

rf = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# =====================================================
# 6. Evaluation function
# =====================================================

def eval_model(y_true, y_pred, label):
    print(label)
    print("RMSE :", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE  :", mean_absolute_error(y_true, y_pred))
    print("R2   :", r2_score(y_true, y_pred))
    print("-" * 40)


# =====================================================
# 7. Validation & test
# =====================================================

y_val_pred = rf.predict(X_val)
eval_model(y_val, y_val_pred, "VALIDATION 2019–2021")

y_test_pred = rf.predict(X_test)
eval_model(y_test, y_test_pred, "TEST FUTUR 2022–2023")
