import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Malformed experiment*")
warnings.filterwarnings(
    "ignore", message="Hint: Inferred schema contains integer column*"
)


df = pd.read_csv("data/train.csv")

df["datetime"] = pd.to_datetime(df["datetime"])
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek

target = "count"
drop_cols = ["datetime", "count", "casual", "registered"]
X = df.drop(columns=drop_cols)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("bike-demand-tracking")

# ✅ Hyperparameter grid 
param_grid = [
    {"n_estimators": 100, "max_depth": None, "random_state": 42, "n_jobs": -1},
    {"n_estimators": 200, "max_depth": None, "random_state": 42, "n_jobs": -1},
    {"n_estimators": 200, "max_depth": 20, "random_state": 42, "n_jobs": -1},
    {"n_estimators": 300, "max_depth": 20, "random_state": 42, "n_jobs": -1},
]

mlflow.sklearn.autolog()  #  autolog 

for params in param_grid:
    with mlflow.start_run(
        run_name=f"rf_tune_{params['n_estimators']}_depth_{params['max_depth']}"
    ):
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # ✅ Metrics per run
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        print("Trained:", params, "| RMSE:", rmse)