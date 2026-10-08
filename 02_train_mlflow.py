import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# 1) Load data
df = pd.read_csv("data/train.csv")

# 2) Basic feature engineering from datetime
df["datetime"] = pd.to_datetime(df["datetime"])
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek

# 3) Define target + features
# IMPORTANT: casual & registered are components of count -> leakage -> remove them
target = "count"
drop_cols = ["datetime", "count", "casual", "registered"]
X = df.drop(columns=drop_cols)
y = df[target]

# 4) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) Model params (we will log these)
params = {
        "n_estimators": 200,
        "random_state": 42,
        "n_jobs": -1
    }

# 6) Train
trainedmodel = RandomForestRegressor(**params)
trainedmodel.fit(X_train, y_train)

# 7) Predict & evaluate
pred = trainedmodel.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)


# 8) Create a simple plot artifact (Actual vs Pred)
plt.figure()
plt.scatter(y_test, pred)
plt.xlabel("Actual count")
plt.ylabel("Predicted count")
plt.title("Actual vs Predicted")
plot_path = "actual_vs_pred.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

# 9)  set local tracking folder
    # This creates an "mlruns" folder in your project directory.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 10) set experiment (groups runs)
mlflow.set_experiment("bike-demand-tracking")


# 11) Start MLflow run (this is ONE training attempt)
with mlflow.start_run(run_name="rf_baseline"):
    
    # Log parameters
    mlflow.log_params(params)

    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    print("==== MLflow Run Results ====")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    # Log artifacts
    mlflow.log_artifact(plot_path)

    
    # This makes the model loadable via MLflow .
    mlflow.sklearn.log_model(
    trainedmodel, 
    artifact_path="model", 
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

    #  add tags (extra info)
    mlflow.set_tag("dataset", "bike_sharing_train.csv")
    mlflow.set_tag("model_type", "RandomForestRegressor")

    # Logging Model
    mlflow.sklearn.log_model(trainedmodel, "model", serialization_format="pickle")